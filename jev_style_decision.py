"""Jev-Style-0.8B-Decision-v3: typed decisions with transformers / PyTorch (CUDA, MPS, CPU).

Self-contained runtime for chaoliangUNSW/Jev-Style-0.8B-Decision-v3 (Apache-2.0). No dependency on any training code:
rendering, verdict readout and calibration are implemented below and reproduce the reference
implementation used for evaluation (see release_config.json -> "runtime_parity").

Calibration temperature: with no category (the default) probabilities use the global temperature of
readout_config.json (temperatures.global = 0.880); pass category=... (CLI --category, JSONL "category")
for the fitted group temperature of that category's family x question type x option-count bucket, or
temperature=... to override (1.0 = uncalibrated scores).
"""
# ----------------------------------------------------------------------------------------------
# Shared core (identical in jev_style_decision.py, jev_style_decision_gguf.py and
# jev_style_decision_mlx.py): input rendering, verdict readout, calibrated probabilities.
#
# Input layout ("macjev-render-v1"; token segments are encoded separately and concatenated):
#
#   State:\n<state>\n\n
#   Question [<type>]: <question>\nOptions:\n
#   - <option 1>\n ... - <option K>\n
#   Judge each option:\n
#   <option 1> ->\n ... <option K> ->\n
#
# Score of option k = logit(" yes") - logit(" no") at the k-th " ->" token (computed from the final
# hidden state and the tied embedding rows, float32). Probabilities = softmax(scores / T), where T is
# the calibration temperature shipped in readout_config.json:
#   * no category given (the default): T = temperatures.global (the file's global temperature);
#   * category="..." given: T = the fitted group temperature of (family of that category x question
#     type x option-count bucket), or temperatures.global when that group was not fitted;
#   * temperature=... given: that value (1.0 = uncalibrated scores).
# T is clamped to temperatures.clamp. Text inside the state or options is tokenised with special
# tokens disabled, so e.g. "<|im_end|>" in user text can never act as a control token.
#
# Budgets: whole input <= 25,600 tokens; question + options + readout ("head") <= 2,048 tokens.
# Larger inputs raise InputBudgetError. Nothing is ever truncated.
# ----------------------------------------------------------------------------------------------
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

MODEL_NAME = "Jev-Style-0.8B-Decision-v3"
TEMPLATE_VERSION = "macjev-render-v1"
READOUT_FORMAT = "macjev-readout-v1"
CONTEXT_LIMIT = 25_600          # state + question + options + readout
HARD_HEAD_MAX = 2048            # question + options + readout
QTYPES = ("choice", "score", "noul")
HERE = Path(__file__).resolve().parent

# calibration families (category prefix -> family), same table the temperatures were fitted with
FAMILY_BY_CATEGORY_PREFIX = (("typed_official", "typed"), ("typed_synthetic", "typed_synth"), ("general_", "general"),
                             ("intent", "intent"), ("nli", "nli"), ("theme_", "theme"), ("mac_", "mac"),
                             ("long_", "long"))


class InputBudgetError(ValueError):
    """The rendered input exceeds a token budget. Nothing was truncated."""


class QuestionError(ValueError):
    """The question/options are malformed."""


# -- questions ------------------------------------------------------------------------------------
def option_names(question):
    """Canonical option identifiers, in the order the probabilities are returned."""
    if not isinstance(question, dict):
        raise QuestionError("question must be a dict {'t', 'ins', 'crit'}")
    t, crit = question.get("t"), question.get("crit")
    if not isinstance(question.get("ins"), str) or not question["ins"].strip():
        raise QuestionError("question text ('ins') must be a non-empty string")
    if t == "choice":
        if not isinstance(crit, dict) or not crit:
            raise QuestionError("choice needs a non-empty dict {option name: description or None}")
        return [str(k) for k in crit]
    if t == "score":
        if not isinstance(crit, list) or not 2 <= len(crit) <= 10:
            raise QuestionError("score needs a list of 2..10 level descriptions")
        return [str(i) for i in range(len(crit))]
    if t == "noul":
        if crit is not None and not isinstance(crit, dict):
            raise QuestionError("noul criteria must be None or {'false': ..., 'true': ...}")
        return ["false", "true"]
    raise QuestionError(f"unknown question type {t!r} (expected one of {QTYPES})")


def make_question(question, options=None, qtype=None):
    """Build a typed question.

    * ``question`` already a dict {"t", "ins", "crit"}: validated and returned.
    * ``qtype="choice"`` (default when ``options`` is given): ``options`` = {name: description or None}
      or a list of names.
    * ``qtype="score"``: ``options`` = list of 2..10 level descriptions (level 0 first).
    * ``qtype="noul"`` (default when no options): a true/false statement; ``options`` may be
      {"false": "...", "true": "..."} to describe the two outcomes.
    """
    if isinstance(question, dict):
        q = dict(question)
    else:
        if qtype is None:
            qtype = "choice" if options is not None else "noul"
        if qtype == "choice":
            if isinstance(options, (list, tuple)):
                if len(set(map(str, options))) != len(options):
                    raise QuestionError("duplicate option names")
                crit = {str(o): None for o in options}
            else:
                crit = options
        elif qtype == "score":
            crit = list(options) if options is not None else None
        else:
            crit = options
        q = {"t": qtype, "ins": question, "crit": crit}
    option_names(q)
    return q


def serialize_state(state):
    """Strings pass through unchanged; any other JSON value is serialised (ensure_ascii=False)."""
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


def _criterion(value):
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


def render_options(question):
    t, crit = question["t"], question.get("crit")
    if t == "choice":
        return [k if v is None or v == "" else f"{k}: {_criterion(v)}" for k, v in crit.items()]
    if t == "score":
        return [f"level {i}: {_criterion(c)}" for i, c in enumerate(crit)]
    crit = crit or {}
    false_c, true_c = crit.get("false"), crit.get("true")
    return ["false: " + (_criterion(false_c) if false_c not in (None, "") else "no, the statement does not hold"),
            "true: " + (_criterion(true_c) if true_c not in (None, "") else "yes, the statement holds")]


# -- tokenizer + renderer -----------------------------------------------------------------------
class TextEncoder:
    """HF ``tokenizers`` tokenizer.json; no BOS/EOS, special tokens in text are split (never control tokens)."""

    def __init__(self, tokenizer_json):
        from tokenizers import Tokenizer
        self.tk = Tokenizer.from_file(str(tokenizer_json))
        self.tk.encode_special_tokens = True

    def __call__(self, text):
        return self.tk.encode(text, add_special_tokens=False).ids


class Rendered:
    __slots__ = ("ids", "prefix_len", "slots", "names", "head_tokens")

    def __init__(self, ids, prefix_len, slots, names, head_tokens):
        self.ids, self.prefix_len, self.slots, self.names, self.head_tokens = ids, prefix_len, slots, names, head_tokens


class Renderer:
    def __init__(self, encode, readout_cfg, max_len=CONTEXT_LIMIT, head_max=HARD_HEAD_MAX):
        if readout_cfg.get("format") != READOUT_FORMAT or readout_cfg.get("template") != TEMPLATE_VERSION:
            raise ValueError("readout_config.json is not a macjev-readout-v1 / macjev-render-v1 config")
        if readout_cfg.get("readout") != "verdict":
            raise ValueError("this runtime implements the verdict readout only")
        if not 0 < int(max_len) <= CONTEXT_LIMIT:
            raise ValueError(f"max_len must be in 1..{CONTEXT_LIMIT}")
        if not 0 < int(head_max) <= HARD_HEAD_MAX:
            raise ValueError(f"head_max must be in 1..{HARD_HEAD_MAX}")
        self.enc, self.max_len, self.head_max = encode, int(max_len), int(head_max)
        st = readout_cfg["slot_tokens"]
        self.yes, self.no, arrow = int(st["yes"]["id"]), int(st["no"]["id"]), int(st["verdict_slot"]["id"])
        for text, want in ((" yes", self.yes), (" no", self.no), (" ->", arrow)):
            got = self.enc(text)
            if got != [want]:
                raise ValueError(f"tokenizer mismatch: {text!r} -> {got}, readout_config expects [{want}]")
        self.arrow = [arrow]
        self.newline = self.enc("\n")
        self.dash = self.enc("- ")
        self.judge = self.enc("Judge each option:\n")

    def prefix_ids(self, state):
        return self.enc("State:\n") + self.enc(serialize_state(state)) + self.enc("\n\n")

    def render(self, state, question, head_max=None, max_len=None):
        head_max = self.head_max if head_max is None else int(head_max)
        max_len = self.max_len if max_len is None else min(int(max_len), self.max_len)
        if head_max > HARD_HEAD_MAX:
            raise InputBudgetError(f"head_max may not exceed {HARD_HEAD_MAX}")
        names = option_names(question)
        opts = [self.enc(o) for o in render_options(question)]
        suffix = self.enc(f"Question [{question['t']}]: {question['ins']}\nOptions:\n")
        for o in opts:
            suffix += self.dash + o + self.newline
        suffix += self.judge
        rel = []
        for o in opts:
            suffix += o + self.arrow
            rel.append(len(suffix) - 1)
            suffix += self.newline
        if len(suffix) > head_max:
            raise InputBudgetError(f"question + options + readout need {len(suffix)} tokens; the head budget is "
                                   f"{head_max} (hard cap {HARD_HEAD_MAX}). Nothing was truncated: shorten the "
                                   f"question/options or split the options over several questions.")
        prefix = self.prefix_ids(state)
        ids = prefix + suffix
        if len(ids) > max_len:
            raise InputBudgetError(f"input needs {len(ids)} tokens (state {len(prefix)} + head {len(suffix)}); the "
                                   f"limit is {max_len} (model maximum {CONTEXT_LIMIT}). Nothing was truncated: "
                                   f"shorten the state.")
        return Rendered(ids, len(prefix), [len(prefix) + s for s in rel], names, len(suffix))


# -- calibration ----------------------------------------------------------------------------------
def family(category):
    for prefix, fam in FAMILY_BY_CATEGORY_PREFIX:
        if category.startswith(prefix):
            return fam
    return "other"


def option_bucket(k):
    return "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11-20" if k <= 20 else "21+"


def lookup_temperature(temps, category, qtype, n_options):
    """Calibration temperature. ``category`` None/"" -> the global temperature; otherwise the fitted
    group (family(category) x qtype x option bucket), falling back to the global temperature."""
    g = None
    if category:
        g = (temps.get("groups") or {}).get(f"{family(category)}|{qtype}|{option_bucket(n_options)}")
    t = g["T"] if g else temps.get("global", 1.0)
    lo, hi = temps.get("clamp", [0.3, 5.0])
    return float(min(hi, max(lo, t)))


def concentration(p):
    k = len(p)
    if k < 2:
        return 1.0
    ent = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum()
    return float(np.clip(1.0 - ent / math.log(k), 0.0, 1.0))


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def verify_manifest(model_dir, only=None):
    """Re-hash the files listed in manifest.json (all, or those whose path starts with one of ``only``).
    Documentation (README.md, figures/, assets/) is recorded in the manifest but not checked here, so
    a card edit never makes the runtime refuse to load."""
    model_dir = Path(model_dir)
    man = json.loads((model_dir / "manifest.json").read_text())
    bad, missing, checked = [], [], 0
    for name, rec in man["files"].items():
        if name == "README.md" or name.startswith(("assets/", "figures/")):
            continue
        if only and not any(name == o or name.startswith(o.rstrip("/") + "/") for o in only):
            continue
        p = model_dir / name
        if not p.exists():
            missing.append(name)
        elif _sha256(p) != rec["sha256"]:
            bad.append(name)
        checked += 1
    return {"ok": not bad and not missing, "checked": checked, "bad": bad, "missing": missing}


class DecisionBase:
    """Backend-independent part. Subclasses implement ``_scores(rendered) -> list[float]`` and may
    override ``_scores_many(list of rendered) -> list of list[float]`` (several questions, one state).

    Calibration: ``category`` (constructor default or per call) selects the fitted group temperature
    of that category's family; with no category anywhere, the global temperature of
    readout_config.json (temperatures.global) is used."""
    backend = "base"

    def _setup(self, model_dir, tokenizer_json, category=None, head_max=HARD_HEAD_MAX, max_len=CONTEXT_LIMIT):
        self.model_dir = Path(model_dir)
        self.readout_config = json.loads((self.model_dir / "readout_config.json").read_text())
        self.temperatures = self.readout_config["temperatures"]
        self.default_category = category or None       # None -> temperatures.global
        self.encode = TextEncoder(tokenizer_json)
        self.renderer = Renderer(self.encode, self.readout_config, max_len=max_len, head_max=head_max)

    def temperature(self, question, category=None):
        """T for ``question``: group temperature of ``category`` (or the constructor's default category);
        the global temperature when neither is given."""
        return lookup_temperature(self.temperatures, category or self.default_category, question["t"],
                                  len(option_names(question)))

    def _scores_many(self, rendered):
        return [self._scores(r) for r in rendered]

    def _result(self, r, q, scores, category=None, temperature=None):
        scores = [float(x) for x in scores]
        t = float(temperature) if temperature is not None else self.temperature(q, category)
        z = np.asarray(scores, float) / t
        if not np.all(np.isfinite(z)):
            raise FloatingPointError("non-finite decision scores")
        p = np.exp(z - z.max())
        p /= p.sum()
        i = int(p.argmax())
        return {"answer": r.names[i], "probabilities": dict(zip(r.names, p.tolist())),
                "scores": dict(zip(r.names, scores)), "temperature": t, "top_probability": float(p[i]),
                "entropy_concentration": concentration(p), "input_tokens": len(r.ids), "head_tokens": r.head_tokens,
                "model": MODEL_NAME, "backend": self.backend}

    def decide(self, state, question, options=None, qtype=None, category=None, temperature=None, head_max=None):
        """Score one question about ``state``.

        Returns {"answer", "probabilities" {option: p}, "scores" {option: logit(yes)-logit(no)},
        "temperature", "top_probability", "entropy_concentration", "input_tokens", "head_tokens"}.
        Temperature: with no ``category`` (here or in the constructor) the global temperature of
        readout_config.json is used; ``category`` picks the fitted group temperature of its family
        (e.g. "mac_gate", "general_topic", "theme_routing", "intent", "typed_official");
        ``temperature`` overrides both (1.0 = uncalibrated scores).
        Raises InputBudgetError (never truncates) or QuestionError.
        """
        q = make_question(question, options, qtype)
        r = self.renderer.render(state, q, head_max=head_max)
        return self._result(r, q, self._scores(r), category, temperature)

    def decide_many(self, state, questions, category=None, temperature=None, head_max=None):
        """Several questions about the same state (each a dict {"t","ins","crit"}); results in order.
        Same outputs as calling decide() per question. The llama.cpp runtime sends all questions in one
        request and shares the state in whole 1,024-token ubatches (see JevStyleDecisionGGUF, also for
        its faster, not bit-identical many_mode="batched"). All questions are rendered and
        budget-checked before any scoring."""
        qs = [make_question(q) for q in questions]
        rs = [self.renderer.render(state, q, head_max=head_max) for q in qs]
        if not rs:
            return []
        return [self._result(r, q, sc, category, temperature) for r, q, sc in zip(rs, qs, self._scores_many(rs))]


def base_arg_parser(description):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--model-dir", default=str(HERE), help="folder with the weights and readout_config.json")
    ap.add_argument("--state", help="state as plain text")
    ap.add_argument("--state-json", help="state as a JSON value")
    ap.add_argument("--question", help="question text (or a JSON question {'t','ins','crit'})")
    ap.add_argument("--options", help="JSON: {name: description} or [names] (choice); [levels] (score)")
    ap.add_argument("--qtype", choices=QTYPES)
    ap.add_argument("--category", help="calibration family key, e.g. mac_gate, general_topic, theme_routing, intent "
                                       "(default: none -> the global temperature of readout_config.json)")
    ap.add_argument("--temperature", type=float, help="override the calibrated temperature")
    ap.add_argument("--head-max", type=int, default=HARD_HEAD_MAX)
    ap.add_argument("--max-len", type=int, default=CONTEXT_LIMIT)
    ap.add_argument("--jsonl", help="batch mode: input JSON lines {id?, state, question, options?, qtype?, "
                                    "category?}; one JSON result per line on stdout")
    ap.add_argument("--verify", action="store_true", help="check sha256 of the files in manifest.json first")
    return ap


def run_cli(args, engine):
    def one(rec):
        q = rec["question"]
        return engine.decide(rec.get("state", ""), q, options=rec.get("options"), qtype=rec.get("qtype"),
                             category=rec.get("category"), temperature=rec.get("temperature", args.temperature))
    if args.jsonl:
        src = sys.stdin if args.jsonl == "-" else open(args.jsonl, encoding="utf-8")
        for n, line in enumerate(src):
            if not line.strip():
                continue
            rec = json.loads(line)
            rid = rec.get("id", n)
            try:
                out = {"id": rid, **one(rec)}
            except (InputBudgetError, QuestionError) as e:
                out = {"id": rid, "error": f"{type(e).__name__}: {e}"}
            print(json.dumps(out, ensure_ascii=False), flush=True)
        return 0
    if args.question is None:
        raise SystemExit("--question (or --jsonl) is required")
    state = json.loads(args.state_json) if args.state_json is not None else (args.state or "")
    question = args.question
    if question.lstrip().startswith("{"):
        question = json.loads(question)
    rec = {"state": state, "question": question, "options": json.loads(args.options) if args.options else None,
           "qtype": args.qtype, "category": args.category}
    print(json.dumps(one(rec), ensure_ascii=False, indent=2))
    return 0
# ---------------------------------------------------------------------------- end of shared core


# ------------------------------------------------------------------------------ PyTorch backend
ATTN_CHUNK = 1024
_CHUNKED_NAME = "jev_chunked_sdpa"
_CHUNKED_REGISTERED = False


def _chunked_sdpa_forward(module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None,
                          **kwargs):
    """Query-chunked SDPA (MPS / CPU): the same computation as transformers' sdpa, but the
    [heads x queries x keys] score matrix is built ATTN_CHUNK queries at a time, so a 25,600-token
    input does not need ~21 GB for one attention call. Inputs of <= ATTN_CHUNK tokens take the
    unchanged sdpa path."""
    import torch
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    q_len, kv_len = query.shape[2], key.shape[2]
    chunk = int(getattr(module, "jev_attn_chunk", ATTN_CHUNK) or ATTN_CHUNK)
    if q_len <= chunk:
        return sdpa_attention_forward(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling,
                                      is_causal=is_causal, **kwargs)
    causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)
    past = kv_len - q_len
    outs = []
    for s in range(0, q_len, chunk):
        e = min(q_len, s + chunk)
        if attention_mask is not None:
            m = attention_mask if attention_mask.shape[-2] == 1 else attention_mask[:, :, s:e, :]
        elif causal:
            qpos = torch.arange(s + past, e + past, device=query.device)
            kpos = torch.arange(kv_len, device=query.device)
            m = (kpos[None, :] <= qpos[:, None])[None, None]
        else:
            m = None
        out, _ = sdpa_attention_forward(module, query[:, :, s:e], key, value, m, dropout=dropout, scaling=scaling,
                                        is_causal=False, **kwargs)
        outs.append(out)
    return torch.cat(outs, dim=1), None


def _enable_chunked_attention(model, chunk=ATTN_CHUNK):
    global _CHUNKED_REGISTERED
    if getattr(model.config, "_attn_implementation", None) not in ("sdpa", _CHUNKED_NAME):
        return False
    try:
        from transformers import AttentionInterface
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, AttentionMaskInterface
    except ImportError:
        return False
    if not _CHUNKED_REGISTERED:
        AttentionInterface.register(_CHUNKED_NAME, _chunked_sdpa_forward)
        AttentionMaskInterface.register(_CHUNKED_NAME, ALL_MASK_ATTENTION_FUNCTIONS["sdpa"])
        _CHUNKED_REGISTERED = True
    model.set_attn_implementation(_CHUNKED_NAME)
    for m in model.modules():
        if hasattr(m, "is_causal"):
            m.jev_attn_chunk = int(chunk)
    return True


class JevStyleDecision(DecisionBase):
    """Transformers / PyTorch runtime (CUDA, Apple MPS or CPU).

    >>> m = JevStyleDecision(".")                       # float32 on the best available device
    >>> m.decide({"messages": ["Refund still missing after 3 weeks"]},
    ...          "Which team should handle this ticket?",
    ...          options={"billing": "payments, refunds", "tech": "bugs, crashes", "sales": "pricing, plans"},
    ...          category="theme_routing")["probabilities"]
    """
    backend = "torch"

    def __init__(self, model_dir=HERE, device=None, dtype="float32", category=None, head_max=HARD_HEAD_MAX,
                 max_len=CONTEXT_LIMIT, attn_chunk=ATTN_CHUNK, verify=False):
        import torch
        self.torch = torch
        model_dir = Path(model_dir)
        if verify:
            res = verify_manifest(model_dir)
            if not res["ok"]:
                raise RuntimeError(f"integrity check failed: {res}")
        self._setup(model_dir, model_dir / "tokenizer.json", category, head_max, max_len)
        if device is None:
            device = ("cuda" if torch.cuda.is_available() else
                      "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
        dt = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        try:
            from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM as cls
        except ImportError as e:
            raise ImportError("this model needs a transformers version with Qwen3.5 support "
                              "(transformers.models.qwen3_5)") from e
        try:
            model = cls.from_pretrained(str(model_dir), dtype=dt)
        except TypeError:                                       # transformers 4.x keyword
            model = cls.from_pretrained(str(model_dir), torch_dtype=dt)
        self.model = model.to(device).eval()
        self.device, self.dtype = device, dt
        self.chunked_attention = _enable_chunked_attention(self.model, attn_chunk) if device != "cuda" else False
        w = self.model.get_output_embeddings().weight
        self.direction = (w[self.renderer.yes].float() - w[self.renderer.no].float()).detach()

    def _scores(self, r):
        torch = self.torch
        with torch.no_grad():
            ids = torch.tensor([r.ids], device=self.device)
            h = self.model.model(input_ids=ids, use_cache=False).last_hidden_state      # final normed hidden states
            hs = h[0, torch.tensor(r.slots, device=self.device)].float()
            return (hs @ self.direction).cpu().tolist()


def main(argv=None):
    ap = base_arg_parser(f"{MODEL_NAME}: typed decisions with transformers / PyTorch")
    ap.add_argument("--device", choices=["cuda", "mps", "cpu"])
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    args = ap.parse_args(argv)
    engine = JevStyleDecision(args.model_dir, device=args.device, dtype=args.dtype, category=args.category,
                              head_max=args.head_max, max_len=args.max_len, verify=args.verify)
    return run_cli(args, engine)


if __name__ == "__main__":
    raise SystemExit(main())
