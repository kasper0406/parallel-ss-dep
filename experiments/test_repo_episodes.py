"""Tests for the Meta-TTT P0 repo-episode builder + three-arm eval harness
(`gen_repo_episodes.py`, `eval_repo_adaptive.py`).

Network-free: uses hand-built synthetic repos + the (cached) SmolLM2 tokenizer.
The CE-arm test uses a tiny causal CPU stub (real TinyLM needs Triton/GPU).

Run:
  PYTHONPATH=. .venv/bin/python -m pytest experiments/test_repo_episodes.py -v
"""
from __future__ import annotations

import functools

import pytest
import torch
import torch.nn as nn

from experiments import gen_repo_episodes as G
from experiments import eval_repo_adaptive as E


@functools.lru_cache(maxsize=1)
def _tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")


# --------------------------------------------------------------------------- #
# Synthetic repos.
# --------------------------------------------------------------------------- #

DEF_FILE = (
    "import math\n"
    "\n"
    "def compute_total(items, tax_rate):\n"
    '    """Sum item prices and apply a tax rate."""\n'
    "    subtotal = sum(p for _, p in items)\n"
    "    return subtotal * (1.0 + tax_rate)\n"
    "\n"
    "def _trivial(x):\n"          # 1-arg, no doc, 1 body line -> NOT quality
    "    return x\n"
)

HELPER_FILE = (
    "PI = 3.14159\n"
    "TWO_PI = 2 * PI\n"
    "MAX_ITEMS = 100\n"
    "DEFAULT_TAX = 0.2\n"
)

USE_FILE = (
    "from store.pricing import compute_total\n"
    "\n"
    "cart = [('apple', 1.0), ('bread', 2.5)]\n"
    "rate = 0.2\n"
    "grand = compute_total(cart, rate)\n"
    "print(grand)\n"
)


def _repo():
    return {
        "store/pricing.py": DEF_FILE,
        "store/constants.py": HELPER_FILE,
        "app/checkout.py": USE_FILE,
    }


# --------------------------------------------------------------------------- #
# AST link miner.
# --------------------------------------------------------------------------- #

def test_miner_finds_planted_cross_file_link():
    links = G.mine_cross_file_links(_repo())
    ids = {l.identifier for l in links}
    assert "compute_total" in ids
    L = next(l for l in links if l.identifier == "compute_total")
    assert L.def_path == "store/pricing.py"
    assert L.use_path == "app/checkout.py"
    assert L.def_line == 3
    assert L.use_line == 5           # `grand = compute_total(cart, rate)`


def test_miner_ignores_same_file_use():
    # `helper` defined AND called in the same file -> not a cross-file link.
    files = {
        "a.py": "def helper(a, b):\n    return a + b\n\nx = helper(1, 2)\n",
        "b.py": "y = 3\n",
    }
    links = G.mine_cross_file_links(files)
    assert all(l.identifier != "helper" for l in links)


def test_miner_ignores_builtins():
    files = {
        "a.py": "def foo(a, b):\n    return a + b\n",
        "b.py": "print(len([1, 2, 3]))\nrange(10)\n",
    }
    links = G.mine_cross_file_links(files)
    # print/len/range are builtins and must not be linked even though b.py
    # calls them; foo is defined but never cross-file-called.
    assert links == []


def test_miner_ignores_imports_only():
    # `thing` is imported but never actually referenced (no ast.Name/Call) ->
    # no use site -> no link.
    files = {
        "a.py": "def thing(a, b):\n    '''doc'''\n    return a - b\n",
        "b.py": "from a import thing\nz = 1\n",
    }
    links = G.mine_cross_file_links(files)
    assert all(l.identifier != "thing" for l in links)


def test_miner_skips_generic_names():
    # `run` is generic (test-fixture-ish) -> skipped even when quality + xfile.
    files = {
        "a.py": "def run(a, b):\n    '''doc'''\n    return a + b\n",
        "b.py": "from a import run\nx = run(1, 2)\n",
    }
    assert G.mine_cross_file_links(files) == []
    # but the same structure with a distinctive name IS linked
    files2 = {
        "a.py": "def orchestrate(a, b):\n    '''doc'''\n    return a + b\n",
        "b.py": "from a import orchestrate\nx = orchestrate(1, 2)\n",
    }
    assert any(l.identifier == "orchestrate"
               for l in G.mine_cross_file_links(files2))


def test_miner_import_flag_and_require_import():
    imported = {
        "a.py": "def widget_factory(a, b):\n    '''d'''\n    return a\n",
        "b.py": "from a import widget_factory\ny = widget_factory(1, 2)\n",
    }
    L = next(l for l in G.mine_cross_file_links(imported)
             if l.identifier == "widget_factory")
    assert L.imported is True

    # used but NOT imported (e.g. via `*` import or attribute injection): the
    # link is still found (imported=False) unless require_import.
    not_imported = {
        "a.py": "def widget_factory(a, b):\n    '''d'''\n    return a\n",
        "b.py": "y = widget_factory(1, 2)\n",
    }
    links = G.mine_cross_file_links(not_imported)
    assert any(l.identifier == "widget_factory" and l.imported is False
               for l in links)
    assert G.mine_cross_file_links(not_imported, require_import=True) == []


def test_miner_def_quality_filter():
    # `_trivial` (1 arg, no doc, 1 body line) is low quality -> not linked even
    # when cross-file called.
    files = {
        "a.py": "def _trivial(x):\n    return x\n",
        "b.py": "from a import _trivial\nw = _trivial(5)\n",
    }
    links = G.mine_cross_file_links(files)
    assert all(l.identifier != "_trivial" for l in links)

    # add a second arg -> now quality -> linked
    files2 = {
        "a.py": "def _trivial(x, y):\n    return x\n",
        "b.py": "from a import _trivial\nw = _trivial(5, 6)\n",
    }
    links2 = G.mine_cross_file_links(files2)
    assert any(l.identifier == "_trivial" for l in links2)


# --------------------------------------------------------------------------- #
# Episode assembly.
# --------------------------------------------------------------------------- #

@functools.lru_cache(maxsize=1)
def _episode():
    tok = _tok()
    repo = _repo()
    links = G.mine_cross_file_links(repo)
    L = next(l for l in links if l.identifier == "compute_total")
    # tiny repo -> below the 4k budget; build with a relaxed floor for testing
    ep = G._build_episode_relaxed(repo, L, tok)
    return ep, L


def test_episode_def_first_use_last():
    ep, L = _episode()
    paths = [f["path"] for f in ep["context_files"]]
    assert paths[-1] == L.use_path                     # use file last
    assert paths[0] == L.def_path                      # def file first
    # def strictly in the first half
    assert paths.index(L.def_path) < len(paths) / 2


def test_episode_prefix_line_split_exact():
    ep, L = _episode()
    use_text = _repo()[L.use_path]
    # prefix + line == the use file up through the usage line
    keep = use_text.splitlines(keepends=True)
    expected = "".join(keep[:L.use_line])
    assert ep["task_prefix"] + ep["task_line"] == expected
    # task_prefix does NOT contain the usage line
    assert ep["task_line"] not in ep["task_prefix"]


def test_episode_char_span_correct():
    ep, L = _episode()
    s0, s1 = ep["task_char_span"]
    assert ep["task_line"][s0:s1] == "compute_total"


# --------------------------------------------------------------------------- #
# Control builder.
# --------------------------------------------------------------------------- #

def test_control_token_count_preserved_per_file():
    tok = _tok()
    ep, L = _episode()
    ctrl = G.build_control(ep, tok, seed=0)
    # map path -> file for both
    real = {f["path"]: f for f in ep["context_files"][:-1]}
    for f in ctrl["context_files"][:-1]:
        real_ids = E._file_ids(real[f["path"]], tok)     # per-line tokenization
        ctrl_ids = E._file_ids(f, tok)
        assert len(ctrl_ids) == len(real_ids), (
            f"token count drift on {f['path']}: "
            f"{len(ctrl_ids)} vs {len(real_ids)}")


def test_control_token_count_preserved_adversarial_line_endings():
    # Risk-3 regression: EXACT length-match must survive files with (a) no
    # trailing newline, (b) an embedded form-feed \x0c (common as a Python page
    # break) and (c) CRLF / bare-CR — all of which str.splitlines would cut on,
    # gluing chunks under permutation. split_lines_nl (split on \n only) must
    # keep the count exactly permutation-invariant.
    tok = _tok()
    # def + use files are clean (mineable); the adversarial chars live in a
    # NON-TASK context file (the control shuffles it; parseability irrelevant).
    files = {
        "pkg/util.py": ("import os\n"
                        "def load_config(path, default):\n"
                        "    '''Load a config file.'''\n"
                        "    return {}\n"),
        # non-task context file with form-feed, CRLF, and no trailing newline:
        "pkg/data.py": ("HEADER = 1\n"
                        "\x0c\n"                       # form-feed page break
                        "ROW_A = 10\r\n"               # CRLF line
                        "ROW_B = 20\r\n"
                        "ROW_C = 30\n"
                        "TRAILER = 99"),               # no trailing newline
        "pkg/extra.py": "A = 1\nB = 2\nC = 3",         # no trailing newline
        "app/run.py": ("from pkg.util import load_config\n"
                       "cfg = load_config('x', {})\n"
                       "print(cfg)\n"),
    }
    links = G.mine_cross_file_links(files)
    L = next(l for l in links if l.identifier == "load_config")
    ep = G._build_episode_relaxed(files, L, tok)
    # ensure the adversarial file is actually in the episode context
    assert any(f["path"] == "pkg/data.py" for f in ep["context_files"][:-1])
    for seed in range(8):
        ctrl = G.build_control(ep, tok, seed=seed)
        real = {f["path"]: f for f in ep["context_files"][:-1]}
        for f in ctrl["context_files"][:-1]:
            assert (len(E._file_ids(f, tok))
                    == len(E._file_ids(real[f["path"]], tok))), (
                f"length drift on {f['path']} seed={seed}")


def test_control_task_file_untouched():
    tok = _tok()
    ep, L = _episode()
    ctrl = G.build_control(ep, tok, seed=0)
    assert ctrl["context_files"][-1]["path"] == ep["task_file"]
    assert ctrl["context_files"][-1] == ep["context_files"][-1]
    # task file must NOT gain a token_ids override
    assert "token_ids" not in ctrl["context_files"][-1]


def test_control_deterministic_under_seed():
    tok = _tok()
    ep, L = _episode()
    c1 = G.build_control(ep, tok, seed=7)
    c2 = G.build_control(ep, tok, seed=7)
    assert c1["context_files"] == c2["context_files"]
    c3 = G.build_control(ep, tok, seed=8)
    # different seed -> different shuffle (multi-line def file makes this ~sure)
    assert c1["context_files"] != c3["context_files"]


def _shuffled_def_text(ep, L, tok, seed):
    ctrl = G.build_control(ep, tok, seed=seed)
    def_file = next(f for f in ctrl["context_files"]
                    if f["path"] == L.def_path)
    return tok.decode(E._file_ids(def_file, tok))


def test_control_definition_line_no_longer_intact():
    tok = _tok()
    ep, L = _episode()
    orig = _repo()[L.def_path]

    # (guaranteed) every seed yields a true permutation whose ORDER differs
    # from the original -> the definition is not presented in natural form.
    for seed in (0, 1, 2, 3, 4):
        s = _shuffled_def_text(ep, L, tok, seed)
        assert sorted(orig.splitlines()) == sorted(s.splitlines())  # permutation
        assert s != orig                                            # order broke

    # (mechanism) line-shuffling CAN separate the def line from its body; find a
    # seed that demonstrably breaks the def-line -> docstring adjacency (a single
    # random permutation may keep it, so search — deterministic since
    # build_control is seeded).
    def_line = "def compute_total(items, tax_rate):\n"
    body_line = '    """Sum item prices and apply a tax rate."""\n'
    broke = any((def_line + body_line) not in _shuffled_def_text(ep, L, tok, s)
                for s in range(50))
    assert broke, "expected some seed to break the def-line -> body adjacency"


# --------------------------------------------------------------------------- #
# Repo split.
# --------------------------------------------------------------------------- #

def test_same_identifier_multiple_uses_get_distinct_ids_and_correct_control():
    # A repo where ONE identifier is used in TWO files produces two episodes
    # sharing (repo, identifier); episode_id must disambiguate so each episode
    # matches its OWN control (the bug that caused spurious length mismatches).
    tok = _tok()
    files = {
        "core/proc.py": ("def process_data(rows, opts):\n"
                         "    '''Process rows.'''\n"
                         "    return [r for r in rows]\n"),
        "a/one.py": ("from core.proc import process_data\n"
                     "AAA = 1\nBBB = 2\nCCC = 3\n"
                     "out1 = process_data([1], {})\n"),
        "b/two.py": ("from core.proc import process_data\n"
                     "XXX = 9\nYYY = 8\nZZZ = 7\n"
                     "out2 = process_data([2], {})\n"),
    }
    links = [l for l in G.mine_cross_file_links(files)
             if l.identifier == "process_data"]
    assert len(links) == 2                      # used in two files
    eps = [G._build_episode_relaxed(files, l, tok) for l in links]
    assert eps[0]["episode_id"] != eps[1]["episode_id"]
    assert eps[0]["repo_name"] == eps[1]["repo_name"]           # collide on repo
    assert (eps[0]["link"]["identifier"]
            == eps[1]["link"]["identifier"])                    # and identifier

    # emulate the eval's dict keying + matching
    ctrls = {E._ep_key(G.build_control(e, tok, seed=0)): G.build_control(e, tok, 0)
             for e in eps}
    for e in eps:
        c = ctrls[E._ep_key(e)]
        # matched control must have the SAME non-task file set as its episode
        assert ({f["path"] for f in c["context_files"][:-1]}
                == {f["path"] for f in e["context_files"][:-1]})
        assert (len(E.context_ids(c, tok)) == len(E.context_ids(e, tok)))


def test_repo_split_disjoint_and_deterministic():
    names = [f"user{i}/proj{i}" for i in range(400)]
    a = {n: G.repo_split(n, seed=0) for n in names}
    b = {n: G.repo_split(n, seed=0) for n in names}
    assert a == b                                       # deterministic
    train = {n for n, s in a.items() if s == "train"}
    ev = {n for n, s in a.items() if s == "eval"}
    assert train.isdisjoint(ev)                         # disjoint
    assert len(ev) > 0 and len(train) > 0               # both populated


# --------------------------------------------------------------------------- #
# Eval CE arms — tiny causal CPU stub.
# --------------------------------------------------------------------------- #

class _CausalStub(nn.Module):
    """Minimal model matching the eval interface: forward(x, skip_lm_head=True)
    returns a hidden that DEPENDS ON ALL PRECEDING TOKENS (causal running
    mean), so `real` (with context) and `none` (without) genuinely differ."""

    def __init__(self, vocab: int, d: int = 16, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.embed = nn.Embedding(vocab, d)
        self.lm_head = nn.Linear(d, vocab, bias=False)
        with torch.no_grad():
            self.embed.weight.normal_(0, 1.0, generator=g)
            self.lm_head.weight.normal_(0, 0.3, generator=g)
        self.use_memory = False
        self._film_bypass = True

    def forward(self, x, skip_lm_head=False, doc_ids=None, **kw):
        e = self.embed(x)                               # (B,T,d)
        csum = torch.cumsum(e, dim=1)
        denom = torch.arange(1, e.shape[1] + 1, device=e.device).view(1, -1, 1)
        h = e + csum / denom                            # causal running mean
        if skip_lm_head:
            return h
        return self.lm_head(h)


def test_eval_real_vs_none_differ_and_span_math():
    tok = _tok()
    ep, L = _episode()
    model = _CausalStub(vocab=tok.vocab_size, seed=1)

    r = E.eval_episode(model, ep, None, tok, device="cpu", bf16=False)
    assert r["real"]["line_ce"] is not None
    assert r["none"]["line_ce"] is not None
    # context changes the hidden at task positions -> arms differ
    assert abs(r["real"]["line_ce"] - r["none"]["line_ce"]) > 1e-4

    # --- independent hand-recompute of the span CE for the `none` arm ---
    task_prefix_ids = G.perline_ids(ep["task_prefix"], tok)   # per-line (match)
    line_ids, span_idx = E.task_line_span_tokens(ep, tok)
    assert len(span_idx) >= 1
    full = task_prefix_ids + line_ids
    P = len(task_prefix_ids)
    x = torch.tensor([full])
    with torch.no_grad():
        h = model(x, skip_lm_head=True)
        logits = model.lm_head(h)[0]                    # (T, V)
    # CE for each span token: predict full[P+i] from logits[P+i-1]
    ref = []
    logp = torch.log_softmax(logits.float(), dim=-1)
    for i in span_idx:
        pos = P + i
        ref.append(-logp[pos - 1, full[pos]].item())
    ref_span_ce = sum(ref) / len(ref)

    got = E.arm_ce(model, task_prefix_ids, line_ids, span_idx,
                   device="cpu", bf16=False)
    assert got["span_ce"] == pytest.approx(ref_span_ce, abs=1e-5)
    # and it equals the aggregate none-arm span value
    assert r["none"]["span_ce"] == pytest.approx(ref_span_ce, abs=1e-5)


def test_eval_control_length_matched_arm():
    tok = _tok()
    ep, L = _episode()
    ctrl = G.build_control(ep, tok, seed=0)
    model = _CausalStub(vocab=tok.vocab_size, seed=2)
    r = E.eval_episode(model, ep, ctrl, tok, device="cpu", bf16=False)
    assert "shuffled" in r
    # real vs shuffled context are exactly length-matched
    assert r["_ctx_len_real"] == r["_ctx_len_shuffled"]
    assert r["shuffled"]["line_ce"] is not None


def test_aggregate_lifts_sign():
    # two fake results: ingestion helps (real lower than none/shuffled)
    results = [
        {"bucket": "4-8k", "identifier_in_task_prefix": False,
         "ctx_length_match": True,
         "real": {"line_ce": 1.0, "span_ce": 1.0},
         "shuffled": {"line_ce": 1.5, "span_ce": 1.6},
         "none": {"line_ce": 2.0, "span_ce": 2.2}},
        {"bucket": "4-8k", "identifier_in_task_prefix": True,
         "ctx_length_match": True,
         "real": {"line_ce": 0.8, "span_ce": 0.9},
         "shuffled": {"line_ce": 1.1, "span_ce": 1.2},
         "none": {"line_ce": 1.4, "span_ce": 1.5}},
    ]
    agg = E.aggregate(results)
    o = agg["overall"]
    assert o["lift_none_minus_real_line"] == pytest.approx(0.8)      # (2-1+1.4-0.8)/2
    assert o["lift_shuffled_minus_real_line"] == pytest.approx(0.4)  # (0.5+0.3)/2
    assert o["lift_none_minus_real_span"] > o["lift_shuffled_minus_real_span"]
    assert o["n_shuffled"] == 2
    strat = agg["none_stratified_by_identifier_in_prefix"]
    assert set(strat.keys()) == {"True", "False"}


def test_aggregate_excludes_length_mismatched_shuffled():
    # a length-mismatched control must NOT contaminate the shuffled metric
    results = [
        {"bucket": "4-8k", "ctx_length_match": True,
         "real": {"line_ce": 1.0, "span_ce": 1.0},
         "shuffled": {"line_ce": 1.5, "span_ce": 1.5},
         "none": {"line_ce": 2.0, "span_ce": 2.0}},
        {"bucket": "4-8k", "ctx_length_match": False,   # excluded from shuffled
         "real": {"line_ce": 1.0, "span_ce": 1.0},
         "shuffled": {"line_ce": 9.9, "span_ce": 9.9},
         "none": {"line_ce": 2.0, "span_ce": 2.0}},
    ]
    agg = E.aggregate(results)["overall"]
    assert agg["n_shuffled"] == 1
    assert agg["lift_shuffled_minus_real_line"] == pytest.approx(0.5)  # only ep0
    assert agg["lift_none_minus_real_line"] == pytest.approx(1.0)      # both eps
