"""v0.42 — "A. Professional polish" (SPEC.md 56, A46, M45).

Covers the six A items as implemented (opt-in / byte-identical-by-default,
per the screening precedent — every new surface is off by default and the
pre-v0.42 call paths stay byte-identical):

- 56.1 the **provenance certificate** core (``provenance.py``) —
  * ``canonical_hash(payload)`` — sha256 of the canonical JSON encoding
    (``sort_keys``, compact separators, ``default=str``); key order and
    whitespace never change it, and a non-JSON value hashes through
    ``default=str``;
  * ``env_provenance()`` — the environment facts (``python`` / ``numpy`` /
    the fixed-order ``extras`` dict / ``git_sha``), never raising;
  * ``provenance_payload(...)`` — the pure merge into a fixed key order
    (``tool`` → … → ``config_hash`` → ``git_sha``), ``config_hash`` None
    only when the run carries no ``run_config``; a non-dict ``env``
    degrades;
  * ``provenance_card(payload)`` — the plain-text certificate (one
    ``key: value`` line per entry, ``extras`` indented, None → ``—``);
  * ``svg_provenance(payload)`` — the SVG card (ARIA, ``role="img"``,
    default==explicit, okabe/dark re-color, the empty → message edge).
- 56.2 the **brand chrome + report cover** — ``html_cover(summary, payload)``
  (masthead + provenance table; ``payload=None`` masthead-only; ints and
  None coerced) and ``html_report(..., cover=)`` (``cover=None`` byte-
  identical; a cover renders after ``<body>`` and before the ``<h1>``);
- 56.3 the **design-token registry** — ``TOKENS`` (the single documented
  source) and ``resolve_tokens(palette, dark)`` (a NEW dict, never mutating
  the module globals; okabe/dark override sets merge back onto the canonical
  names via ``_TOKENS_FROM_KEY``; the okabe subset keeps the rest at the
  default; validation mirrors ``_styled`` exactly);
- 56.4 the **app states + expanders** — the empty (Setup), loading (Run),
  and error (points at ``autorefine doctor``) states; the "Provenance
  (certificate)" and "Design tokens" expanders; the "Reduce motion" checkbox;
- 56.5 **reduced motion** — ``svg_live_sparkline(..., reduced_motion=True)``
  suppresses the pulsing halo (the app's one animated affordance) while the
  dot + title stay;
- the A46 round regression — the version stepped to ``0.42.0`` in both
  sources (33.1) and SPEC carries the A46 block + M45 row.

House rules: no cross-test imports (all fixtures synthesized here); the pure
core (``provenance.py`` + the two ``plotting`` functions) is tested by
hand-computation; the app (``dashboard_app.py``) and the CLI (``cli.py``)
are thin renderers tested by source-token assertions + one end-to-end AppTest
(streamlit optional). Note: this Streamlit build's ``st.expander`` has no
``key`` param, so the new expanders are asserted by ``label`` (not key) —
the pre-v0.42 expanders never used one either.
"""
from __future__ import annotations

import json
import re
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

import autorefine
from autorefine.plotting import (
    TOKENS,
    _DARK,
    _OKABE,
    _TOKENS_FROM_KEY,
    html_cover,
    html_report,
    resolve_tokens,
    svg_live_sparkline,
)
from autorefine.provenance import (
    canonical_hash,
    env_provenance,
    provenance_card,
    provenance_payload,
    svg_provenance,
)

REPO = Path(__file__).resolve().parent.parent
APP = REPO / "src" / "autorefine" / "dashboard_app.py"
CLI = REPO / "src" / "autorefine" / "cli.py"

# the canonical-hash fixtures (hand-computed, G2 — a re-run is byte-identical)
_HASH_AB = "d3626ac30a87e6f7a6428233b3c68299976865fa5508e4267c5415c76af7a772"
_HASH_TUPLE = "036b898f9248c0e83d645e262473d1d93b3085400b881ff658928b81ca06de91"


def _assert_xml(svg: str) -> None:
    ET.fromstring(svg)  # valid XML (G2)


# --- 56.1 the provenance certificate core (A46) ------------------------------

def test_canonical_hash_hand_computed():
    """A46 (SPEC.md 56.1): sha256 of the canonical JSON encoding — key order
    and whitespace never change it, and a non-JSON value hashes through
    ``default=str`` (a tuple → its JSON list)."""
    # the fixed fixture (key order + compact separators)
    assert canonical_hash({"a": 2, "b": 1}) == _HASH_AB
    # key order is irrelevant (sort_keys)
    assert canonical_hash({"b": 1, "a": 2}) == _HASH_AB
    # whitespace / nested order is irrelevant
    assert canonical_hash({"outer": {"b": 2, "a": 1}}) == \
        canonical_hash({"outer": {"a": 1, "b": 2}})
    # a non-JSON value (a tuple) degrades through default=str → its list
    assert canonical_hash({"x": (1, 2)}) == _HASH_TUPLE
    assert canonical_hash({"x": (1, 2)}) == canonical_hash({"x": [1, 2]})
    # deterministic for the same payload
    assert canonical_hash({"a": 2, "b": 1}) == canonical_hash({"a": 2, "b": 1})


def test_env_provenance_shape():
    """A46 (SPEC.md 56.1): the environment facts — the four keys, a string
    ``python`` / ``numpy``, the fixed-order ``extras`` dict (streamlit /
    pillow / soundfile), and a ``git_sha`` that is a short hex or None. Never
    raises (every probe degrades)."""
    env = env_provenance()
    assert set(env.keys()) == {"python", "numpy", "extras", "git_sha"}
    assert isinstance(env["python"], str) and env["python"]
    assert isinstance(env["numpy"], str) and env["numpy"]
    # the three optional packages, in the fixed order (pyproject [gui/
    # image/audio])
    assert list(env["extras"].keys()) == ["streamlit", "pillow", "soundfile"]
    assert env["git_sha"] is None or re.fullmatch(r"[0-9a-f]{6,}",
                                                 env["git_sha"])


def test_provenance_payload_key_order():
    """A46 (SPEC.md 56.1): the fixed key order, the run identity merged in,
    and ``config_hash`` present only when a ``run_config`` is supplied. A
    non-dict ``env`` degrades to the empty facts (never a crash)."""
    env = {
        "python": "3.13", "numpy": "2.0",
        "extras": {"streamlit": "1.51.0", "pillow": None, "soundfile": "0.14"},
        "git_sha": "abc1234",
    }
    pl = provenance_payload(env, seed=7, task="parity", target=95.0,
                            run_config={"a": 2, "b": 1})
    assert list(pl.keys()) == [
        "tool", "version", "python", "numpy", "extras", "task", "seed",
        "target", "config_hash", "git_sha"]
    assert pl["tool"] == "autorefine"
    assert pl["version"] == autorefine.__version__
    assert pl["python"] == "3.13" and pl["numpy"] == "2.0"
    assert pl["extras"] == {"streamlit": "1.51.0", "pillow": None,
                            "soundfile": "0.14"}
    assert pl["task"] == "parity" and pl["seed"] == 7 and pl["target"] == 95.0
    assert pl["config_hash"] == _HASH_AB  # the same canonical hash
    assert pl["git_sha"] == "abc1234"
    # config_hash is None only when there is no run_config
    assert provenance_payload(env, run_config=None)["config_hash"] is None
    # a non-dict env degrades (no crash; extras fall back to all-None)
    deg = provenance_payload("not-a-dict", seed=1)
    assert deg["python"] is None and deg["numpy"] is None
    assert deg["extras"] == {"streamlit": None, "pillow": None,
                             "soundfile": None}
    assert deg["config_hash"] is None


def test_provenance_card():
    """A46 (SPEC.md 56.1): the plain-text certificate — the header line, one
    ``key: value`` line per payload entry in order, ``extras`` as indented
    lines, and a None reading ``—``. A non-dict payload renders the header
    alone."""
    payload = {
        "tool": "autorefine", "version": "0.42.0",
        "python": "3.13", "numpy": "2.0",
        "extras": {"streamlit": "1.51.0", "pillow": None, "soundfile": None},
        "task": "parity", "seed": 7, "target": 95.0,
        "config_hash": "abc123", "git_sha": None,
    }
    card = provenance_card(payload)
    lines = card.splitlines()
    assert lines[0] == "AutoRefine provenance certificate"
    assert "tool: autorefine" in card
    assert "version: 0.42.0" in card
    assert "task: parity" in card
    assert "seed: 7" in card
    assert "target: 95.0" in card
    assert "config_hash: abc123" in card
    # extras are indented; a None version reads `—`
    assert "  streamlit: 1.51.0" in card
    assert "  pillow: —" in card
    assert "  soundfile: —" in card
    # a top-level None reads `—`
    assert "git_sha: —" in card
    # entry order is preserved (task before seed, seed before target)
    assert card.index("task: parity") < card.index("seed: 7") < \
        card.index("target: 95.0")
    # a non-dict payload is the header alone (never a crash)
    assert provenance_card(None) == "AutoRefine provenance certificate"
    assert provenance_card({"tool": "autorefine"}) == \
        "AutoRefine provenance certificate\ntool: autorefine"


def test_svg_provenance():
    """A46 (SPEC.md 56.1): the SVG certificate card — valid XML, ARIA
    (``role="img"``), default==explicit (G2), the okabe/dark theme re-color,
    and the empty payload → the header + message."""
    payload = provenance_payload(env_provenance(), seed=7, task="parity",
                                 target=95.0, run_config={"a": 2, "b": 1})
    svg = svg_provenance(payload)
    _assert_xml(svg)
    assert 'role="img"' in svg and "aria-label=" in svg
    # default call is byte-identical to an explicit one (G2)
    assert svg == svg_provenance(payload, palette="default", dark=False)
    # the payload's values render (the task + a long hash truncated w/ title)
    assert "parity" in svg
    assert _HASH_AB[:16] in svg  # the truncated config_hash
    assert f"<title>{_HASH_AB}</title>" in svg  # the full value on hover
    # okabe re-colors the data/line tokens; dark carries the dark surface
    okabe = svg_provenance(payload, palette="okabe")
    _assert_xml(okabe)
    dark = svg_provenance(payload, dark=True)
    _assert_xml(dark)
    assert "#0e1117" in dark  # the dark background (51.4.2)
    assert "#0e1117" not in svg  # not the default
    # the empty payload renders the header + a message (never a crash)
    empty = svg_provenance({})
    _assert_xml(empty)
    assert "no provenance payload" in empty


# --- 56.2 the brand chrome + report cover (A46) ------------------------------

def test_html_cover():
    """A46 (SPEC.md 56.2): the cover block — the brand masthead (the
    ``AutoRefine`` wordmark + the version tag) and the provenance table.
    ``payload=None`` renders the masthead only; ints and None are coerced
    (never a crash); dynamic text is HTML-escaped."""
    payload = {
        "tool": "autorefine", "version": "0.42.0",
        "extras": {"streamlit": "1.51.0", "pillow": None},
        "task": "parity", "seed": 7, "target": 95.0,
        "config_hash": "abc123", "git_sha": None,
    }
    cov = html_cover({"task": "parity", "seed": 7}, payload)
    assert "<div class=\"brand\">AutoRefine</div>" in cov
    assert f"v{autorefine.__version__}" in cov  # the version tag
    assert "<table>" in cov  # the provenance table
    assert "  streamlit" in cov  # an extras row
    # a None cell reads `—`
    assert "<td>—</td>" in cov
    # the payload=None path is the masthead only (no table)
    mast = html_cover({"task": "parity"}, None)
    assert "AutoRefine" in mast and "<table>" not in mast
    # an int is coerced to a string (never an AttributeError on .escape)
    assert "seed: 7" in provenance_card(payload) or "7" in cov
    # a non-dict summary degrades (never a crash)
    assert html_cover(None, payload).count("AutoRefine") >= 1


def test_html_report_cover_byte_identity():
    """A46 (SPEC.md 56.2): ``cover=None`` is byte-identical to a call without
    the kwarg (28.5); a cover renders after ``<body>`` and before the
    ``<h1>`` (the top of the body); a call without a cover has no cover
    header at all."""
    summary, entries = {"task": "t", "seed": 7}, []
    cov = html_cover(summary, {"tool": "autorefine"})
    none = html_report(summary, entries)
    default = html_report(summary, entries, cover=None)
    assert none == default  # byte-identical default path (G2)
    with_cov = html_report(summary, entries, cover=cov)
    assert with_cov != none  # the cover changes the output
    # the cover is at the top of the body, before the report heading
    body = with_cov.index("<body>")
    assert with_cov.index('<header class="cover">') > body
    assert with_cov.index("<h1>") > with_cov.index('<header class="cover">')
    # the no-cover path has no cover header
    assert '<header class="cover">' not in none


# --- 56.3 the design-token registry (A46) ------------------------------------

def test_resolve_tokens_default():
    """A46 (SPEC.md 56.3): the default call resolves to exactly ``dict``
    (``TOKENS``) — a NEW dict (the module globals are never touched), and the
    token set is the same 13 the ``_styled`` render context swaps."""
    out = resolve_tokens()
    assert out == dict(TOKENS)
    # it is a fresh dict, not the module object itself (safe to mutate)
    assert out is not TOKENS
    # the registry covers the full 13-token set the theme swaps
    assert set(TOKENS) == set(_TOKENS_FROM_KEY.values())
    assert len(TOKENS) == 13
    # spot-check a default against the module constant it aliases (_BG)
    assert TOKENS["bg"] == "white"


def test_resolve_tokens_okabe_dark():
    """A46 (SPEC.md 56.3): the ``okabe`` / ``dark`` override sets merge back
    onto the canonical names via ``_TOKENS_FROM_KEY`` (no case/name
    guessing); the okabe set is a *subset* (bg/axis keep the theme default);
    okabe wins where the two overlap (dark is applied first); validation
    mirrors ``_styled`` exactly (``ValueError`` on a bad palette / dark)."""
    # okabe: every okabe key resolves onto its token
    ok = resolve_tokens("okabe")
    for key, value in _OKABE.items():
        token = _TOKENS_FROM_KEY.get(key)
        if token is not None:
            assert ok[token] == value, key
    # okabe does NOT define bg/axis → they keep the default
    assert ok["bg"] == TOKENS["bg"] and ok["axis"] == TOKENS["axis"]
    # dark: every dark key resolves onto its token
    dk = resolve_tokens(dark=True)
    for key, value in _DARK.items():
        token = _TOKENS_FROM_KEY.get(key)
        if token is not None:
            assert dk[token] == value, key
    # okabe + dark: on overlap okabe wins (dark applied first, okabe on top);
    # bg/axis come from the dark set (okabe does not define them)
    both = resolve_tokens("okabe", True)
    assert both["line"] == _OKABE["_LINE"]
    assert both["bg"] == _DARK["_BG"] and both["axis"] == _DARK["_AXIS"]
    # a token neither set defines stays at its default
    assert both["noscore_bg"] in (TOKENS["noscore_bg"], _DARK["_NOSCORE_BG"])
    # validation mirrors _styled exactly (51.4.1 / 51.4.2)
    with pytest.raises(ValueError):
        resolve_tokens("nope")
    with pytest.raises(ValueError):
        resolve_tokens(dark="yes")
    with pytest.raises(ValueError):
        resolve_tokens(dark=1)


# --- 56.5 reduced motion (A46) -----------------------------------------------

def test_svg_live_sparkline_reduced_motion():
    """A46 (SPEC.md 56.5): ``reduced_motion=True`` suppresses the pulsing
    halo (the app's one animated affordance) — the ``r="7"`` halo + soft
    ``opacity="0.28"`` are gone — while the plain dot (``r="4"``) and the
    ``current best`` title stay; the default call keeps the halo and is
    byte-identical to an explicit one (G2)."""
    series = [60.0, 72.5, 80.0]
    default = svg_live_sparkline(series)
    rm = svg_live_sparkline(series, reduced_motion=True)
    _assert_xml(default)
    _assert_xml(rm)
    # the default keeps the halo
    assert 'r="7"' in default and 'opacity="0.28"' in default
    # reduced motion drops the halo but keeps the dot + title
    assert 'r="7"' not in rm
    assert 'opacity="0.28"' not in rm
    assert 'r="4"' in rm  # the plain dot in both
    assert "current best 80.00 (live)" in rm
    # the default call is byte-identical to an explicit one (G2)
    assert default == svg_live_sparkline(series, palette="default",
                                         dark=False, reduced_motion=False)
    # static (live=False) already has no halo; reduced motion is a no-op there
    assert 'r="7"' not in svg_live_sparkline(series, live=False)


# --- 56.4 the app wiring (A46) -----------------------------------------------

def test_app_source_wires_56():
    """A46 (SPEC.md 56.4): the app wires all six A items — the empty
    (Setup), loading (Run), and error (points at ``autorefine doctor``)
    states; the "Provenance (certificate)" + "Design tokens" expanders (no
    ``key`` — this Streamlit build's ``st.expander`` has none); the "Reduce
    motion" checkbox; the reduced-motion passthrough at both sparkline sites;
    and the provenance/token renders. All default no-op (byte-identical)."""
    src = APP.read_text(encoding="utf-8")
    for token in (
        # the three explicit states (56.4)
        '"No data loaded',
        '"RUNNING — the improvement loop',
        'autorefine doctor',
        # the two new expanders (labels; NO key on st.expander here)
        'st.expander("Provenance (certificate)"',
        'st.expander("Design tokens"',
        # the reduce-motion checkbox + its passthrough at both sites (56.5)
        'key="reduce_motion"',
        'reduced_motion = st.session_state.get("reduce_motion", False)',
        "reduced_motion=reduced_motion",
        # the design-token registry render (56.3)
        "resolve_tokens(",
        # the provenance certificate render (56.1)
        "svg_provenance(",
        "provenance_card(",
        "from autorefine.provenance import (",
    ):
        assert token in src, token
    # the reduce-motion checkbox is off by default (byte-identical)
    assert 'value=False, key="reduce_motion"' in src


def test_cli_source_wires_certificate():
    """A46 (SPEC.md 56.1.1): the CLI wires ``report --certificate`` — the
    provenance imports, the mutual-exclusivity guards (``--json`` /
    ``--history``), the payload build + card print + cover into
    ``html_report`` (all under the flag, so the default path is
    byte-identical), and the ``--certificate`` parser argument."""
    src = CLI.read_text(encoding="utf-8")
    for token in (
        "from .provenance import (",
        "    env_provenance,",
        "    provenance_card,",
        "    provenance_payload,",
        '"--certificate and --json are mutually exclusive',
        '"--certificate and --history are mutually exclusive',
        "provenance_card(cert_payload)",
        "html_cover(summary, cert_payload)",
        "cover=cert_cover",
        'p_rep.add_argument("--certificate"',
    ):
        assert token in src, token
    # the certificate path is gated (the default report stays byte-identical)
    assert 'if getattr(args, "certificate", False):' in src


# --- 56.4 the app end-to-end (A46) -------------------------------------------

def test_app_default_path_end_to_end(tmp_path):
    """A46 (SPEC.md 56.4): the *default* path runs end-to-end — a finished
    run persists its result, and the three new Setup-tab controls (the
    "Provenance (certificate)" + "Design tokens" expanders and the "Reduce
    motion" checkbox) are present without breaking the pre-v0.42 surface.
    (The expanders are matched by label — this Streamlit build's
    ``st.expander`` has no ``key``; the checkbox has one.)"""
    pytest.importorskip("streamlit", reason="dashboard app is optional")
    from streamlit.testing.v1 import AppTest

    # a small classification CSV (quadrant XOR, 2 classes)
    lines = ["a,b,churn"]
    for i in range(60):
        a = (i % 5) / 5.0
        b = ((i // 5) % 4) / 4.0
        churn = 1.0 if (a > 0.4) ^ (b > 0.4) else 0.0
        lines.append(f"{a:.2f},{b:.2f},{churn:.0f}")
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    at = AppTest.from_file(str(APP), default_timeout=300)
    at.run()
    assert not at.exception
    at.text_input(key="csv_path").set_value(str(csv_path))
    at.text_input(key="runs_dir").set_value(str(tmp_path / "runs"))
    at.number_input(key="experiments").set_value(2)
    at.number_input(key="max_train").set_value(5.0)
    at.run()
    assert not at.exception
    at.button(key="run_button").set_value(True).run()
    assert not at.exception, at.exception

    # the default path finished and persisted its result
    assert at.session_state["result"] is not None
    # the two new expanders render (matched by label — no key on st.expander)
    exp_labels = [str(e.label) for e in at.expander]
    assert "Provenance (certificate)" in exp_labels, exp_labels
    assert "Design tokens" in exp_labels, exp_labels
    # the reduce-motion checkbox is present (default off) + has a key
    assert any(e.key == "reduce_motion" for e in at.checkbox), \
        "the Reduce motion checkbox renders"


# --- A46 round regression -----------------------------------------------------

def test_version_round_v042():
    """A46 (SPEC.md 56.6, 33.1): the version stepped to ``0.42.0`` in both
    sources (v0.42 ⇒ ``0.42.0``, M45, SPEC.md 56)."""
    py = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert py["project"]["version"] == autorefine.__version__ == "0.42.0"


def test_spec_cites_a46_and_round():
    """A46 (SPEC.md 56.6/56.7): SPEC.md defines the A46 acceptance block and
    the M45 index row + milestone — the A25 index machinery reads both
    (defined == set(range(1, 47)) includes this round)."""
    spec = (REPO / "SPEC.md").read_text(encoding="utf-8")
    assert "### 56.6 Acceptance (A46)" in spec
    assert re.search(r"^\s*\| M45 \| v0\.42\s*\|\s*56\s*\|\s*A46\s*\|",
                     spec, re.MULTILINE)
    assert "**M45**" in spec
