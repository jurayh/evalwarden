"""DATA-004 (train/eval leakage) spike tests.

Feasibility shape under test: the training file is a side input, loaded
strictly and handed to the check via with_options -- the channel the
differ already uses for its baseline model. No new artifact type.

Fixtures in the house style, over a 40-item eval whose first 30 items
passed and last 10 failed:
(a) planted exact copies, some passed, some failed -- the exact tier
    fires and contaminated wins are counted exactly;
(b) planted near-duplicates via token-level edits (adjacent swap,
    sentence reorder, substitution, a changed number) -- the
    near-duplicate tier fires and names the matched pairs;
(c) a topical control: same instruction boilerplate, same topic
    vocabulary, different questions -- the check stays silent;
(d) no training file supplied -- the check stays silent.
Plus the loader's strictness contract and an end-to-end audit/CLI run.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from evalwarden.canonical import dumps_canonical
from evalwarden.checks import REGISTRY
from evalwarden.checks.data import LeakageCheck
from evalwarden.cli import app
from evalwarden.engine import audit
from evalwarden.leakage import (
    TrainingFileError,
    TrainingRecord,
    item_text,
    leakage_stats,
    load_training_csv,
    load_training_file,
    load_training_jsonl,
)
from evalwarden.model import Attempt, Confidence, Severity, TaskSample

from .conftest import make_model

BOILER = (
    "Read the following item carefully and answer the question. "
    "Choose the best answer and give only the answer term."
)
TOPICS = [
    ("geography", "river capital mountain country border ocean desert valley coast island"),
    ("biology", "cell species evolution gene protein organism habitat tissue enzyme bacteria"),
    ("physics", "force energy velocity mass friction gravity circuit voltage photon momentum"),
    ("statistics", "sample mean variance distribution hypothesis regression correlation median quartile outlier bias"),
]
BODIES = [
    "How does the {a} relate to the {b} in this scenario. Consider the role of the {c} and the {d} before answering. Name the {e} that best explains the {f}.",
    "A student claims the {a} causes the {b} to change over time. Which {c} provides the strongest evidence about the {d}. Explain using the {e} and the {f}.",
    "Compare the {a} with the {b} in terms of the {c}. Which statement about the {d} is most accurate given the {e} and the {f}.",
    "In a study of the {a}, researchers measured the {b} under several conditions. What happens to the {c} when the {d} increases. Refer to the {e} and the {f}.",
]
N_ITEMS = 40
N_PASS = 30


def _eval_items() -> list[tuple[str, str, str]]:
    """(task_id, prompt, target) for the 40-item eval."""
    items = []
    for i in range(N_ITEMS):
        topic, words = TOPICS[i // 10]
        vocab = words.split()
        vals = {k: vocab[(i + j) % len(vocab)] for j, k in enumerate("abcdef")}
        prompt = f"Item {i}. {BOILER} " + BODIES[i % len(BODIES)].format(**vals)
        items.append((f"t{i:02d}", prompt, vocab[i % len(vocab)]))
    return items


def _model():
    items = _eval_items()
    tasks = [TaskSample(id=tid, prompt=p, target=t) for tid, p, t in items]
    attempts = [
        Attempt(
            task_id=tid,
            status="pass" if i < N_PASS else "fail",
            score=1.0 if i < N_PASS else 0.0,
        )
        for i, (tid, _, _) in enumerate(items)
    ]
    return make_model(tasks=tasks, attempts=attempts)


def _by_id() -> dict[str, tuple[str, str]]:
    return {tid: (p, t) for tid, p, t in _eval_items()}


def _swap(text: str) -> str:
    toks = text.split()
    toks[20], toks[21] = toks[21], toks[20]
    return " ".join(toks)


def _reorder(text: str) -> str:
    parts = [p.strip() for p in text.split(".") if p.strip()]
    parts[2], parts[3] = parts[3], parts[2]
    return ". ".join(parts) + "."


_SUBS = {"relate": "connect", "causes": "makes", "Compare": "Contrast",
         "researchers": "scientists", "measured": "recorded", "role": "part"}


def _substitute(text: str) -> str:
    toks = text.split()
    done = 0
    for i, tok in enumerate(toks):
        core = tok.strip(".:")
        if core in _SUBS and done < 3:
            toks[i] = tok.replace(core, _SUBS[core])
            done += 1
    return " ".join(toks)


def _topical_control(topic_words: str, idx: int) -> TrainingRecord:
    vocab = topic_words.split()
    a, b, c = vocab[idx % 10], vocab[(idx + 3) % 10], vocab[(idx + 5) % 10]
    prompt = (
        f"Item {idx}. {BOILER} Name a well known example of a {a} and "
        f"describe where it is usually found. Explain why the {b} matters "
        f"to people who study the {c} in school textbooks today."
    )
    return TrainingRecord(id=f"ctrl-{idx}", prompt=prompt)


# -- (a) exact tier ---------------------------------------------------------


def test_a_exact_tier_fires_and_counts_contaminated_wins():
    by_id = _by_id()
    training = [
        TrainingRecord(id=f"ft-{tid}", prompt=by_id[tid][0], target=by_id[tid][1])
        for tid in ("t00", "t01", "t02", "t31", "t32")  # 3 passed, 2 failed
    ]
    training += [_topical_control(TOPICS[2][1], 100 + k) for k in range(8)]
    findings = LeakageCheck(training).run(_model())
    assert len(findings) == 1
    finding = findings[0]
    assert finding.id == "DATA-004"
    assert "exact" in finding.title
    assert "3 contaminated win(s)" in finding.title
    assert "10% of all passes" in finding.title  # 3 of 30 passes
    assert finding.severity is Severity.ERROR  # share 0.10 >= 0.10
    assert finding.confidence is Confidence.HIGH
    assert any("ft-t31" in e and "failed" in e for e in finding.evidence)
    assert any("contaminated wins: 3" in e for e in finding.evidence)


# -- (b) near-duplicate tier ---------------------------------------------------


def test_b_near_tier_fires_on_token_edits_and_names_pairs():
    by_id = _by_id()
    training = [
        TrainingRecord(id="ft-swap", prompt=_swap(by_id["t05"][0]), target=by_id["t05"][1]),
        TrainingRecord(id="ft-reorder", prompt=_reorder(by_id["t06"][0]), target=by_id["t06"][1]),
        TrainingRecord(id="ft-sub", prompt=_substitute(by_id["t33"][0]), target=by_id["t33"][1]),
    ]
    findings = LeakageCheck(training).run(_model())
    assert len(findings) == 1
    finding = findings[0]
    assert "near-duplicate" in finding.title
    assert "2 contaminated win(s)" in finding.title  # t05, t06 passed; t33 failed
    assert finding.severity is Severity.HIGH  # 2/30 = 6.7% >= 3%
    assert finding.confidence is Confidence.MEDIUM
    for train_id in ("ft-swap", "ft-reorder", "ft-sub"):
        assert any(train_id in e for e in finding.evidence)


def test_b_number_change_is_a_near_duplicate():
    by_id = _by_id()
    prompt, target = by_id["t07"]
    changed = prompt.replace("Item 7.", "Item 99.", 1)
    assert changed != prompt
    training = [TrainingRecord(id="ft-num", prompt=changed, target=target)]
    findings = LeakageCheck(training).run(_model())
    assert len(findings) == 1
    assert "near-duplicate" in findings[0].title
    assert any("ft-num" in e and "t07" in e for e in findings[0].evidence)


# -- (c) topical control -------------------------------------------------------


def test_c_topical_control_stays_silent():
    training = [
        _topical_control(words, ti * 10 + k)
        for ti, (_, words) in enumerate(TOPICS)
        for k in range(10)
    ]
    assert LeakageCheck(training).run(_model()) == []
    # And at the stats level: no control row crosses either threshold.
    model = _model()
    stats = leakage_stats(
        {t.id: item_text(t.prompt, t.target) for t in model.tasks},
        training,
        {t.id for t in model.tasks[:N_PASS]},
    )
    assert stats.exact == [] and stats.near == []


def test_c2_same_template_different_content_stays_silent():
    # Regression for the scaffolding discount: a training row built on
    # the SAME item template with different slot words is not a copy of
    # any eval item, but on raw shingles it reaches Jaccard 0.45 /
    # containment 0.62 against same-template items -- scaffolding, not
    # leakage. Over distinctive shingles it must stay silent.
    items = _eval_items()
    training = []
    for i, (tid, prompt, target) in enumerate(items[:10]):
        topic, words = TOPICS[i // 10]
        vocab = words.split()
        vals = {k: vocab[(i + 1 + j) % len(vocab)] for j, k in enumerate("abcdef")}
        twin_prompt = f"Item {i}. {BOILER} " + BODIES[i % len(BODIES)].format(**vals)
        assert twin_prompt != prompt
        training.append(TrainingRecord(id=f"twin-{tid}", prompt=twin_prompt, target=target))
    assert LeakageCheck(training).run(_model()) == []


# -- (d) silence without a training file ---------------------------------------


def test_d_silent_without_training_file():
    assert LeakageCheck().run(_model()) == []
    assert LeakageCheck([]).run(_model()) == []
    # In a full registry audit (no options), DATA-004 never fires.
    model = _model()
    fired = [f for check in REGISTRY for f in check.run(model) if f.id == "DATA-004"]
    assert fired == []


def test_lone_failed_leak_is_below_the_fire_floor():
    by_id = _by_id()
    training = [
        TrainingRecord(id="ft-one", prompt=by_id["t35"][0], target=by_id["t35"][1])
    ]
    assert LeakageCheck(training).run(_model()) == []


# -- stats-level behavior -------------------------------------------------------


def test_exact_match_uses_normalized_text():
    stats = leakage_stats(
        {"e1": "What is the River, exactly?"},
        [TrainingRecord(id="r1", prompt="what is the river exactly")],
        passed_ids={"e1"},
    )
    assert len(stats.exact) == 1
    assert stats.contaminated_wins == 1
    assert stats.contaminated_share == 1.0


def test_short_item_embedded_in_long_row_fires_via_containment():
    short = "alpha beta gamma delta epsilon zeta eta theta"
    long_row = (
        "training example with a long preamble about many unrelated "
        "things before the actual question begins here: " + short +
        " and then a long trailing explanation covering several other "
        "topics in considerable depth and detail for completeness"
    )
    stats = leakage_stats(
        {"e1": short}, [TrainingRecord(id="r1", prompt=long_row)], passed_ids=set()
    )
    assert stats.exact == []
    assert len(stats.near) == 1
    assert stats.near[0].containment == 1.0
    assert stats.near[0].jaccard < 0.45  # containment, not jaccard, caught it


def test_exact_twin_wins_over_near_match():
    stats = leakage_stats(
        {"e1": "one two three four five six seven eight nine ten"},
        [
            TrainingRecord(id="near-row", prompt="one two three four five six seven eight nine eleven"),
            TrainingRecord(id="exact-row", prompt="One two three four five six seven eight nine ten"),
        ],
        passed_ids=set(),
    )
    assert [m.train_id for m in stats.exact] == ["exact-row"]
    assert stats.near == []


# -- loader strictness -----------------------------------------------------------


def test_loader_jsonl_defaults_and_targets():
    records = load_training_jsonl(
        '{"prompt": "q one", "target": "a"}\n\n{"id": "x", "prompt": "q two"}\n',
        "train.jsonl",
    )
    assert [r.id for r in records] == ["train-1", "x"]
    assert records[0].target == "a"
    assert records[1].target is None


def test_loader_jsonl_errors_name_the_line():
    with pytest.raises(TrainingFileError, match="line 2"):
        load_training_jsonl('{"prompt": "ok"}\nnot json\n', "train.jsonl")
    with pytest.raises(TrainingFileError, match="line 1.*'prompt'"):
        load_training_jsonl('{"text": "no prompt field"}\n', "train.jsonl")
    with pytest.raises(TrainingFileError, match="duplicate"):
        load_training_jsonl(
            '{"id": "x", "prompt": "a"}\n{"id": "x", "prompt": "b"}\n', "train.jsonl"
        )


def test_loader_csv_and_errors():
    records = load_training_csv("id,prompt,target\nr1,hello world,ans\n,second row,\n", "t.csv")
    assert [r.id for r in records] == ["r1", "train-3"]
    assert records[1].target is None
    with pytest.raises(TrainingFileError, match="'prompt' column"):
        load_training_csv("text,target\nhello,ans\n", "t.csv")
    with pytest.raises(TrainingFileError, match="row 2.*'prompt'"):
        load_training_csv("prompt,target\n,ans\n", "t.csv")


def test_loader_file_formats(tmp_path: Path):
    good = tmp_path / "train.jsonl"
    good.write_text('{"prompt": "q"}\n', encoding="utf-8")
    assert len(load_training_file(good)) == 1
    bad = tmp_path / "train.txt"
    bad.write_text('{"prompt": "q"}\n', encoding="utf-8")
    with pytest.raises(TrainingFileError, match="unsupported"):
        load_training_file(bad)
    with pytest.raises(TrainingFileError, match="no such training file"):
        load_training_file(tmp_path / "missing.jsonl")


# -- engine + CLI end to end ------------------------------------------------------


def _write_artifact(tmp_path: Path) -> Path:
    artifact = tmp_path / "eval.json"
    artifact.write_text(dumps_canonical(_model()), encoding="utf-8")
    return artifact


def _write_training(tmp_path: Path) -> Path:
    by_id = _by_id()
    train = tmp_path / "train.jsonl"
    rows = [
        {"id": f"ft-{tid}", "prompt": by_id[tid][0], "target": by_id[tid][1]}
        for tid in ("t00", "t31")  # one passed, one failed: 1 contaminated win
    ]
    train.write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    return train


def test_engine_audit_with_and_without_training_file(tmp_path: Path):
    artifact = _write_artifact(tmp_path)
    train = _write_training(tmp_path)
    with_file = audit(artifact, training_file=train)
    leaked = [f for f in with_file.findings if f.id == "DATA-004"]
    assert len(leaked) == 1
    assert leaked[0].severity is Severity.HIGH  # 1/30 = 3.3% >= 3%
    without = audit(artifact)
    assert [f for f in without.findings if f.id == "DATA-004"] == []


def test_engine_malformed_training_file_is_audit_error(tmp_path: Path):
    artifact = _write_artifact(tmp_path)
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"text": "no prompt"}\n', encoding="utf-8")
    from evalwarden.adapters import AuditError

    with pytest.raises(AuditError, match="'prompt'"):
        audit(artifact, training_file=bad)


runner = CliRunner()


def test_cli_audit_training_file_flag(tmp_path: Path):
    artifact = _write_artifact(tmp_path)
    train = _write_training(tmp_path)
    out = tmp_path / "report.html"
    result = runner.invoke(
        app,
        ["audit", str(artifact), "--training-file", str(train), "--output", str(out)],
    )
    assert result.exit_code == 1, result.output  # HIGH finding, fail-on high
    assert "DATA-004" in result.output
    listed = runner.invoke(app, ["checks"])
    assert "DATA-004" in listed.output
    explained = runner.invoke(app, ["explain", "DATA-004"])
    assert explained.exit_code == 0
    assert "Training data leaks into the eval set" in explained.output
