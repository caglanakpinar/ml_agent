"""The benchmark cases are what they claim to be: each reference solution passes its checks (so the planted traps are real and the
problem is solvable), the data is deterministic, the committed files match the generators, the agent-visible file does not hold the
answer, and every case names sub agents that exist and a command that the CLI accepts."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from benchmarks import run as bench
from benchmarks.cases import BY_ID, CASES
from console.topics import discover_topics
from ml_core.data import load_frame

IDS = [c.id for c in CASES]
GROUND_TRUTH_COLUMNS = {"is_fraud", "segment", "is_positive", "fault", "dominant_topic", "regime", "rul_true"}


@pytest.fixture(scope="module")
def generated():
    return {case.id: bench.frames(case) for case in CASES}


def test_at_least_ten_cases_with_unique_ids():
    assert len(CASES) >= 10
    assert len(set(IDS)) == len(IDS)


@pytest.mark.parametrize("case_id", IDS)
def test_reference_solution_passes_every_check(case_id, generated):
    result = BY_ID[case_id].reference(generated[case_id])
    failed = [name for name, ok in result["checks"].items() if not ok]
    assert not failed, f"{case_id}: {failed}\n{result['metrics']}"


@pytest.mark.parametrize("case_id", IDS)
def test_generation_is_deterministic(case_id, generated):
    again = bench.frames(BY_ID[case_id])
    for stem, frame in generated[case_id].items():
        pd.testing.assert_frame_equal(frame, again[stem])


@pytest.mark.parametrize("case_id", IDS)
def test_agent_visible_file_does_not_hold_the_truth(case_id, generated):
    case = BY_ID[case_id]
    assert "data" in generated[case_id]
    assert not GROUND_TRUTH_COLUMNS & set(generated[case_id]["data"].columns)
    for name in case.extra:
        assert name.removesuffix(".csv") in generated[case_id]


@pytest.mark.parametrize("case_id", IDS)
def test_cases_are_hard_enough(case_id, generated):
    case = BY_ID[case_id]
    data = generated[case_id]["data"]
    assert len(case.traps) >= 4, "a 'very complex' case plants several distinct difficulties"
    assert len(data) >= 5000 and data.shape[1] >= 4


@pytest.mark.parametrize("case_id", IDS)
def test_topics_exist(case_id):
    known = {f"{category}/{name}" for category, names in discover_topics().items() for name in names}
    assert set(BY_ID[case_id].topics) <= known


def test_cases_cover_many_problem_types():
    used = {topic for case in CASES for topic in case.topics}
    assert len(used) >= 12
    assert any(len(case.topics) > 1 for case in CASES), "at least one case needs several sub agents"


@pytest.mark.parametrize("case_id", IDS)
def test_committed_data_matches_the_generator(case_id, generated):
    folder = bench.case_dir(BY_ID[case_id])
    if not (folder / "data.csv").exists():
        pytest.skip("run `python -m benchmarks.run generate` first")
    for stem, frame in generated[case_id].items():
        fresh = hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()
        assert hashlib.sha256((folder / f"{stem}.csv").read_bytes()).hexdigest() == fresh, f"{stem}.csv is stale — run `python -m benchmarks.run generate`"
    assert (folder / "problem.md").exists() and (folder / "case.json").exists()
    assert len(load_frame(folder / "data.csv")) == len(generated[case_id]["data"])


def test_prompt_contains_problem_and_data_path(capsys):
    assert bench.main(["prompt", "c01"]) == 0
    out = capsys.readouterr().out
    assert "churned_next_30d" in out and str(bench.data_file(BY_ID["c01_churn_leakage_drift"])) in out


def test_solve_builds_one_cli_command_per_topic(monkeypatch, tmp_path):
    seen = []

    class FakeProc:
        stdout = iter(["ok\n"])

        def wait(self):
            return 0

    monkeypatch.setattr(bench, "RUNS", tmp_path)
    monkeypatch.setattr(bench.subprocess, "Popen", lambda command, **kw: (seen.append(command), FakeProc())[1])
    assert bench.main(["solve", "c13", "--yes"]) == 0
    assert [c[3] for c in seen] == BY_ID["c13_engine_rul_two_stage"].topics
    for command in seen:
        assert command[2] == "solve" and "--yes" in command and command[command.index("--data") + 1].endswith("c13_engine_rul_two_stage/data.csv")


def test_solve_arguments_are_accepted_by_the_cli():
    from click.testing import CliRunner

    from cli import cli

    result = CliRunner().invoke(cli, ["solve", "--help"])
    assert result.exit_code == 0
    for option in ("--data", "--question", "--problem", "--yes"):
        assert option in result.output
