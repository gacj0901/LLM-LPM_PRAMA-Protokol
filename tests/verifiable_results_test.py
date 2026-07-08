import json

from aptadynamic_llm.ingest import load_sessions
from aptadynamic_llm.omega import omega_sessions


def write_raw(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def token(logprob, entropy=0.0, gap=0.0):
    return {"top1_logprob": logprob, "entropy": entropy, "gap": gap}


def test_load_sessions_preserves_verifiable_raw_json_fields(tmp_path):
    write_raw(
        tmp_path / "raw.json",
        {
            "session_id": "session-a",
            "model": "fixture-model",
            "turns": [
                {
                    "finish_reason": "stop",
                    "tokens": [token(-0.25, entropy=0.2, gap=0.7), token(-0.50)],
                },
                {
                    "finish_reason": "length",
                    "tokens": [token(-1.25, entropy=0.4, gap=0.3)],
                },
            ],
        },
    )

    df = load_sessions(str(tmp_path))

    assert list(df["session_id"].unique()) == ["session-a"]
    assert list(df["generation_id"].unique()) == ["session-a:turn000", "session-a:turn001"]
    assert list(df["turn_index"]) == [0, 0, 1]
    assert list(df["pos"]) == [0, 1, 0]
    assert list(df["finish_reason"]) == ["stop", "stop", "length"]
    assert list(df["surprisal"]) == [0.25, 0.50, 1.25]
    assert df.loc[0, "entropy"] == 0.2
    assert df.loc[0, "gap"] == 0.7


def test_load_sessions_ignores_non_verifiable_json_inputs(tmp_path):
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    write_raw(tmp_path / "missing_turns.json", {"session_id": "bad", "model": "fixture"})

    df = load_sessions(str(tmp_path))

    assert df.empty


def test_omega_sessions_treats_raw_turns_as_independent_generations(tmp_path):
    write_raw(
        tmp_path / "raw.json",
        {
            "session_id": "session-a",
            "model": "fixture-model",
            "turns": [
                {"finish_reason": "stop", "tokens": [token(-0.1), token(-0.2)]},
                {"finish_reason": "length", "tokens": [token(-0.3), token(-0.4)]},
            ],
        },
    )
    df = load_sessions(str(tmp_path))

    emitted = omega_sessions(df, min_sessions=0)

    assert [row[0] for row in emitted] == ["session-a:turn000", "session-a:turn001"]
    assert [row[1] for row in emitted] == ["stop", "length"]
    assert emitted[0][2].tolist() == [0.1, 0.2]
    assert emitted[1][2].tolist() == [0.3, 0.4]
