"""Unit tests for the command parser and validator (no ROS required)."""

from stretch_llm.llm.parser import (clamp_command, extract_commands,
                                    extract_name, parse_command,
                                    validate_commands)


def test_extract_name_variants():
    assert extract_name("base_forward(0.2)") == "base_forward"
    assert extract_name("base_forward (0.2)") == "base_forward"
    assert extract_name("base_forward") == "base_forward"


def test_parse_numeric():
    assert parse_command("base_forward(0.2)") == ("base_forward", 0.2)


def test_parse_no_params():
    assert parse_command("grip_close()") == ("grip_close", None)
    assert parse_command("grip_close") == ("grip_close", None)


def test_parse_mixed_params():
    name, params = parse_command("nav_relative(forward, 0.5)")
    assert name == "nav_relative"
    assert params == ("forward", 0.5)


def test_parse_never_executes_code():
    # The v27 parser used eval(); a hostile argument must stay a string.
    name, params = parse_command("nav_to_named(__import__)")
    assert params == "__import__"


def test_extract_commands_from_raw_response():
    raw = "Sure! base_forward(0.5), then grip_open"
    assert "base_forward(0.5)" in extract_commands(raw)
    assert any(c.startswith("grip_open") for c in extract_commands(raw))


def test_validate_rejects_hallucinations():
    out = validate_commands(["fly_to_moon(1.0)", "stop"], verbose=False)
    assert out == ["stop"]


def test_validate_rejects_mode_violation():
    out = validate_commands(["base_forward(0.5)"], mode="navigation",
                            verbose=False)
    assert out == []
    out = validate_commands(["nav_relative(forward,0.5)"], mode="navigation",
                            verbose=False)
    assert out == ["nav_relative(forward,0.5)"]


def test_clamp_out_of_range():
    assert clamp_command("base_forward(9.0)") == "base_forward(2.0)"
    assert clamp_command("base_forward(0.5)") == "base_forward(0.5)"
