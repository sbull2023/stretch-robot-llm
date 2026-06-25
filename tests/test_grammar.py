"""Grammar invariants: the vocabulary must match the paper exactly."""

from stretch_llm.llm.grammar import (ALLOWED_COMMANDS, COMMAND_SPECS,
                                     HIGH_LEVEL_BEHAVIOURS, PERCEPTION_TOOLS,
                                     Mode, allowed_in_mode)


def test_vocabulary_size_matches_paper():
    assert len(ALLOWED_COMMANDS) == 47


def test_every_command_has_a_spec():
    for name in ALLOWED_COMMANDS:
        assert name in COMMAND_SPECS


def test_levels_partition():
    for name in HIGH_LEVEL_BEHAVIOURS:
        assert COMMAND_SPECS[name].level == 2
    n_l2 = sum(1 for s in COMMAND_SPECS.values() if s.level == 2)
    assert len(HIGH_LEVEL_BEHAVIOURS) == 6
    assert n_l2 == len(HIGH_LEVEL_BEHAVIOURS) + 1  # place_object is dispatcher-only


def test_perception_tools_legal_in_both_modes():
    for name in PERCEPTION_TOOLS:
        assert allowed_in_mode(name, "position")
        assert allowed_in_mode(name, "navigation")


def test_strict_mode_separation():
    assert not allowed_in_mode("base_forward", "navigation")
    assert not allowed_in_mode("nav_relative", "position")
    assert allowed_in_mode("stop", "position")
    assert allowed_in_mode("stop", "navigation")


def test_bounded_commands_have_sane_ranges():
    for spec in COMMAND_SPECS.values():
        if spec.bounds:
            lo, hi = spec.bounds
            assert lo < hi
            assert lo >= 0.0


def test_mode_enum_values():
    assert Mode.POSITION.value == "position"
    assert Mode.NAVIGATION.value == "navigation"
