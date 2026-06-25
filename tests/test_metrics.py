"""Metric-suite unit tests."""

from evaluation.metrics import Example, evaluate


def test_perfect_prediction():
    ex = Example(predicted=["base_forward(0.5)"],
                 reference=["base_forward(0.5)"])
    r = evaluate([ex])
    assert r.em == 1.0 and r.sm == 1.0
    assert r.mvr == 0.0 and r.hallucination_rate == 0.0


def test_set_match_ignores_order():
    ex = Example(predicted=["grip_open", "lift_up(0.1)"],
                 reference=["lift_up(0.1)", "grip_open"])
    r = evaluate([ex])
    assert r.em == 0.0 and r.sm == 1.0


def test_tool_first_compliance():
    good = Example(predicted=["get_pointcloud_summary"],
                   reference=["get_pointcloud_summary"],
                   requires_tool_first=True)
    bad = Example(predicted=["base_forward(1.0)"],
                  reference=["get_pointcloud_summary"],
                  requires_tool_first=True)
    assert evaluate([good]).tfc == 1.0
    assert evaluate([bad]).tfc == 0.0


def test_mode_violation_and_hallucination():
    ex = Example(predicted=["base_forward(0.5)", "teleport_home"],
                 reference=["nav_relative(forward,0.5)"],
                 mode="navigation")
    r = evaluate([ex])
    assert r.mvr == 1.0
    assert r.hallucination_rate == 1.0


def test_level_accuracy():
    exs = [Example(predicted=["pick_object(cup)"],
                   reference=["pick_object(cup)"],
                   predicted_level=2, reference_level=2),
           Example(predicted=["lift_up(0.1)"], reference=["lift_up(0.1)"],
                   predicted_level=1, reference_level=2)]
    assert evaluate(exs).level_accuracy == 0.5
