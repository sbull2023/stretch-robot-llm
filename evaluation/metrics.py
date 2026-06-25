"""Grounding metrics for the experimental evaluation (Section 7).

* **EM**  -- exact match: the predicted command list equals the
  reference, command for command and argument for argument.
* **SM**  -- set match: the predicted and reference command sets agree
  when order is ignored.
* **TFC** -- tool-first compliance: on instructions that require a
  perception check before action, the first predicted command is a
  perception tool.
* **MVR** -- mode-violation rate: fraction of predictions that contain a
  command incompatible with the active control mode.
* **Hallucination rate** -- fraction of predictions with at least one
  command outside the closed vocabulary.
* **Level accuracy** -- agreement between the predicted level token and
  the level of the logged trajectory.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from stretch_llm.llm.grammar import (ALLOWED_COMMANDS, PERCEPTION_TOOLS,
                                     allowed_in_mode)
from stretch_llm.llm.parser import extract_name, parse_command


def _normalise(cmd: str):
    name, params = parse_command(cmd)
    if isinstance(params, float):
        params = round(params, 3)
    elif isinstance(params, tuple):
        params = tuple(round(p, 3) if isinstance(p, float) else p
                       for p in params)
    return name, params


@dataclass
class Example:
    predicted: List[str]
    reference: List[str]
    mode: str = "position"
    requires_tool_first: bool = False
    predicted_level: Optional[int] = None
    reference_level: Optional[int] = None


@dataclass
class Report:
    em: float
    sm: float
    tfc: float
    mvr: float
    hallucination_rate: float
    level_accuracy: Optional[float]
    n: int

    def as_dict(self):
        return self.__dict__.copy()


def exact_match(ex: Example) -> bool:
    return [_normalise(c) for c in ex.predicted] == \
           [_normalise(c) for c in ex.reference]


def set_match(ex: Example) -> bool:
    return {_normalise(c) for c in ex.predicted} == \
           {_normalise(c) for c in ex.reference}


def tool_first_compliant(ex: Example) -> bool:
    if not ex.requires_tool_first:
        return True
    return bool(ex.predicted) and \
        extract_name(ex.predicted[0]) in PERCEPTION_TOOLS


def mode_violation(ex: Example) -> bool:
    return any(
        extract_name(c) in ALLOWED_COMMANDS
        and not allowed_in_mode(extract_name(c), ex.mode)
        for c in ex.predicted
    )


def hallucinated(ex: Example) -> bool:
    return any(extract_name(c) not in ALLOWED_COMMANDS for c in ex.predicted)


def evaluate(examples: List[Example]) -> Report:
    n = len(examples)
    if n == 0:
        raise ValueError("Empty evaluation set")
    tool_cases = [e for e in examples if e.requires_tool_first]
    level_cases = [e for e in examples
                   if e.predicted_level is not None
                   and e.reference_level is not None]
    return Report(
        em=sum(exact_match(e) for e in examples) / n,
        sm=sum(set_match(e) for e in examples) / n,
        tfc=(sum(tool_first_compliant(e) for e in tool_cases)
             / len(tool_cases)) if tool_cases else 1.0,
        mvr=sum(mode_violation(e) for e in examples) / n,
        hallucination_rate=sum(hallucinated(e) for e in examples) / n,
        level_accuracy=(sum(e.predicted_level == e.reference_level
                            for e in level_cases) / len(level_cases))
        if level_cases else None,
        n=n,
    )
