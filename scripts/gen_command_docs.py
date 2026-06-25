#!/usr/bin/env python3
"""Regenerate docs/COMMANDS.md from the grammar module."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stretch_llm.llm.grammar import COMMAND_SPECS  # noqa: E402

HEADER = """# Command Reference

Auto-generated from `stretch_llm/llm/grammar.py` — edit the grammar, then
run `python scripts/gen_command_docs.py`.

| Command | Level | Mode | Params | Safe range | Description |
|---|---|---|---|---|---|
"""

rows = []
for spec in sorted(COMMAND_SPECS.values(),
                   key=lambda s: (s.level, s.mode.value, s.name)):
    if not spec.in_vocabulary:
        continue
    params = ", ".join(spec.params) if spec.params else "—"
    bounds = f"[{spec.bounds[0]}, {spec.bounds[1]}]" if spec.bounds else "—"
    rows.append(f"| `{spec.name}` | {spec.level} | {spec.mode.value} "
                f"| {params} | {bounds} | {spec.description or '—'} |")

footer = ("\n`place_object` exists as a dispatcher-only behaviour outside "
          "the vocabulary: the run loop routes placement keywords to it "
          "directly, so the LLM never emits it.\n")

out = Path(__file__).resolve().parents[1] / "docs" / "COMMANDS.md"
out.write_text(HEADER + "\n".join(rows) + "\n" + footer)
print(f"Wrote {out} ({len(rows)} commands)")
