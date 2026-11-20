"""Fixed smoke-test prompts for engineering checks (backend agreement, rollout benchmarks).

Hand-written, not taken from any benchmark (no contamination concern). They are never
used for training or evaluation.
"""

from __future__ import annotations

from qopd.types import Problem

SMOKE_PROBLEMS: tuple[str, ...] = (
    "What is 17 + 26?",
    "Compute 12 times 13.",
    "What is the remainder when 100 is divided by 7?",
    "Simplify the fraction 84/126.",
    "Solve for x: 3x + 5 = 20.",
    "What is the sum of the first 10 positive integers?",
    "A rectangle has sides 6 and 9. What is its area?",
    "What is 2 to the power 10?",
    "Find the greatest common divisor of 48 and 180.",
    "How many minutes are there in 3.5 hours?",
    "What is the least common multiple of 4, 6 and 10?",
    "If a train travels 150 km in 2 hours, what is its average speed in km/h?",
    "Solve for y: y^2 = 49 with y > 0.",
    "What is 15% of 80?",
    "A triangle has angles 50 and 60 degrees. What is the third angle?",
    "What is the next prime number after 31?",
    "Evaluate 7! / 5!.",
    "How many diagonals does a hexagon have?",
    "What is the square root of 196?",
    "Convert the binary number 101101 to decimal.",
)


def smoke_problems(n: int) -> list[Problem]:
    if n > len(SMOKE_PROBLEMS):
        raise ValueError(f"only {len(SMOKE_PROBLEMS)} smoke problems")
    return [Problem(f"smoke/{i}", q, "", "smoke") for i, q in enumerate(SMOKE_PROBLEMS[:n])]
