"""Safe arithmetic calculator (no arbitrary code exec)."""

from __future__ import annotations

import ast
import operator as op
from typing import Any

_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.Mod: op.mod,
    ast.FloorDiv: op.floordiv,
}


def _eval(node: Any) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported expression")


def calculate(expression: str) -> str:
    expr = (expression or "").strip().replace("^", "**")
    if not expr or len(expr) > 120:
        return "Calculator: empty or too long expression."
    try:
        tree = ast.parse(expr, mode="eval")
        val = _eval(tree)
        return f"Calculator: {expression} = {val}"
    except Exception as e:
        return f"Calculator error: {e}"
