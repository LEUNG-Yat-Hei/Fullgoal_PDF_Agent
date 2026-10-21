import functools
import logging
import sys
import time
from typing import Any, Callable

from langchain.tools import BaseTool


class _Ansi:
    """Minimal ANSI color codes."""

    RESET = "[0m"
    YELLOW = "[33m"
    GREEN = "[32m"
    RED = "[31m"
    GRAY = "[90m"


def _color(text: Any, code: str) -> str:
    return f"{code}{text}{_Ansi.RESET}"


def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # Re-running the cell (or building several ToolLoggers) must not stack handlers and print each line twice
    if not logger.handlers:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(handler)
    logger.propagate = False
    return logger

class ToolLogger:
    """Decorator that logs each call to a tool: its arguments, duration, and result or error.

    Use it on a BaseTool subclass (wraps `_run`) or on a plain function (e.g. under `@tool`).
    """

    def __init__(self, name: str = "tools", preview_len: int = 300):
        self.logger = setup_logger(name)
        self.preview_len = preview_len

    def __call__(self, target):
        if isinstance(target, type) and issubclass(target, BaseTool):
            run = target._run

            @functools.wraps(run)
            def _run(tool, *args, **kwargs):
                return self._logged(tool.name, lambda: run(tool, *args, **kwargs), args, kwargs)

            target._run = _run
            return target

        @functools.wraps(target)
        def wrapper(*args, **kwargs):
            return self._logged(target.__name__, lambda: target(*args, **kwargs), args, kwargs)

        return wrapper

    def _logged(self, name: str, call: Callable[[], Any], args: tuple, kwargs: dict) -> Any:
        shown = {**{f"arg{i}": a for i, a in enumerate(args)}, **kwargs}
        self.logger.info(f"{_color('▶', _Ansi.YELLOW)} {_color(name, _Ansi.YELLOW)} {self._preview(shown)}")
        start = time.perf_counter()
        try:
            result = call()
        except Exception as e:
            elapsed = time.perf_counter() - start
            self.logger.error(f"{_color('✗', _Ansi.RED)} {_color(name, _Ansi.RED)} {elapsed:.2f}s {type(e).__name__}: {e}")
            raise
        elapsed = time.perf_counter() - start
        self.logger.info(f"{_color('✓', _Ansi.GREEN)} {name} {_color(f'{elapsed:.2f}s', _Ansi.GRAY)} {self._preview(result)}")
        return result

    def _preview(self, value: Any) -> str:
        text = repr(value)  # repr escapes newlines, so each log entry stays on one line
        if len(text) > self.preview_len:
            text = text[: self.preview_len] + f"… (+{len(text) - self.preview_len} chars)"
        return text


tool_logger = ToolLogger()