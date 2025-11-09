"""Utility functions for formatting, printing, and visualization."""

from __future__ import annotations

import textwrap
from typing import Any


class Ansi:
    """ANSI escape sequences used for lightweight CLI styling."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    MAGENTA = "\033[35m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    BLUE = "\033[34m"
    GRAY = "\033[90m"


def stylize(text: str, *styles: str) -> str:
    """Apply ANSI styles when the terminal supports them."""
    if not styles:
        return text
    return "".join(styles) + text + Ansi.RESET


TERMINAL_WIDTH = 96


def ascii_panel(title: str, lines: list[str], *, color: str | None = None) -> str:
    """Render a simple ASCII panel with optional color."""
    wrapped_lines: list[str] = []
    for line in lines or [""]:
        if line is None:
            continue
        stripped = str(line).rstrip()
        if not stripped:
            wrapped_lines.append("")
            continue
        wrapped_lines.extend(
            textwrap.wrap(
                stripped,
                width=TERMINAL_WIDTH - 4,
                replace_whitespace=False,
            )
            or [""]
        )

    content_lines = wrapped_lines or [""]
    width = min(
        TERMINAL_WIDTH,
        max(len(title), *(len(item) for item in content_lines)) + 4,
    )
    border = "+" + "-" * (width - 2) + "+"
    header = f"| {title.center(width - 4)} |"
    separator = border
    body = [f"| {line.ljust(width - 4)} |" for line in content_lines]
    panel_lines = [border, header, separator, *body, border]
    if color:
        return "\n".join(stylize(line, color) for line in panel_lines)
    return "\n".join(panel_lines)


def wrap_text(text: str, *, indent: int = 0) -> str:
    """Wrap a free-form paragraph to the terminal width."""
    return textwrap.fill(
        text,
        width=TERMINAL_WIDTH,
        initial_indent=" " * indent,
        subsequent_indent=" " * indent,
    )


def render_quality_bar(score: float, width: int = 20) -> str:
    """Render a visual progress bar for a quality score."""
    filled = int(score * width)
    empty = width - filled
    return stylize("█" * filled, Ansi.GREEN) + stylize("░" * empty, Ansi.DIM)


def render_quality_metrics(metrics: Any, state: Any = None) -> str:
    """Visual quality dashboard showing all metrics."""
    lines = []
    lines.append(stylize("Quality Metrics", Ansi.BOLD, Ansi.CYAN))
    lines.append("")

    # Main metrics with bars
    lines.append(
        f"  Relevance:   {render_quality_bar(metrics.relevance_score)} {metrics.relevance_score:.2f}"
    )
    lines.append(
        f"  Coverage:    {render_quality_bar(metrics.coverage_score)} {metrics.coverage_score:.2f}"
    )
    lines.append(
        f"  Diversity:   {render_quality_bar(metrics.diversity_score)} {metrics.diversity_score:.2f}"
    )
    lines.append(
        f"  Confidence:  {render_quality_bar(metrics.confidence)} {metrics.confidence:.2f}"
    )
    lines.append(
        f"  Next Loop:   {render_quality_bar(metrics.improvement_potential)} {metrics.improvement_potential:.2f}"
    )

    # Add trend if we have history
    if state and len(state.quality_history) > 1:
        latest = state.quality_history[-1]
        previous = state.quality_history[-2]
        delta = latest - previous
        if delta > 0:
            trend = stylize(f"▲ +{delta:.2%}", Ansi.GREEN)
        elif delta < 0:
            trend = stylize(f"▼ {delta:.2%}", Ansi.RED)
        else:
            trend = stylize("● No change", Ansi.YELLOW)
        lines.append("")
        lines.append(f"  Trend:       {trend}")

    # Overall composite score
    if state and state.quality_history:
        overall = state.quality_history[-1]
        lines.append(f"  Overall:     {render_quality_bar(overall)} {overall:.2f}")

    return "\n".join(lines)
