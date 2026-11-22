"""Plot pieces every study shares: one set of colours, the top-down shelf outline, rate and category maps."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

PANEL_THICKNESS_M = 0.015  # shelf walls, as built in so101sim/shelf_layout.py

# Status colours: outcomes. Each also has its own marker, so colour never carries meaning alone.
OUTCOME_STYLE = {
    "slid": ("#0ca30c", "o", "slid, max tilt ≤ 5°"),
    "leaned": ("#fab219", "D", "leaned 5–20°, stayed up"),
    "toppled": ("#d03b3b", "X", "toppled (> 20°)"),
    "glitch": ("#b5b5b0", "s", "simulator glitch (excluded)"),
}
# Categorical colours, fixed order: push directions, then anything else.
DIRECTION_COLOURS = {"away": "#2a78d6", "left": "#eb6834", "right": "#1baf7a"}
CATEGORY_COLOURS = ["#0ca30c", "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6", "#e34948",
                    "#8a8a86"]
# Sequential blue ramp for an ordered quantity such as friction.
BLUE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
RATE_COLOURMAP = LinearSegmentedColormap.from_list("rate", ["#e8f1fc", "#0d366b"])
REFERENCE_COLOURS = {"theory": "#2a78d6", "limit": "#6250d6", "neutral": "#8a8a86", "danger": "#d03b3b"}
SHELF_COLOUR = "#8a7a5a"


def tidy(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def save(fig, path: Path, dpi: int = 130) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def draw_shelf(ax, layout, action_box=None) -> None:
    """Top-down shelf: walls, the front edge, and the env's action box dashed. Plotted as (y, x), robot's left on
    the left and x (away from the robot) up the page, robot below."""
    t, half, front, back = PANEL_THICKNESS_M, layout.half_width, layout.front_x, layout.back_x
    wall = dict(fc=SHELF_COLOUR, ec="none")
    ax.add_patch(Rectangle((half, front), t, back - front + t, **wall))
    ax.add_patch(Rectangle((-half - t, front), t, back - front + t, **wall))
    ax.add_patch(Rectangle((-half - t, back), 2 * half + 2 * t, t, **wall))
    ax.plot([-half, half], [front, front], color=SHELF_COLOUR, lw=2, ls="--")
    ax.text(0, front - 0.006, "front edge  ·  robot this side ↓", ha="center", va="top", fontsize=8, color="#444")
    ax.text(half + t / 2, (front + back) / 2, "left wall", rotation=90, ha="center", va="center", fontsize=7,
            color="white")
    ax.text(-half - t / 2, (front + back) / 2, "right wall", rotation=90, ha="center", va="center", fontsize=7,
            color="white")
    if action_box is not None:
        x_low, x_high, y_low, y_high = action_box
        ax.add_patch(Rectangle((y_low, x_low), y_high - y_low, x_high - x_low, fc="none", ec="#222",
                               ls=(0, (4, 3)), lw=1.4, zorder=5,
                               label="env's action box (refuses picks outside, 5 mm slack)"))
    ax.set_xlim(half + t + 0.01, -half - t - 0.01)
    ax.set_ylim(front - 0.03, back + t + 0.01)
    ax.set_aspect("equal")
    ax.set_xlabel("y (m)  —  robot's left ←   → robot's right")
    ax.set_ylabel("x (m), away from the robot")


def rate_map(path: Path, rates: dict[tuple[float, float], float], cell_m: float, layout, action_box,
             title: str) -> None:
    """Each (x, y) cell coloured by a 0-1 rate on one fixed scale, over the shelf outline."""
    fig, ax = plt.subplots(figsize=(8.5, 6.4))
    for (x, y), rate in rates.items():
        ax.add_patch(Rectangle((y - cell_m / 2, x - cell_m / 2), cell_m, cell_m, fc=RATE_COLOURMAP(rate),
                               ec="white", lw=0.6))
    draw_shelf(ax, layout, action_box)
    mappable = plt.cm.ScalarMappable(cmap=RATE_COLOURMAP, norm=plt.Normalize(0, 100))
    fig.colorbar(mappable, ax=ax, label="success rate (%)", shrink=0.8)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.12), fontsize=7, frameon=False)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    save(fig, path, dpi=140)


def category_map(path: Path, categories: dict[tuple[float, float], str], cell_m: float, layout, action_box,
                 title: str, first: str = "success") -> None:
    """Each (x, y) cell coloured by a category, `first` getting the first colour, the rest by frequency."""
    counts: dict[str, int] = {}
    for value in categories.values():
        counts[value] = counts.get(value, 0) + 1
    order = ([first] if first in counts else []) + sorted((c for c in counts if c != first), key=lambda c: -counts[c])
    colour = {c: CATEGORY_COLOURS[min(i, len(CATEGORY_COLOURS) - 1)] for i, c in enumerate(order)}
    fig, ax = plt.subplots(figsize=(8.5, 6.8))
    for (x, y), value in categories.items():
        ax.add_patch(Rectangle((y - cell_m / 2, x - cell_m / 2), cell_m, cell_m, fc=colour[value], ec="white",
                               lw=0.6))
    draw_shelf(ax, layout, action_box)
    handles = ax.get_legend_handles_labels()[0]
    handles += [Patch(fc=colour[c], label=f"{c} ({counts[c]} cells)") for c in order]
    ax.legend(handles=handles, fontsize=7, loc="upper left", bbox_to_anchor=(0, -0.12), ncol=2, frameon=False)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    save(fig, path, dpi=140)


def outcome_legend_handles(include_refused: bool = True, include_touch: bool = True) -> list:
    handles = [Line2D([], [], color=c, marker=m, ls="", ms=9, label=label) for c, m, label in OUTCOME_STYLE.values()]
    if include_refused:
        handles.append(Line2D([], [], color="#888", marker="o", ls="", ms=9, mfc="#ddd", mec="#222", mew=1.8,
                              label="dark ring: the path check would refuse"))
    if include_touch:
        handles.append(Line2D([], [], color="#222", marker="v", ls="", ms=6, label="▼ arm touched the board"))
    return handles


def line_with_error(ax, xs, means, errors, colour: str, label: str) -> None:
    ax.errorbar(xs, means, yerr=errors, color=colour, marker="o", ms=6, lw=2, capsize=3, mec="white", label=label)
