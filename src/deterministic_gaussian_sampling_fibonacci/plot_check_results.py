"""Plotting utility for the data saved by test.check_one (2D only).

This script reads the pickled results produced by
``deterministic_gaussian_sampling_fibonacci.test`` (see ``RESULTS_FILE`` in
that module) and plots every two-dimensional sample array found there.
Each run is given its own colour; within a run, the NumPy-based
implementation (``orig``) and the torch-based implementation (``new``) are
distinguished by marker style.

Usage examples (from project root)::

    # Generate some data first (optional: you can also import and call
    # check_one manually to only create a few entries)
    python -m deterministic_gaussian_sampling_fibonacci.test

    # Then create plots for all 2D results
    python -m deterministic_gaussian_sampling_fibonacci.plot_check_results

The script will open a single matplotlib window for 2D results.
Close the window to finish the program.
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Dict, List

import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, Button
from matplotlib import colors as mcolors
import numpy as np


HERE = os.path.dirname(__file__)
RESULTS_FILE = os.path.join(HERE, "check_results.pkl")


def _load_results(path: str) -> List[Dict[str, Any]]:
    """Load the list of result dictionaries from ``path``.

    The file is expected to have been created by ``test.py`` in the same
    directory.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Results file not found at {path!r}. Run the test script first "
            "(python -m deterministic_gaussian_sampling_fibonacci.test)."
        )

    with open(path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, list):
        raise TypeError(f"Unexpected data format in {path!r}: expected list, got {type(data)!r}")

    return data


def _is_2d_points(arr: np.ndarray) -> bool:
    return isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] == 2


def plot_all_results(results: List[Dict[str, Any]]) -> None:
    """Plot every 2D array from the given results list.

    Each element is expected to be a dict as produced by ``check_one`` in
    ``test.py``.
    """
    # Prepare figure for 2D data (created lazily).
    fig2d = None
    ax2d = None
    # Keep track of created 2D scatter artists for checkbox control.
    scatters2d = []
    labels2d = []  # full labels for legend and internal mapping
    checkbox_labels2d = []  # shortened labels for the checkboxes

    # Use a colormap to give each run its own colour.
    n_runs = len(results)
    if n_runs == 0:
        print("No results to plot.")
        return

    # Older matplotlib versions may not provide cm.get_cmap, so we construct a
    # list of colours manually from the "tab20" colormap.
    
    cmap = plt.colormaps["tab20"].resampled(n_runs)
    colors = colors = list(mcolors.TABLEAU_COLORS.values())

    i = 0
    colorUsed = False
    for idx, entry in enumerate(results):
        if(colorUsed is True):
            i = i+1
            colorUsed = False
        color = colors[i]
        dim = entry.get("dim")
        type_ = entry.get("type")
        sample_count = entry.get("sample_count")
        equal = entry.get("equal")

        label_prefix = f"run {idx} (dim={dim}, type={type_}, n={sample_count}, equal={equal})"

        orig = entry.get("orig")
        new = entry.get("new")

        # 2D plotting
        if orig is not None and _is_2d_points(orig):
            if fig2d is None:
                fig2d, ax2d = plt.subplots()
                ax2d.set_title("2D samples: orig (circle) vs new (cross)")
                ax2d.set_xlabel("x")
                ax2d.set_ylabel("y")
                ax2d.grid(True)
                ax2d.set_aspect("equal", adjustable="box")

            scatter_label_orig = f"{label_prefix} orig"
            scatter_orig = ax2d.scatter(
                orig[:, 0],
                orig[:, 1],
                color=color,
                marker="o",
                s=10,
                label=scatter_label_orig,
            )
            scatters2d.append(scatter_orig)
            labels2d.append(scatter_label_orig)
            # Short checkbox label: first 5 and last 4 characters of the full label.
            if len(scatter_label_orig) > 9:
                checkbox_labels2d.append(
                    f"{scatter_label_orig[:5]}...{scatter_label_orig[-4:]}"
                )
            else:
                checkbox_labels2d.append(scatter_label_orig)
            colorUsed = True

        if new is not None and _is_2d_points(new):
            if fig2d is None:
                fig2d, ax2d = plt.subplots()
                ax2d.set_title("2D samples: orig (circle) vs new (cross)")
                ax2d.set_xlabel("x")
                ax2d.set_ylabel("y")
                ax2d.grid(True)
                ax2d.set_aspect("equal", adjustable="box")
            scatter_label_new = f"{label_prefix} new"
            scatter_new = ax2d.scatter(
                new[:, 0],
                new[:, 1],
                color=color,
                marker="x",
                s=10,
                label=scatter_label_new,
            )
            scatters2d.append(scatter_new)
            labels2d.append(scatter_label_new)
            # Short checkbox label: first 5 and last 4 characters of the full label.
            if len(scatter_label_new) > 9:
                checkbox_labels2d.append(
                    f"{scatter_label_new[:5]}...{scatter_label_new[-4:]}"
                )
            else:
                checkbox_labels2d.append(scatter_label_new)
    # Add legend and checkbox widgets (if there is anything to show).
    if ax2d is not None and fig2d is not None and scatters2d:
        legend = ax2d.legend(
            fontsize="small",
            markerscale=1.0,
            bbox_to_anchor=(1.04, 1),
            loc="upper left",
        )

            # Create axes for checkboxes and control buttons on the left side.
        # [left, bottom, width, height] in figure coordinates.
        checkbox_ax = fig2d.add_axes([0.02, 0.30, 0.20, 0.40])
        checkbox_ax.set_title("Hide/show")
        visibility = [True] * len(scatters2d)
        # Use shortened labels for the checkbox UI to keep it compact.
        check = CheckButtons(checkbox_ax, checkbox_labels2d, visibility)

        # Buttons to show or hide all scatters at once.
        show_all_ax = fig2d.add_axes([0.02, 0.72, 0.09, 0.05])
        hide_all_ax = fig2d.add_axes([0.13, 0.72, 0.09, 0.05])
        button_show_all = Button(show_all_ax, "Show all")
        button_hide_all = Button(hide_all_ax, "Hide all")
        def toggle_scatter(label: str) -> None:
            """Callback to hide/show individual scatters based on checkbox state.

            The CheckButtons widget has already toggled the state when this
            callback is called, so we read its status and apply it to the
            corresponding scatter.
            """

            # Map from the (possibly shortened) checkbox label back to the
            # index in the checkbox_labels2d list. This assumes labels are
            # unique, which they are in this context (one per scatter entry).
            try:
                idx = checkbox_labels2d.index(label)
            except ValueError:
                return

            visible_flags = check.get_status()
            if not (0 <= idx < len(visible_flags)):
                return

            scatters2d[idx].set_visible(visible_flags[idx])
            fig2d.canvas.draw_idle()

        def show_all(event) -> None:  # type: ignore[override]
            """Show all scatters and mark all checkboxes as active."""
            visible_flags = check.get_status()
            for i, flag in enumerate(visible_flags):
                if not flag:
                    # This will toggle the checkbox to "on" and trigger
                    # toggle_scatter, which will set the corresponding scatter
                    # visible.
                    check.set_active(i)
            fig2d.canvas.draw_idle()

        def hide_all(event) -> None:  # type: ignore[override]
            """Hide all scatters and mark all checkboxes as inactive."""
            visible_flags = check.get_status()
            for i, flag in enumerate(visible_flags):
                if flag:
                    # This will toggle the checkbox to "off" and trigger
                    # toggle_scatter, which will set the corresponding scatter
                    # invisible.
                    check.set_active(i)
            fig2d.canvas.draw_idle()

        check.on_clicked(toggle_scatter)
        button_show_all.on_clicked(show_all)
        button_hide_all.on_clicked(hide_all)
 
    plt.show()
 
def main() -> None:
    results = _load_results(RESULTS_FILE)
    print(f"Loaded {len(results)} results from {RESULTS_FILE}")
    plot_all_results(results)


if __name__ == "__main__":
     main()
