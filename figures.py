"""
figures.py — CLI entry point for producing thesis figures.

Usage
-----
    python figures.py                          # all figures  → figures/
    python figures.py fdr                      # single figure → figures/fdr.pdf
    python figures.py fdr fdr_riem             # two figures
    python figures.py --output out/ unimodal   # custom output dir
    python figures.py --pgf                    # save .pgf instead of .pdf
    python figures.py --list                   # print available names and exit

Available figure names:  fdr, fdr_riem, mollweide, unimodal
"""

import argparse
import os
import sys

FIGURE_NAMES = ["fdr", "fdr_riem", "mollweide", "unimodal"]


def _build_registry():
    """
    Import all script modules and return the figure registry.

    Imports are deferred here so that apply_thesis_style() (which locks the
    matplotlib backend) is always called BEFORE any script module is imported.
    Scripts no longer import matplotlib.pyplot at module level, so this order
    is safe.
    """
    from scripts.fdr       import simulate_fdr,       plot_fdr
    from scripts.fdr_riem  import simulate_fdr_riem,  plot_fdr_riem
    from scripts.mollweide import (simulate_mollweide,
                                   plot_mollweide_density,
                                   plot_mollweide_speed)
    from scripts.unimodal_v3 import (simulate_unimodal_v3,
                                     plot_unimodal_v3,
                                     plot_unimodal_v3_ratio)
    return {
        "fdr": (
            simulate_fdr,
            [(plot_fdr, "fdr")],
        ),
        "fdr_riem": (
            simulate_fdr_riem,
            [(plot_fdr_riem, "fdr_riem")],
        ),
        "mollweide": (
            simulate_mollweide,
            [(plot_mollweide_density, "mollweide_density"),
             (plot_mollweide_speed,   "mollweide_speed")],
        ),
        "unimodal": (
            simulate_unimodal_v3,
            [(plot_unimodal_v3,       "unimodal"),
             (plot_unimodal_v3_ratio, "unimodal_ratio")],
        ),
    }


def _outpath(output_dir, stem, ext):
    return os.path.join(output_dir, f"{stem}.{ext}")


def main():
    parser = argparse.ArgumentParser(
        prog="figures.py",
        description="Produce thesis figures into an output directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Available figure names: " + ", ".join(FIGURE_NAMES) + "\n\n"
            "Examples:\n"
            "  python figures.py                          # all figures\n"
            "  python figures.py fdr                      # single figure\n"
            "  python figures.py fdr fdr_riem             # two figures\n"
            "  python figures.py --output out/ unimodal   # custom output dir\n"
            "  python figures.py --pgf                    # PGF output\n"
        ),
    )
    parser.add_argument(
        "figure",
        nargs="*",
        metavar="FIGURE",
        help=f"Figure name(s) to produce. Choices: {', '.join(FIGURE_NAMES)}. "
             "Omit to produce all.",
    )
    parser.add_argument(
        "--output", "-o",
        default="output",
        metavar="DIR",
        help="Output directory (default: output/)",
    )
    parser.add_argument(
        "--pgf",
        action="store_true",
        help="Save .pgf files instead of .pdf (for direct LaTeX \\input{})",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print available figure names and exit",
    )
    args = parser.parse_args()

    if args.list:
        print("Available figures:")
        for name in FIGURE_NAMES:
            print(f"  {name}")
        sys.exit(0)

    # Validate names before any expensive import or simulation
    requested = args.figure if args.figure else FIGURE_NAMES
    unknown = [n for n in requested if n not in FIGURE_NAMES]
    if unknown:
        parser.error(
            f"Unknown figure(s): {', '.join(unknown)}. "
            "Use --list to see valid names."
        )

    # --- Apply style BEFORE importing any script module ---
    # (Scripts no longer import pyplot at module level, so this ordering is safe.)
    from viz_style import apply_thesis_style
    backend = "pgf" if args.pgf else "pdf"
    colors = apply_thesis_style(backend=backend)

    # Now safe to import script modules (pyplot already initialised with correct backend)
    registry = _build_registry()

    os.makedirs(args.output, exist_ok=True)
    ext = "pgf" if args.pgf else "pdf"

    for name in requested:
        simulate_fn, plot_entries = registry[name]
        print(f"[{name}] Running simulation...", flush=True)
        result = simulate_fn()
        for plot_fn, stem in plot_entries:
            outpath = _outpath(args.output, stem, ext)
            print(f"[{name}] Saving {outpath}...", flush=True)
            plot_fn(result, outpath, colors)
        print(f"[{name}] Done.", flush=True)

    print(f"\nAll done. Files written to: {os.path.abspath(args.output)}/")


if __name__ == "__main__":
    main()
