#!/usr/bin/env python3
"""Radial open / enclosed porosity, pore width and pore extent of CrystoGen XYZ point clouds.

Runs the same analysis as the Clusters → Radial Profile → S₂ option in the GUI,
outside the GUI, on one or more XYZ files:

* the points are voxelised (auto voxel = 1.5× median nearest-neighbour spacing);
* empty voxels are split by flooding from the outside: *enclosed* pores are
  sealed off by solid, *open* pores are reached by the flood but lie inside the
  crystal outline (closing + hole filling); everything else is ignored;
* for each radial shell (distance from the origin (0, 0, 0)):
    - porosity = pore-voxel fraction of the shell (a plain count);
    - pore diameter = volume-weighted mean local thickness, the width of the
      largest sphere that fits in the pore and covers the voxel (what gas
      adsorption or Hg intrusion would see);
    - pore extent = S₂ correlation length ξ, the separation where the
      normalised two-point correlation first drops below 1/e. For slit-like
      pores this runs *along* the slit, so it is a length, not a width.
* over the whole crystal, the local thickness gives a pore size distribution.

Units: XYZ coordinates are in units of the lattice a-axis. With --a-length (Å)
every length (r, voxel, max lag, lags, ξ, D) is converted to nm (XYZ × a / 10),
and --voxel / --max-lag are read in nm. Without it lengths stay in XYZ units.

Outputs (in --out-dir):

    s2_radial.csv       one row per (file, shell): density, porosity, mean pore
                        diameter and ξ for both pore definitions
    s2_curves.csv       full S₂ curves per (file, pore definition, shell, lag)
    s2_psd.csv          per (file, pore diameter): volume-weighted pore size
                        distribution (local thickness) of enclosed and open pores
    s2_summary.csv      per file: core porosity (unweighted and site-weighted)
                        and peak enclosed / open porosity with its radius
    <label>_porosity_stacked.png  enclosed + open porosity stacked versus radius
    <label>_porosity_overlay.png  total, enclosed and open as overlapping areas
    <label>_pore_diameter.png     mean pore diameter (local thickness) versus radius
    <label>_pore_extent.png       pore extent (correlation length ξ) versus radius
    <label>_psd.png               pore size distribution histogram
                                  (all but the stacked plot: the lower-peaked of
                                  enclosed / open on its own right-hand axis)
    comparison.png      core and peak porosity for every file side by side

Usage:
    python scripts/analyse_s2_porosity.py thin/*.XYZ thick/*.XYZ
    python scripts/analyse_s2_porosity.py --labels Thin Thick -o s2_out \\
        thin/thin_lastframe.XYZ thick/thick_lastframe.XYZ
    python scripts/analyse_s2_porosity.py --bins 40 --voxel 2.0 --max-lag 40 file.XYZ
    python scripts/analyse_s2_porosity.py -a 10.53 file.XYZ
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Allow running directly from a source checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cgaspects.analysis.cluster_analysis import (  # noqa: E402
    _auto_voxel,
    pore_size_distribution,
    radial_profile,
    two_point_profile,
    xyz_length_scale,
)
from cgaspects.fileio.xyz_file import CrystalCloud  # noqa: E402

# Same palette as the slides: enclosed amber, open brick, pale amber for core.
ENCLOSED, OPEN, CORE = "#E39B2D", "#C8553D", "#F2CB8A"
# Neutral inks: text, secondary text and recessive axes; white surface for gaps.
INK, INK_MUTED, AXIS, SURFACE = "#222222", "#6B6B6B", "#BDBDBD", "#FFFFFF"

PLOT_STYLE = {
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlepad": 12,
    "axes.labelsize": 10,
    "axes.labelcolor": INK_MUTED,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "text.color": INK,
    "xtick.color": AXIS,
    "ytick.color": AXIS,
    "xtick.labelcolor": INK_MUTED,
    "ytick.labelcolor": INK_MUTED,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "legend.labelcolor": INK,
}


def analyse_file(path: Path, frame: int, nbins: int, voxel: float, max_lag: float,
                 length_scale: float):
    """Return (radial, curves, psd DataFrames, voxel, max_lag) for one XYZ file."""
    coords = CrystalCloud.parse_xyz_file(path)[frame].coords * length_scale
    voxel = voxel or _auto_voxel(coords)
    max_lag = max_lag or 20 * voxel

    prof = radial_profile(coords, None, None, nbins)
    edges = np.append(prof["r_lo"].to_numpy(), prof["r_hi"].iloc[-1])
    summary, curves = two_point_profile(coords, edges, voxel, max_lag)
    psd, widths = pore_size_distribution(coords, voxel, edges)
    return pd.concat([prof, summary, widths], axis=1), curves, psd, voxel, max_lag


def summarise(label: str, radial: pd.DataFrame, core_frac: float) -> dict:
    """Core porosity (inner shells) and peak porosity per pore definition."""
    r_max = radial["r_mid"].max()
    core = radial[radial["r_mid"] < core_frac * r_max]
    enc = core["s2_enclosed_porosity"]
    weights = core["n_points"].where(enc.notna(), 0)
    row = {
        "label": label,
        "r_max": r_max,
        "core_cutoff": core_frac * r_max,
        "core_shells": int(enc.notna().sum()),
        "core_enclosed_unweighted": enc.mean(),
        "core_enclosed_site_weighted": (enc.fillna(0) * weights).sum() / weights.sum(),
    }
    for name in ("enclosed", "open"):
        col = radial[f"s2_{name}_porosity"]
        i = col.idxmax()
        row[f"peak_{name}"] = col[i]
        row[f"peak_{name}_r"] = radial.loc[i, "r_mid"]
    return row


def _save(fig, ax, path: Path, y_from_zero: bool = True) -> None:
    """Shared finish for the profile figures: y from 0, legend, offset spines, save."""
    import matplotlib.pyplot as plt

    # After plotting: setting a limit earlier would freeze autoscaling at (0, 1).
    if y_from_zero:
        for a in fig.axes:
            a.set_ylim(bottom=0)
    ax.spines[["left", "bottom"]].set_position(("outward", 6))
    fig.legend(loc="upper right", ncols=3, handlelength=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path)
    plt.close(fig)


def plot_profile(radial: pd.DataFrame, rmax_frac: float, unit: str, max_lag: float,
                 stem: Path) -> None:
    """Porosity (stacked and overlaid), pore diameter and pore extent versus radius.

    One figure each: ``<stem>_porosity_stacked.png``,
    ``<stem>_porosity_overlay.png``, ``<stem>_pore_diameter.png`` (mean local
    thickness) and ``<stem>_pore_extent.png`` (S₂ correlation length ξ; a
    dashed line at ``max_lag`` marks shells with pores whose ξ exceeds it).
    """
    import matplotlib.pyplot as plt

    d = radial[radial["r_mid"] <= rmax_frac * radial["r_mid"].max()]
    r = d["r_mid"].to_numpy()
    enc = 100 * d["s2_enclosed_porosity"].fillna(0).to_numpy()
    opn = 100 * d["s2_open_porosity"].fillna(0).to_numpy()
    xlabel = f"distance from origin r ({unit})"

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(6, 4.6))
        ax.stackplot(r, enc, opn, labels=["Enclosed", "Open"], colors=[ENCLOSED, OPEN],
                     edgecolor=SURFACE, linewidth=1.5)
        ax.plot(r, enc + opn, color=INK, lw=1, alpha=0.6)
        ax.set(xlabel=xlabel, ylabel="porosity (%)", xlim=(r.min(), r.max()))
        _save(fig, ax, stem.with_name(f"{stem.name}_porosity_stacked.png"))

        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        ax_enc, ax_opn = _split_axes(ax, enc, opn, "porosity (%)", with_total=True)
        _area(ax, r, enc + opn, INK_MUTED, "Total", alpha=0.18)
        _area(ax_enc, r, enc, ENCLOSED, "Enclosed")
        _area(ax_opn, r, opn, OPEN, "Open")
        ax.set(xlabel=xlabel, xlim=(r.min(), r.max()))
        _save(fig, ax, stem.with_name(f"{stem.name}_porosity_overlay.png"))

        for suffix, col, ylabel in (
            ("pore_diameter", "{}_pore_diameter", f"pore diameter D ({unit})"),
            ("pore_extent", "s2_{}_corr_length", f"pore extent ξ ({unit})"),
        ):
            fig, ax = plt.subplots(figsize=(6.4, 4.6))
            y_enc = d[col.format("enclosed")].to_numpy()
            y_opn = d[col.format("open")].to_numpy()
            ax_enc, ax_opn = _split_axes(ax, y_enc, y_opn, ylabel, with_total=False)
            for target, y, colour, name in ((ax_enc, y_enc, ENCLOSED, "Enclosed"),
                                            (ax_opn, y_opn, OPEN, "Open")):
                target.plot(r, y, color=colour, lw=2, label=name, solid_capstyle="round")
                if suffix == "pore_extent":
                    # Pores present but S₂ never fell below 1/e within max lag:
                    # ξ > max lag, unlike the gaps where the shell has no pores.
                    beyond = (d[f"s2_{name.lower()}_porosity"] > 0).to_numpy() & np.isnan(y)
                    if beyond.any():
                        target.hlines(np.full(beyond.sum(), max_lag),
                                      d["r_lo"][beyond], d["r_hi"][beyond],
                                      colors=colour, linestyles=(0, (4, 3)), lw=1.5,
                                      label=f"{name}, ξ > max lag ({max_lag:.3g} {unit})")
            ax.set(xlabel=xlabel, xlim=(r.min(), r.max()))
            # Widths sit in a narrow band well above 0; let that axis autoscale.
            _save(fig, ax, stem.with_name(f"{stem.name}_{suffix}.png"),
                  y_from_zero=suffix != "pore_diameter")


def _split_axes(ax, enc, opn, ylabel: str, with_total: bool):
    """Give enclosed and open their own y-axes; return ``(ax_enc, ax_opn)``.

    Whichever peaks higher stays on ``ax`` (left, shared with the total when
    ``with_total``); the other moves to a twin right axis drawn in its colour,
    so it is not flattened by the larger one.
    """
    twin = ax.twinx()
    enc_small = np.nanmax(enc) < np.nanmax(opn)
    small, big = ("enclosed", "open") if enc_small else ("open", "enclosed")
    colour = ENCLOSED if enc_small else OPEN

    ax.set_ylabel(f"{ylabel}, {big}{' and total' if with_total else ''}")
    twin.set_ylabel(f"{ylabel}, {small}", color=colour)
    twin.spines[["left", "bottom", "top"]].set_visible(False)
    twin.spines["right"].set_visible(True)
    twin.spines["right"].set_position(("outward", 6))
    twin.spines["right"].set_color(colour)
    twin.tick_params(axis="y", colors=colour, labelcolor=colour)
    return (twin, ax) if enc_small else (ax, twin)


def _area(ax, x, y, colour: str, name: str, alpha: float = 0.4) -> None:
    """Translucent filled area with a solid outline (NMR-deconvolution style)."""
    ax.fill_between(x, y, color=colour, alpha=alpha, lw=0, label=name)
    ax.plot(x, y, color=colour, lw=1.5)


def _bars(ax, edges, y, colour: str, name: str, alpha: float = 0.4) -> None:
    """Translucent histogram with a solid outline."""
    ax.stairs(y, edges, fill=True, color=colour, alpha=alpha, lw=0, label=name)
    ax.stairs(y, edges, color=colour, lw=1.5)


def plot_psd(psd: pd.DataFrame, unit: str, path: Path) -> None:
    """Volume-weighted pore size distribution histogram: total, enclosed and open.

    Bars are dφ/dD, so each histogram's area is that pore type's whole-crystal
    porosity. Enclosed and open get their own y-axes (see ``_split_axes``).
    """
    import matplotlib.pyplot as plt

    edges = np.r_[psd["bin_lo"].to_numpy(), psd["bin_hi"].iloc[-1]]
    width = (psd["bin_hi"] - psd["bin_lo"]).to_numpy()
    enc = 100 * psd["enclosed_fraction"].to_numpy() / width
    opn = 100 * psd["open_fraction"].to_numpy() / width

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        ax_enc, ax_opn = _split_axes(ax, enc, opn, f"dφ/dD (% per {unit})", with_total=True)
        _bars(ax, edges, enc + opn, INK_MUTED, "Total", alpha=0.18)
        _bars(ax_enc, edges, enc, ENCLOSED, "Enclosed")
        _bars(ax_opn, edges, opn, OPEN, "Open")
        ax.set(xlabel=f"pore diameter D ({unit})", xlim=(0, edges[-1]))
        _save(fig, ax, path)


def plot_comparison(summary: pd.DataFrame, core_frac: float, path: Path) -> None:
    """Core (site-weighted) and peak porosity for each file (slide-7 style)."""
    import matplotlib.pyplot as plt

    series = [
        ("core_enclosed_site_weighted", f"Enclosed, core (r < {core_frac:g} r_max)", CORE),
        ("peak_enclosed", "Enclosed, peak shell", ENCLOSED),
        ("peak_open", "Open, peak shell", OPEN),
    ]
    x = np.arange(len(summary))
    width = 0.26
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(max(6, 1.9 * len(summary) + 2), 4.6))
        for k, (col, name, colour) in enumerate(series):
            vals = 100 * summary[col].to_numpy()
            bars = ax.bar(x + (k - 1) * width, vals, width, label=name, color=colour,
                          edgecolor=SURFACE, linewidth=1.5)
            ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=3, color=INK_MUTED)
        ax.set_xticks(x, summary["label"])
        ax.tick_params(axis="x", length=0, labelsize=10, labelcolor=INK)
        ax.set_title("Core and peak porosity")
        ax.set_ylabel("porosity (%)")
        ax.margins(y=0.12)
        ax.spines["left"].set_position(("outward", 6))
        fig.legend(loc="lower center", ncols=len(series), handlelength=1.2)
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        fig.savefig(path)
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", type=Path, help="CrystoGen XYZ file(s)")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Display name per file (default: file stem)")
    parser.add_argument("-o", "--out-dir", type=Path, default=Path("s2_output"), dest="out_dir",
                        help="Output folder (default: ./s2_output)")
    parser.add_argument("--frame", type=int, default=-1,
                        help="Frame index in each XYZ file (default: -1, last frame)")
    parser.add_argument("--bins", type=int, default=30,
                        help="Number of radial shells (default: 30)")
    parser.add_argument("-a", "--a-length", type=float, default=0.0, dest="a_length",
                        help="Lattice a-axis length in Å; converts all lengths to nm "
                             "(default: not set, lengths stay in XYZ units)")
    parser.add_argument("--voxel", type=float, default=0.0,
                        help="Voxel edge in nm, or XYZ units without --a-length "
                             "(default: auto, 1.5× NN spacing)")
    parser.add_argument("--max-lag", type=float, default=0.0, dest="max_lag",
                        help="Largest S₂ separation in nm, or XYZ units without --a-length "
                             "(default: auto, 20 voxels)")
    parser.add_argument("--core-frac", type=float, default=0.4, dest="core_frac",
                        help="Core = shells with r_mid < core_frac × r_max (default: 0.4)")
    parser.add_argument("--plot-rmax-frac", type=float, default=1.0, dest="plot_rmax_frac",
                        help="Only plot shells with r_mid ≤ this × r_max, e.g. 0.87 to drop "
                             "surface-dominated outer shells (default: 1.0, all)")
    parser.add_argument("--no-plots", action="store_true", dest="no_plots",
                        help="Write the CSVs only")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Resolution of the PNG plots (default: 300)")
    args = parser.parse_args()
    PLOT_STYLE["savefig.dpi"] = args.dpi

    if args.labels is not None and len(args.labels) != len(args.files):
        parser.error(f"--labels needs one name per file ({len(args.files)} files given)")
    labels = args.labels or [p.stem for p in args.files]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    length_scale, unit = xyz_length_scale(args.a_length)
    if unit != "nm":
        print("warning: no --a-length given, lengths are in XYZ units, not nm", file=sys.stderr)

    radial_frames, curve_frames, psd_frames, summary_rows = [], [], [], []
    bar = tqdm(list(zip(args.files, labels)), desc="files", unit="file")
    for path, label in bar:
        if not path.exists():
            bar.write(f"skip (not found): {path}")
            continue
        bar.set_postfix_str(path.name)
        try:
            radial, curves, psd, voxel, max_lag = analyse_file(
                path, args.frame, args.bins, args.voxel, args.max_lag, length_scale)
        except Exception as exc:  # noqa: BLE001 - keep going through the batch
            bar.write(f"skip ({exc}): {path}")
            continue

        radial.insert(0, "label", label)
        radial.insert(1, "file", str(path))
        radial.insert(2, "length_unit", unit)
        curves.insert(0, "label", label)
        curves.insert(1, "length_unit", unit)
        radial_frames.append(radial)
        curve_frames.append(curves)
        psd_frames.append(psd.assign(label=label, length_unit=unit))
        row = summarise(label, radial, args.core_frac)
        row.update(file=str(path), length_unit=unit, a_length_angstrom=args.a_length or None,
                   voxel=voxel, max_lag=max_lag)
        summary_rows.append(row)
        bar.write(f"  {label}: voxel {voxel:.3g} {unit}, core enclosed "
                  f"{row['core_enclosed_site_weighted']:.2%} (r < {row['core_cutoff']:.3g}), "
                  f"peak enclosed "
                  f"{row['peak_enclosed']:.2%} @ r={row['peak_enclosed_r']:.3g}, peak open "
                  f"{row['peak_open']:.2%} @ r={row['peak_open_r']:.3g} {unit}")

        if not args.no_plots:
            import matplotlib

            matplotlib.use("Agg")
            stem = args.out_dir / label.replace(" ", "_")
            plot_profile(radial, args.plot_rmax_frac, unit, max_lag, stem)
            plot_psd(psd, unit, stem.with_name(f"{stem.name}_psd.png"))
    bar.close()

    if not summary_rows:
        return 1

    summary = pd.DataFrame(summary_rows)
    pd.concat(radial_frames, ignore_index=True).to_csv(args.out_dir / "s2_radial.csv", index=False)
    pd.concat(curve_frames, ignore_index=True).to_csv(args.out_dir / "s2_curves.csv", index=False)
    psd_cols = ["label", "length_unit", "pore_diameter", "bin_lo", "bin_hi",
                "enclosed_fraction", "open_fraction"]
    pd.concat(psd_frames, ignore_index=True)[psd_cols].to_csv(args.out_dir / "s2_psd.csv",
                                                              index=False)
    summary.to_csv(args.out_dir / "s2_summary.csv", index=False)
    if not args.no_plots:
        plot_comparison(summary, args.core_frac, args.out_dir / "comparison.png")

    print(f"\nWrote s2_radial.csv, s2_curves.csv, s2_psd.csv, s2_summary.csv"
          f"{' and plots' if not args.no_plots else ''} to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
