#!/usr/bin/env python3
"""Estimate internal porosity of one or more checkpoint files.

Checkpoints store the crystal as strips along the x-axis: for every (y, z)
line there is a set of *blocks* (contiguous runs of filled cells). Whenever a
line has more than one block, the empty cells sitting *between* the outermost
blocks are enclosed by material on both sides along x -- these are internal
pores. Empty space beyond the first/last filled cell is just the external
surface and is ignored.

Porosity is reported directionally along x:

    internal_void = sum over lines of (empty cells between first & last filled)
    extent        = sum over lines of (last - first + 1)   # material x-span
    porosity      = internal_void / extent

A radial density profile is also computed: cells are binned by their distance
from the centroid of the occupied cells, and for each shell we report the
occupancy fraction (occupied cells / total grid cells in the shell). A solid
core surrounded by surface falloff shows as a high plateau dropping to zero;
internal pores show up as dips in the profile.

Usage:
    python scripts/analyse_porosity.py -n 4 file1.txt file2.txt ...
    python scripts/analyse_porosity.py -n 4 --csv out.csv checkpoints/*.txt
    python scripts/analyse_porosity.py -n 4 --radial-csv radial.csv \\
        --plot radial.png checkpoints/*.txt
    python scripts/analyse_porosity.py -n 4 --plot radial.png \\
        --cartesian --cell structure.txt checkpoints/*.txt
"""

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Allow running directly from a source checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cgaspects.fileio.cg_checkpoint import Checkpoint  # noqa: E402
from cgaspects.fileio.structure import Structure  # noqa: E402
from cgaspects.gui.utils.crystallography import Cell, Crystallography  # noqa: E402


@dataclass
class PorosityResult:
    filepath: Path
    grid_shape: tuple[int, int, int]
    filled: int              # occupied cells
    internal_void: int       # empty cells enclosed along x
    extent: int              # total material x-span (filled + internal_void)
    n_porous_lines: int      # (y, z) lines with >= 1 internal gap
    n_material_lines: int     # (y, z) lines that contain any material

    @property
    def porosity(self) -> float:
        return self.internal_void / self.extent if self.extent else 0.0


@dataclass
class RadialProfile:
    filepath: Path
    r_mid: np.ndarray        # (nbins,) shell centre distance
    occupied: np.ndarray     # (nbins,) occupied cells per shell
    total: np.ndarray        # (nbins,) total grid cells per shell
    units: str = "grid"      # "grid" (cell indices) or "Å" (Cartesian)

    @property
    def density(self) -> np.ndarray:
        """Occupancy fraction per shell (0 where a shell holds no cells)."""
        return np.divide(
            self.occupied, self.total,
            out=np.zeros(self.occupied.shape, dtype=float),
            where=self.total > 0,
        )


def analyse_porosity(chk: Checkpoint) -> tuple[int, int, int, int, int]:
    """Compute internal void / extent / filled counts along the x-axis.

    Returns (filled, internal_void, extent, n_porous_lines, n_material_lines).
    """
    # occ[i, y, z] True where any tile of cell (i, y, z) is occupied.
    occ = chk.data.any(axis=-1)          # (a, b, c)
    a = occ.shape[0]

    has_material = occ.any(axis=0)        # (b, c) lines with any filled cell
    n_occ = occ.sum(axis=0)              # (b, c) filled cells per line

    # First / last filled index along x for each line (argmax finds first True).
    first = np.argmax(occ, axis=0)                    # (b, c)
    last = a - 1 - np.argmax(occ[::-1], axis=0)       # (b, c)
    extent = np.where(has_material, last - first + 1, 0)

    internal = extent - n_occ            # empty cells between first & last

    filled = int(n_occ[has_material].sum())
    internal_void = int(internal[has_material].sum())
    total_extent = int(extent[has_material].sum())
    n_porous_lines = int(np.count_nonzero(internal[has_material] > 0))
    n_material_lines = int(np.count_nonzero(has_material))

    return filled, internal_void, total_extent, n_porous_lines, n_material_lines


def radial_density(
    chk: Checkpoint,
    nbins: int,
    cryst: Crystallography | None = None,
    rmax_from_filled: bool = False,
) -> RadialProfile:
    """Occupancy fraction as a function of distance from the occupied centroid.

    Every grid cell is binned by its distance from the centroid of the occupied
    cells; each shell's density is (occupied / total) cells in it. Without
    ``cryst`` distances are in grid-index units (one unit == one cell along an
    axis); with it the grid indices (fractional coords) are mapped to Cartesian
    space so distances are in Å.

    The bins span 0 to the furthest grid cell by default; with
    ``rmax_from_filled`` they instead stop at the furthest *occupied* cell, so
    the empty corners of the box beyond the material are dropped.
    """
    occ = chk.data.any(axis=-1)          # (a, b, c)
    # (N, 3) fractional coords of every cell; each grid index is a unit-cell step.
    coords = np.indices(occ.shape, dtype=float).reshape(3, -1).T
    if cryst is not None:
        coords = cryst.frac_to_cart(coords)
        units = "Å"
    else:
        units = "grid"

    occ_flat = occ.ravel()
    if occ_flat.any():
        centroid = coords[occ_flat].mean(axis=0)
    else:
        centroid = coords.mean(axis=0)

    dist = np.linalg.norm(coords - centroid, axis=1)   # (N,)

    if rmax_from_filled and occ_flat.any():
        r_max = float(dist[occ_flat].max())
    else:
        r_max = float(dist.max()) if dist.size else 0.0
    edges = np.linspace(0.0, r_max if r_max > 0 else 1.0, nbins + 1)

    total, _ = np.histogram(dist, bins=edges)
    occupied, _ = np.histogram(dist[occ_flat], bins=edges)
    r_mid = 0.5 * (edges[:-1] + edges[1:])

    return RadialProfile(
        filepath=chk.filepath,
        r_mid=r_mid,
        occupied=occupied.astype(np.int64),
        total=total.astype(np.int64),
        units=units,
    )


def process_file(
    path: Path,
    n_tiles: int,
    nbins: int,
    cryst: Crystallography | None = None,
    rmax_from_filled: bool = False,
) -> tuple[PorosityResult, RadialProfile]:
    # Drive a byte-level progress bar off the checkpoint reader's callback.
    with tqdm(total=path.stat().st_size, unit="B", unit_scale=True,
              desc=f"  reading {path.name}", leave=False) as bar:
        def on_progress(current: int, total: int) -> None:
            bar.update(current - bar.n)

        chk = Checkpoint.from_file(path, n_tiles, progress_callback=on_progress)

    filled, void, extent, porous_lines, mat_lines = analyse_porosity(chk)
    result = PorosityResult(
        filepath=path,
        grid_shape=chk.grid_shape,
        filled=filled,
        internal_void=void,
        extent=extent,
        n_porous_lines=porous_lines,
        n_material_lines=mat_lines,
    )
    return result, radial_density(chk, nbins, cryst, rmax_from_filled)


def write_radial_csv(profiles: list[RadialProfile], csv_path: Path) -> None:
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "bin", "r_mid", "occupied", "total", "density"])
        for prof in profiles:
            density = prof.density
            for i, r in enumerate(prof.r_mid):
                w.writerow([
                    prof.filepath, i, f"{r:.6f}",
                    int(prof.occupied[i]), int(prof.total[i]),
                    f"{density[i]:.6f}",
                ])


def plot_radial(profiles: list[RadialProfile], plot_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for prof in profiles:
        ax.plot(prof.r_mid, prof.density, marker=".", label=prof.filepath.name)

    units = profiles[0].units if profiles else "grid"
    unit_label = "Å" if units == "Å" else "grid units"
    ax.set_xlabel(f"Distance from centroid ({unit_label})")
    ax.set_ylabel("Occupancy fraction")
    ax.set_title("Radial density profile")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    if len(profiles) > 1:
        ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def resolve_crystallography(cell_arg: list[str]) -> Crystallography:
    """Build a Crystallography from ``--cell``.

    Accepts either six lattice values ``a b c alpha beta gamma`` or a single
    path to a CrystalGrower structure file, from which the cell is parsed with
    the same reader the GUI uses.
    """
    if len(cell_arg) == 1 and Path(cell_arg[0]).exists():
        structure = Structure.from_file(cell_arg[0])
        if structure.cryst is None:
            raise ValueError(f"No lattice parameters found in {cell_arg[0]}")
        return structure.cryst

    if len(cell_arg) == 6:
        try:
            a, b, c, alpha, beta, gamma = map(float, cell_arg)
        except ValueError as exc:
            raise ValueError(f"--cell values must be numeric: {cell_arg}") from exc
        return Crystallography(Cell(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma))

    raise ValueError(
        "--cell expects either a structure-file path or six values "
        "'a b c alpha beta gamma'"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", type=Path, help="Checkpoint file(s)")
    parser.add_argument("-n", "--n-tiles", type=int, required=True, dest="n_tiles",
                        help="Number of tiles in the structure")
    parser.add_argument("--csv", type=Path, default=None,
                        help="Optional path to write per-file results as CSV")
    parser.add_argument("--radial-csv", type=Path, default=None, dest="radial_csv",
                        help="Optional path to write the radial density profile as CSV")
    parser.add_argument("--plot", type=Path, default=None,
                        help="Optional path to save a radial density plot (e.g. .png)")
    parser.add_argument("--nbins", type=int, default=40,
                        help="Number of radial shells for the density profile (default: 40)")
    parser.add_argument("--rmax-filled", action="store_true", dest="rmax_filled",
                        help="End the radial bins at the furthest filled site "
                             "instead of the furthest grid cell")
    parser.add_argument("--cartesian", action="store_true",
                        help="Measure radial distances in Cartesian Å (requires --cell)")
    parser.add_argument("--cell", nargs="+", default=None, metavar="VALUE",
                        help="Lattice for --cartesian: six values "
                             "'a b c alpha beta gamma' or a structure-file path")
    args = parser.parse_args()

    cryst = None
    if args.cartesian:
        if not args.cell:
            parser.error("--cartesian requires --cell")
        try:
            cryst = resolve_crystallography(args.cell)
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
    elif args.cell:
        parser.error("--cell only applies with --cartesian")

    mode = "Cartesian (Å)" if cryst is not None else "grid units"
    print(f"Analysing {len(args.files)} file(s) — radial density in {mode}, "
          f"{args.nbins} bins")

    results: list[PorosityResult] = []
    profiles: list[RadialProfile] = []
    file_bar = tqdm(args.files, desc="files", unit="file")
    for path in file_bar:
        if not path.exists():
            file_bar.write(f"skip (not found): {path}")
            continue
        try:
            result, profile = process_file(path, args.n_tiles, args.nbins, cryst,
                                           args.rmax_filled)
            results.append(result)
            profiles.append(profile)
            file_bar.write(f"  {path.name}: filled={result.filled:,} "
                           f"porosity={result.porosity:.2%}")
        except Exception as exc:  # noqa: BLE001 - keep going through the batch
            file_bar.write(f"skip ({exc}): {path}")
    file_bar.close()

    if not results:
        return 1

    header = f"{'file':<40} {'grid':>16} {'filled':>10} {'void':>10} {'porosity':>9}"
    print(header)
    print("-" * len(header))
    for r in results:
        grid = "×".join(map(str, r.grid_shape))
        print(f"{r.filepath.name:<40} {grid:>16} {r.filled:>10,} "
              f"{r.internal_void:>10,} {r.porosity:>8.2%}")

    if args.csv:
        with args.csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file", "a", "b", "c", "filled", "internal_void",
                        "extent", "porosity", "porous_lines", "material_lines"])
            for r in results:
                a, b, c = r.grid_shape
                w.writerow([r.filepath, a, b, c, r.filled, r.internal_void,
                            r.extent, f"{r.porosity:.6f}",
                            r.n_porous_lines, r.n_material_lines])
        print(f"\nWrote {len(results)} rows to {args.csv}")

    if args.radial_csv:
        write_radial_csv(profiles, args.radial_csv)
        print(f"Wrote radial profile ({args.nbins} bins × {len(profiles)} files) "
              f"to {args.radial_csv}")

    if args.plot:
        print("Rendering radial density plot ...")
        plot_radial(profiles, args.plot)
        print(f"Wrote radial density plot to {args.plot}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
