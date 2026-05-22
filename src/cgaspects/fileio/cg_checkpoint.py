import logging
import argparse
import time
from pathlib import Path
from dataclasses import dataclass
from collections.abc import Callable
from cgaspects.gui.utils.crystallography import Crystallography

import numpy as np

LOG = logging.getLogger("CGA:checkpoint")


@dataclass
class Checkpoint:
    filepath: Path
    n_tiles: int
    crystallography: Crystallography
    data: np.ndarray

    @property
    def grid_shape(self) -> tuple[int, int, int]:
        return self.data.shape[:3]

    @property
    def n_filled(self) -> int:
        return int(self.data.sum())

    @property
    def n_empty(self) -> int:
        return self.data.size - self.n_filled

    @property
    def fill_fraction(self) -> float:
        return self.n_filled / self.data.size if self.data.size else 0.0

    def __repr__(self) -> str:
        a, b, c = self.grid_shape
        return (
            f"Checkpoint("
            f"file={self.filepath.name!r}, "
            f"grid=({a}×{b}×{c}), "
            f"tiles={self.n_tiles}, "
            f"filled={self.n_filled:,}/{self.data.size:,} "
            f"({self.fill_fraction:.1%})"
            f")"
        )

    def __str__(self) -> str:
        a, b, c = self.grid_shape
        cryst_str = repr(self.crystallography) if self.crystallography else "None"
        return (
            f"Checkpoint\n"
            f"  File          : {self.filepath}\n"
            f"  Grid          : {a} × {b} × {c}\n"
            f"  Tiles         : {self.n_tiles}\n"
            f"  Filled sites  : {self.n_filled:,}\n"
            f"  Empty sites   : {self.n_empty:,}\n"
            f"  Fill fraction : {self.fill_fraction:.2%}\n"
            f"  Crystallography: {cryst_str}\n"
        )

    def to_cartesian(self) -> np.ndarray:
        """Return Cartesian coordinates (Å) of all occupied grid points.

        Each grid index (i, j, k) represents one unit cell, so the integer
        indices are fractional coordinates and map directly via frac_to_cart.

        Returns:
            np.ndarray of shape (N, 3).
        """
        if self.crystallography is None:
            raise ValueError("Cannot convert to Cartesian: Crystallography is not set")

        mask = self.data.any(axis=-1)          # (a, b, c) — any tile occupied
        indices = np.argwhere(mask).astype(float)  # (N, 3)
        return self.crystallography.frac_to_cart(indices)

    @classmethod
    def from_file(
        cls,
        checkpoint_file: str | Path,
        n_tiles: int,
        crysallography: Crystallography = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ):
        t_start = time.perf_counter()
        checkpoint_path = Path(checkpoint_file)
        file_size = checkpoint_path.stat().st_size

        a = b = c = None
        found_strip = False

        with checkpoint_path.open(errors="replace") as f:
            line_iter = iter(f)

            # Read header / locate Strip section
            for line in line_iter:
                line = line.strip()

                if line.startswith("Grid"):
                    try:
                        a, b, c = map(int, next(line_iter).split())
                    except (StopIteration, ValueError) as exc:
                        raise ValueError(
                            "Failed to read Grid dimensions from checkpoint file"
                        ) from exc

                    grid = np.zeros((a, b, c, n_tiles), dtype=bool)

                elif line.startswith("Strip"):
                    found_strip = True
                    break

            if a is None:
                raise ValueError("Grid section was not found in checkpoint file")

            if not found_strip:
                raise ValueError("Strip section was not found in checkpoint file")

            # Parse strip data
            for line in line_iter:
                line = line.strip()

                if not line:
                    continue

                try:
                    y, z = map(int, line.split())
                    num_blocks = int(next(line_iter).strip())

                    for _ in range(num_blocks):
                        start = int(next(line_iter).strip())
                        end = int(next(line_iter).strip()) + 1

                        for cell in range(start, end):
                            for tile in range(n_tiles):
                                site = int(next(line_iter).strip())

                                if site != 0:
                                    grid[cell, y, z, tile] = 1

                    if progress_callback is not None and file_size:
                        progress_callback(f.tell(), file_size)

                    continue

                except ValueError as exc:
                    raise ValueError(f"Invalid strip header line: {line!r}") from exc
                except StopIteration as exc:
                    raise ValueError("Unexpected end of file while reading strip data") from exc

        elapsed = time.perf_counter() - t_start
        LOG.info("Loaded %s in %.3f s", checkpoint_path.name, elapsed)

        return cls(checkpoint_path, n_tiles, crysallography, grid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Checkpoint Reader")
    parser.add_argument("-i", "--input", type=str, required=True, help="Checkpoint file")
    parser.add_argument(
        "-n",
        "--n-tiles",
        type=int,
        required=True,
        help="Number of tiles in structure",
        dest="n_tiles",
    )

    args = parser.parse_args()

    checkpoint_path = Path(args.input)

    print(Checkpoint.from_file(checkpoint_path, args.n_tiles))
