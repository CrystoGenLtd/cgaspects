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
    # (a, b, c) bool mask flagging the edge cells of every strip block. The full
    # ``data`` grid always holds *all* sites; ``edge_mask`` lets callers toggle the
    # interior (middle) cells on/off without re-reading the file.
    edge_mask: np.ndarray = None
    # When False (default) only edge cells are shown; when True the middle cells
    # are included too. Flip this and re-read ``visible_data`` — no file I/O needed.
    show_middle: bool = False

    @property
    def visible_data(self) -> np.ndarray:
        """Grid honouring ``show_middle``: full data, or edges only when off."""
        if self.show_middle or self.edge_mask is None:
            return self.data
        return np.where(self.edge_mask[..., None], self.data, 0)

    @property
    def grid_shape(self) -> tuple[int, int, int]:
        return self.data.shape[:3]

    @property
    def n_filled(self) -> int:
        return int(np.count_nonzero(self.visible_data))

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

        mask = self.visible_data.any(axis=-1)  # (a, b, c) — any tile occupied
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

        # Tokenise the whole file in one C-level pass. The checkpoint is a flat
        # stream of whitespace-separated integers (after the header words), so
        # splitting once is far cheaper than iterating and int()-parsing lines.
        tokens = checkpoint_path.read_text(errors="replace").split()

        # Locate the Grid dimensions and the start of the Strip data. Newlines
        # collapse under split(), and the marker lines can carry trailing words
        # ("Grid dimensions:", "Strip data:"), so skip past any non-integer
        # tokens to find the actual numbers rather than assuming fixed offsets.
        def _is_int(s: str) -> bool:
            try:
                int(s)
                return True
            except ValueError:
                return False

        a = b = c = None
        strip_idx = None
        i = 0
        n_tokens = len(tokens)
        while i < n_tokens:
            tok = tokens[i]
            if a is None and tok.startswith("Grid"):
                dims = []
                j = i + 1
                while j < n_tokens and len(dims) < 3:
                    if tokens[j].startswith("Strip"):
                        break
                    if _is_int(tokens[j]):
                        dims.append(int(tokens[j]))
                    j += 1
                if len(dims) < 3:
                    raise ValueError(
                        "Failed to read Grid dimensions from checkpoint file"
                    )
                a, b, c = dims
                i = j
                continue
            if tok.startswith("Strip"):
                j = i + 1
                while j < n_tokens and not _is_int(tokens[j]):
                    j += 1
                strip_idx = j
                break
            i += 1

        if a is None:
            raise ValueError("Grid section was not found in checkpoint file")

        if strip_idx is None:
            raise ValueError("Strip section was not found in checkpoint file")

        grid = np.zeros((a, b, c, n_tiles), dtype=np.int32)
        edge_mask = np.zeros((a, b, c), dtype=bool)

        # Everything past the Strip marker is integers; parse them in bulk (C)
        # and hand back native Python ints so the structural walk below is pure
        # list indexing with no per-site int()/strip()/iterator overhead.
        try:
            nums = np.array(tokens[strip_idx:], dtype=np.int64).tolist()
        except ValueError as exc:
            raise ValueError("Non-integer value found in strip data") from exc
        del tokens

        total = len(nums)
        pos = 0
        try:
            while pos < total:
                y, z, num_blocks = nums[pos], nums[pos + 1], nums[pos + 2]
                pos += 3

                for _ in range(num_blocks):
                    start = nums[pos]
                    end = nums[pos + 1] + 1
                    pos += 2

                    for cell in range(start, end):
                        if cell == start or cell == end - 1:
                            edge_mask[cell, y, z] = True

                        grid[cell, y, z, :] = nums[pos:pos + n_tiles]
                        pos += n_tiles

                if progress_callback is not None and total:
                    progress_callback(pos, total)
        except (IndexError, ValueError) as exc:
            raise ValueError("Unexpected end of file while reading strip data") from exc

        elapsed = time.perf_counter() - t_start
        LOG.info("Loaded %s in %.3f s", checkpoint_path.name, elapsed)

        return cls(checkpoint_path, n_tiles, crysallography, grid, edge_mask)


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
