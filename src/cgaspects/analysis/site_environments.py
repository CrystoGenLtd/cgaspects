"""Enumerate the local environments of each site type in a CrystoGen checkpoint.

A site type fixes which neighbours of a site are occupied only up to the
simulation's grouping rules, so the instances of one type can sit in many
different surroundings. This module groups the instances of a site type by their
occupied first and second neighbour shells, ranks those environments by how often
they occur, and writes any of them in CrystalGrower's ``*_docking.XYZ`` layout.

The shells follow ``docking.f90``: the first shell is every occupied net neighbour
of the target, and the second shell is every occupied net neighbour of a first-shell
molecule. Unlike ``docking.f90`` the target is never listed as its own neighbour.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from ..fileio.cg_checkpoint import Checkpoint
from ..fileio.structure import Structure
from ..fileio.xyz_file import DockingData

# Sites classified per batch; bounds memory for the very populous bulk types.
_BATCH_SIZE = 20_000

# A site relative to the target: (cell dx, cell dy, cell dz, 1-based tile number).
RelativeSite = tuple[int, int, int, int]


@dataclass(frozen=True)
class SiteTypeSummary:
    """One site type present in the checkpoint."""

    site: int
    count: int
    grown: bool
    tiles: tuple[int, ...]


@dataclass(frozen=True)
class SiteEnvironment:
    """One distinct neighbour environment of a site type."""

    site: int
    rank: int  # 1 = most common
    tile: int
    count: int
    fraction: float  # share of the site type's instances
    position: tuple[int, int, int]  # grid cell of a representative instance
    first_shell: tuple[RelativeSite, ...]
    second_shell: tuple[RelativeSite, ...]
    empty_first_shell: tuple[RelativeSite, ...]  # unoccupied neighbour positions
    pattern: int = 0  # first-shell pattern this belongs to, 1 = most common
    pattern_fraction: float = 0.0  # share of the site type's instances in that pattern


@dataclass(frozen=True)
class _TileShells:
    """Candidate neighbour positions of one tile, relative to its own cell."""

    first: np.ndarray  # (F, 4): dx, dy, dz, 0-based tile
    second: np.ndarray  # (C, 4): positions two hops away, excluding first and target
    via: np.ndarray  # (C, F) bool: second-shell position reachable through first


class SiteEnvironmentAnalysis:
    """Site-type environments for one checkpoint and its structure file."""

    def __init__(self, checkpoint: Checkpoint, structure: Structure):
        self.structure = structure
        self.checkpoint_path = Path(checkpoint.filepath)
        self.n_ungrown_types = len(checkpoint.populations) // 2

        grid = checkpoint.filled_data()
        self._shape = np.array(grid.shape[:3])
        self._occupied = grid > self.n_ungrown_types

        # Group every site by site number so one type can be looked up directly.
        coords = np.argwhere(grid).astype(np.int32)
        sites = grid[grid != 0]
        order = np.argsort(sites, kind="stable")
        self._coords = coords[order]
        numbers, starts, counts = np.unique(sites[order], return_index=True, return_counts=True)
        self._ranges = {
            int(n): (int(s), int(s + c)) for n, s, c in zip(numbers, starts, counts)
        }

        expected = {int(n) + 1: int(c) for n, c in enumerate(checkpoint.populations) if c}
        self.populations_match = expected == {int(n): int(c) for n, c in zip(numbers, counts)}

        # Tiles sharing a bulk site number are the same tile type.
        bulk = checkpoint.bulk_site_numbers
        self._tile_types = np.unique(bulk, return_inverse=True)[1] + 1

        self._neighbours = {
            tile - 1: self._unique_neighbours(tile) for tile in structure.connections
        }
        self._shells = {tile: self._build_shells(tile) for tile in self._neighbours}
        self._centroids = np.array(
            [
                np.mean([atom.frac for atom in structure.templates[tile + 1].atoms], axis=0)
                for tile in range(checkpoint.n_tiles)
            ]
        )

    # ------------------------------------------------------------------ net tables

    def _unique_neighbours(self, tile: int) -> list[tuple[int, int, int, int]]:
        """Net neighbours of a 1-based tile as (dx, dy, dz, 0-based tile), in file order."""
        own = (0, 0, 0, tile - 1)
        relative = (
            (*conn.offset, conn.target - 1) for conn in self.structure.connections[tile]
        )
        return [site for site in dict.fromkeys(relative) if site != own]

    def _build_shells(self, tile: int) -> _TileShells:
        first = self._neighbours[tile]
        excluded = {(0, 0, 0, tile), *first}
        reached_through: dict[tuple[int, int, int, int], set[int]] = {}
        for index, (dx, dy, dz, hop_tile) in enumerate(first):
            for ex, ey, ez, end_tile in self._neighbours[hop_tile]:
                site = (dx + ex, dy + ey, dz + ez, end_tile)
                if site not in excluded:
                    reached_through.setdefault(site, set()).add(index)

        second = list(reached_through)
        via = np.zeros((len(second), len(first)), dtype=bool)
        for row, site in enumerate(second):
            via[row, list(reached_through[site])] = True
        return _TileShells(
            first=np.array(first, dtype=np.int32).reshape(-1, 4),
            second=np.array(second, dtype=np.int32).reshape(-1, 4),
            via=via,
        )

    def _occupied_at(self, cells: np.ndarray, relative: np.ndarray) -> np.ndarray:
        """(N, R) bool: occupancy of R relative positions around N cells."""
        positions = cells[:, None, :] + relative[None, :, :3]
        tiles = np.broadcast_to(relative[:, 3], positions.shape[:2])
        inside = ((positions >= 0) & (positions < self._shape)).all(axis=2)
        occupied = np.zeros(positions.shape[:2], dtype=bool)
        x, y, z = positions[inside].T
        occupied[inside] = self._occupied[x, y, z, tiles[inside]]
        return occupied

    # ------------------------------------------------------------------ queries

    def site_types(self) -> list[SiteTypeSummary]:
        """Every site type present, most populated first."""
        summaries = []
        for site, (start, stop) in self._ranges.items():
            tiles = np.unique(self._coords[start:stop, 3]) + 1
            summaries.append(
                SiteTypeSummary(
                    site=site,
                    count=stop - start,
                    grown=site > self.n_ungrown_types,
                    tiles=tuple(int(t) for t in tiles),
                )
            )
        return sorted(summaries, key=lambda s: (-s.count, s.site))

    def environments(self, site: int) -> list[SiteEnvironment]:
        """Distinct environments of a site type, most common first."""
        if site not in self._ranges:
            return []
        start, stop = self._ranges[site]
        instances = self._coords[start:stop]

        # (tile, packed occupancy) → [count, representative cell]
        found: dict[tuple[int, bytes], list] = {}
        for tile, shells in self._shells.items():
            cells = instances[instances[:, 3] == tile, :3]
            for batch_start in range(0, len(cells), _BATCH_SIZE):
                batch = cells[batch_start : batch_start + _BATCH_SIZE]
                first = self._occupied_at(batch, shells.first)
                second = self._occupied_at(batch, shells.second)
                second &= (first.astype(np.int32) @ shells.via.T.astype(np.int32)) > 0
                packed = np.packbits(np.hstack([first, second]), axis=1)
                keys = packed.view(np.dtype((np.void, packed.shape[1]))).ravel()
                unique, index, counts = np.unique(keys, return_index=True, return_counts=True)
                for key, i, count in zip(unique, index, counts):
                    entry = found.setdefault((tile, key.tobytes()), [0, batch[i]])
                    entry[0] += int(count)

        total = stop - start
        ordered = sorted(found.items(), key=lambda kv: (-kv[1][0], kv[0][0], kv[0][1]))
        environments = [
            self._environment(site, rank, tile, count, total, cell)
            for rank, ((tile, _), (count, cell)) in enumerate(ordered, start=1)
        ]

        # Second shells vary far more than first shells, so also rank the first-shell
        # patterns: environments sharing one differ only further from the target.
        pattern_counts: Counter = Counter()
        for env in environments:
            pattern_counts[(env.tile, env.first_shell)] += env.count
        patterns = sorted(pattern_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        pattern_rank = {key: rank for rank, (key, _) in enumerate(patterns, start=1)}
        return [
            replace(
                env,
                pattern=pattern_rank[(env.tile, env.first_shell)],
                pattern_fraction=pattern_counts[(env.tile, env.first_shell)] / total,
            )
            for env in environments
        ]

    def _environment(
        self, site: int, rank: int, tile: int, count: int, total: int, cell: np.ndarray
    ) -> SiteEnvironment:
        shells = self._shells[tile]
        first = self._occupied_at(cell[None, :], shells.first)[0]
        second = self._occupied_at(cell[None, :], shells.second)[0]
        second &= (shells.via & first[None, :]).any(axis=1)

        def as_sites(rows: np.ndarray) -> tuple[RelativeSite, ...]:
            return tuple((int(x), int(y), int(z), int(t) + 1) for x, y, z, t in rows)

        return SiteEnvironment(
            site=site,
            rank=rank,
            tile=tile + 1,
            count=count,
            fraction=count / total,
            position=(int(cell[0]), int(cell[1]), int(cell[2])),
            first_shell=as_sites(shells.first[first]),
            second_shell=as_sites(shells.second[second]),
            empty_first_shell=as_sites(shells.first[~first]),
        )

    def contacts(self, env: SiteEnvironment) -> list[tuple[RelativeSite, RelativeSite]]:
        """Net edges walked to build the shells: target–first and first–second."""
        target = (0, 0, 0, env.tile)
        second = set(env.second_shell)
        edges = [(target, site) for site in env.first_shell]
        for site in env.first_shell:
            dx, dy, dz, tile = site
            for ex, ey, ez, end_tile in self._neighbours[tile - 1]:
                end = (dx + ex, dy + ey, dz + ez, end_tile + 1)
                if end in second:
                    edges.append((site, end))
        return edges

    # ------------------------------------------------------------------ export

    def docking_rows(self, env: SiteEnvironment) -> np.ndarray:
        """(N, 7) rows in the column layout of ``DockingData``."""
        sites = [
            *[((0, 0, 0, env.tile), DockingData.SHELL_CENTRAL)],
            *[(site, DockingData.SHELL_FIRST) for site in env.first_shell],
            *[(site, DockingData.SHELL_SECOND) for site in env.second_shell],
        ]
        relative = np.array([site for site, _ in sites], dtype=np.int64)
        tiles = relative[:, 3] - 1

        # Same origin as the simulation's XYZ output, in units of the a axis.
        origin = (self._shape + 1) // 2
        frac = np.asarray(env.position) + relative[:, :3] - origin + self._centroids[tiles]
        cryst = self.structure.cryst
        xyz = cryst.frac_to_cart(frac) / cryst.cell.a

        return np.column_stack(
            [
                self._tile_types[tiles],
                tiles + 1,
                [shell for _, shell in sites],
                xyz,
                np.full(len(sites), env.site),
            ]
        )

    def docking_data(self, env: SiteEnvironment, filepath: Path | str) -> DockingData:
        """Build the in-memory ``DockingData`` for an environment."""
        return DockingData(filepath=Path(filepath), raw=self.docking_rows(env))

    def write_docking_xyz(self, filepath: Path | str, env: SiteEnvironment) -> Path:
        """Write an environment as a CrystalGrower ``*_docking.XYZ`` file."""
        filepath = Path(filepath)
        rows = self.docking_rows(env)
        with filepath.open("w", encoding="utf-8") as f:
            f.write(f"{len(rows):9d}\ndocking file\n")
            for row in rows:
                f.write(
                    f"{int(row[0]):4d} {int(row[1]):4d} {int(row[2]):4d} "
                    f"{row[3]:9.3f} {row[4]:9.3f} {row[5]:9.3f} {int(row[6]):4d}\n"
                )
        return filepath
