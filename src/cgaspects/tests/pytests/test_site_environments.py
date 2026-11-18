"""Tests for site-type environment enumeration from checkpoint files."""

from pathlib import Path

import numpy as np
import pytest

from cgaspects.analysis.site_environments import SiteEnvironmentAnalysis
from cgaspects.fileio.cg_checkpoint import Checkpoint
from cgaspects.fileio.structure import MolAtom, MolTemplate, Structure, TileConnection
from cgaspects.fileio.xyz_file import DockingData
from cgaspects.gui.utils.crystallography import Cell, Crystallography

RES = Path(__file__).parent.parent / "res"

BULK = 4
GROWTH = 1
GROWN = 3


def chain_structure() -> Structure:
    """One tile per cubic cell, linked to its neighbours along a and b."""
    cell = Cell(a=2.0, b=2.0, c=2.0, alpha=90.0, beta=90.0, gamma=90.0)
    offsets = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)]
    return Structure(
        filepath=Path("chain.txt"),
        cell=cell,
        cryst=Crystallography(cell),
        templates={
            1: MolTemplate("X", 1, [MolAtom("C", np.array([0.25, 0.25, 0.25]))], []),
        },
        connections={1: [TileConnection(target=1, offset=o) for o in offsets]},
    )


def checkpoint_from(grid: np.ndarray) -> Checkpoint:
    numbers, counts = np.unique(grid[grid > 0], return_counts=True)
    populations = np.zeros(4, dtype=np.int64)
    populations[numbers - 1] = counts
    return Checkpoint(
        filepath=Path("chain_checkpoint.txt"),
        n_tiles=1,
        crystallography=None,
        data=grid,
        populations=populations,
        bulk_gaps=np.empty((0, 4), dtype=np.int32),
    )


@pytest.fixture
def analysis() -> SiteEnvironmentAnalysis:
    grid = np.zeros((9, 9, 3, 1), dtype=np.int32)
    # Three growth sites around a row of grown molecules: the two above the row
    # share an environment, the one at the end of the row does not.
    grid[2:7, 4, 1, 0] = GROWN
    grid[3, 5, 1, 0] = GROWTH
    grid[5, 5, 1, 0] = GROWTH
    grid[7, 4, 1, 0] = GROWTH
    grid[1, 1, 1, 0] = BULK
    return SiteEnvironmentAnalysis(checkpoint_from(grid), chain_structure())


def test_site_types_report_state_and_population(analysis):
    summaries = {s.site: s for s in analysis.site_types()}
    assert summaries[GROWN].count == 5 and summaries[GROWN].grown
    assert summaries[GROWTH].count == 3 and not summaries[GROWTH].grown


def test_environments_ranked_by_frequency(analysis):
    environments = analysis.environments(GROWTH)
    assert [env.rank for env in environments] == [1, 2]
    assert [env.count for env in environments] == [2, 1]
    assert environments[0].fraction == pytest.approx(2 / 3)
    assert sum(env.count for env in environments) == 3


def test_first_shell_patterns_ranked_by_frequency(analysis):
    environments = analysis.environments(GROWTH)
    assert [env.pattern for env in environments] == [1, 2]
    assert environments[0].pattern_fraction == pytest.approx(2 / 3)
    assert environments[1].pattern_fraction == pytest.approx(1 / 3)


def test_shells_exclude_the_target(analysis):
    environments = analysis.environments(GROWN)
    centre = next(env for env in environments if env.position == (3, 4, 1))
    assert set(centre.first_shell) == {(1, 0, 0, 1), (-1, 0, 0, 1)}
    assert (0, 0, 0, 1) not in centre.second_shell


def test_second_shell_needs_an_occupied_first_shell(analysis):
    env = next(e for e in analysis.environments(GROWTH) if e.position == (7, 4, 1))
    assert env.first_shell == ((-1, 0, 0, 1),)
    assert env.second_shell == ((-2, 0, 0, 1),)
    assert (1, 0, 0, 1) in env.empty_first_shell


def test_unknown_site_has_no_environments(analysis):
    assert analysis.environments(2) == []


def test_docking_export_round_trips(analysis, tmp_path):
    env = analysis.environments(GROWTH)[0]
    path = analysis.write_docking_xyz(tmp_path / "site_docking.XYZ", env)
    docking = DockingData.from_file(path)

    assert (docking.shells == DockingData.SHELL_CENTRAL).sum() == 1
    assert (docking.shells == DockingData.SHELL_FIRST).sum() == len(env.first_shell)
    assert (docking.shells == DockingData.SHELL_SECOND).sum() == len(env.second_shell)
    assert set(docking.site_numbers) == {GROWTH}
    # Cell (x, y, z) minus the box centre (5, 5, 2) plus the 0.25 fractional centroid.
    x, y, z = env.position
    assert docking.coords[0] == pytest.approx([x - 4.75, y - 4.75, z - 1.75])


@pytest.fixture(scope="module")
def fixture_analysis() -> SiteEnvironmentAnalysis:
    structure = Structure.from_file(RES / "test_structure.txt")
    checkpoint = Checkpoint.from_file(RES / "test_checkpoint.txt", structure.n_tiles)
    return SiteEnvironmentAnalysis(checkpoint, structure)


def test_bulk_cells_restored_to_match_header(fixture_analysis):
    assert fixture_analysis.populations_match


def test_environment_counts_cover_every_instance(fixture_analysis):
    summary = fixture_analysis.site_types()[0]
    environments = fixture_analysis.environments(summary.site)
    assert sum(env.count for env in environments) == summary.count
    counts = [env.count for env in environments]
    assert counts == sorted(counts, reverse=True)
