"""Tests for crystal net connectivity parsing from CrystoGen structure files."""

from pathlib import Path

import pytest

from cgaspects.fileio.structure import Structure, TileConnection

RES = Path(__file__).parent.parent / "res"


@pytest.fixture(scope="module")
def structure() -> Structure:
    return Structure.from_file(RES / "test_structure.txt")


def test_templates_parsed(structure):
    assert sorted(structure.templates) == [1, 2, 3, 4]
    assert all(len(t.atoms) == 26 for t in structure.templates.values())


def test_connections_parsed_for_all_tiles(structure):
    assert sorted(structure.connections) == [1, 2, 3, 4]
    assert all(len(c) == 13 for c in structure.connections.values())


def test_connection_targets_and_offsets(structure):
    conns = structure.connections[1]
    assert conns[0] == TileConnection(target=3, offset=(-1, 1, -1))
    assert conns[1] == TileConnection(target=2, offset=(0, 1, -1))
    # Same-cell neighbour keeps its explicit (0,0,0) offset
    assert conns[3] == TileConnection(target=4, offset=(0, 0, 0))
    # Self-connection into the adjacent cell along c
    assert conns[6] == TileConnection(target=1, offset=(0, 0, 1))

    assert structure.connections[4][-1] == TileConnection(target=1, offset=(0, 0, -1))


def test_connections_are_reciprocal(structure):
    """Every connection t1 → t2 at offset o has a partner t2 → t1 at -o."""
    for tile, conns in structure.connections.items():
        for conn in conns:
            back = TileConnection(
                target=tile,
                offset=(-conn.offset[0], -conn.offset[1], -conn.offset[2]),
            )
            assert back in structure.connections[conn.target], (
                f"missing reciprocal of M{tile} → M{conn.target} {conn.offset}"
            )


def test_all_targets_are_valid_tiles(structure):
    tiles = set(structure.templates)
    for conns in structure.connections.values():
        assert {c.target for c in conns} <= tiles
