from pathlib import Path
from dataclasses import dataclass, field
import logging
import re

import numpy as np

from ..gui.utils.crystallography import Cell, Crystallography
from ..utils.periodic_table import get_cov_radius

logger = logging.getLogger("CGA:Structure")


@dataclass
class MolAtom:
    symbol: str
    frac: np.ndarray  # fractional coordinates, shape (3,)


@dataclass
class TileConnection:
    """One face connection from a tile (molecule) to a neighbouring tile.

    ``target`` is the neighbour's tile number within the unit cell and
    ``offset`` the (x, y, z) unit-cell translation of the neighbour relative
    to the source tile's cell ((0, 0, 0) when within the same cell).
    """

    target: int
    offset: tuple[int, int, int]


@dataclass
class MolTemplate:
    formula: str
    mol_type: int
    atoms: list[MolAtom]
    bonds: list[tuple[int, int]]  # 0-based atom index pairs


@dataclass
class Structure:
    """All data parsed from a CrystoGen structure file."""

    filepath: Path
    cell: Cell | None
    cryst: Crystallography | None
    templates: dict[int, MolTemplate] = field(default_factory=dict)
    # tile number → face connections to neighbouring tiles (the crystal net)
    connections: dict[int, list[TileConnection]] = field(default_factory=dict)

    @classmethod
    def from_file(cls, file_path: str | Path) -> "Structure":
        """Parse a CrystoGen structure file, extracting all data in one pass."""
        file_path = Path(file_path)
        if not file_path.exists():
            logger.warning("Structure file not found: %s", file_path)
            return cls(filepath=file_path, cell=None, cryst=None)

        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except OSError as e:
            logger.error("Error reading structure file %s: %s", file_path, e)
            return cls(filepath=file_path, cell=None, cryst=None)

        non_prim_idx = next((i for i, ln in enumerate(lines) if "Non primitive data" in ln), None)

        cell = _parse_cell(lines, non_prim_idx, file_path.name)
        cryst = Crystallography(cell) if cell is not None else None
        templates = _parse_templates(lines, non_prim_idx, file_path.name)
        connections = _parse_connections(lines, non_prim_idx, file_path.name)

        if cryst is not None:
            for tmpl in templates.values():
                if not tmpl.bonds:
                    tmpl.bonds = _infer_bonds(tmpl.atoms, cryst)

        return cls(
            filepath=file_path,
            cell=cell,
            cryst=cryst,
            templates=templates,
            connections=connections,
        )

    @property
    def n_tiles(self):
        return len(self.templates)


def _parse_cell(lines: list[str], non_prim_idx: int | None, filename: str = "") -> Cell | None:
    """Parse lattice parameters from the 'Non primitive data' block."""
    if non_prim_idx is None:
        logger.warning("'Non primitive data' section not found in %s", filename)
        return None
    try:
        a, b, c = map(float, lines[non_prim_idx + 1].strip().split())
        alpha, beta, gamma = map(float, lines[non_prim_idx + 2].strip().split())
        logger.info(
            "Parsed lattice parameters from %s: a=%.4f b=%.4f c=%.4f α=%.3f° β=%.3f° γ=%.3f°",
            filename,
            a,
            b,
            c,
            alpha,
            beta,
            gamma,
        )
        return Cell(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)
    except (ValueError, IndexError) as e:
        logger.warning("Could not parse lattice parameters from %s: %s", filename, e)
        return None


# Tile header in the net section, e.g. "C16H10 1 (1,0) 13":
# formula, tile number, one or more (n_vertices, Q) pairs, neighbour count.
_TILE_HEADER_RE = re.compile(
    r"^(\S+)\s+(\d+)((?:\s*\(\s*\d+\s*,\s*\d+\s*\))+)\s+(\d+)\s*$"
)

# Neighbour entry, e.g. "3(-1,1,-1)" or bare "3" (same unit cell).
_NEIGHBOUR_RE = re.compile(
    r"(\d+)(?:\(\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\))?"
)


def _parse_connections(
    lines: list[str], non_prim_idx: int | None, filename: str = ""
) -> dict[int, list[TileConnection]]:
    """Parse the crystal net (tile → neighbour tile connectivity) block.

    The net section precedes the lattice-parameter lines: for each tile in the
    unit cell there is a header line ("FORMULA TILE_NUM (v,Q)... N_NEIGHBOURS")
    followed by the list of neighbouring tiles through faces, each optionally
    carrying a relative unit-cell offset.
    """
    end = non_prim_idx if non_prim_idx is not None else len(lines)
    connections: dict[int, list[TileConnection]] = {}

    i = 0
    while i < end:
        header = _TILE_HEADER_RE.match(lines[i].strip())
        if header is None:
            i += 1
            continue

        tile_num = int(header.group(2))
        n_neighbours = int(header.group(4))
        neighbours: list[TileConnection] = []
        i += 1

        # Collect neighbour entries from the following lines.  Neighbour lines
        # start with a digit; vertex lines (e.g. "C 2[3] ...") start with an
        # atom symbol and terminate the neighbour list.
        while i < end and len(neighbours) < n_neighbours:
            stripped = lines[i].strip()
            if not stripped:
                i += 1
                continue
            if not stripped[0].isdigit():
                break
            for m in _NEIGHBOUR_RE.finditer(stripped):
                if len(neighbours) >= n_neighbours:
                    break
                offset = (
                    (int(m.group(2)), int(m.group(3)), int(m.group(4)))
                    if m.group(2) is not None
                    else (0, 0, 0)
                )
                neighbours.append(TileConnection(target=int(m.group(1)), offset=offset))
            i += 1

        if len(neighbours) != n_neighbours:
            logger.warning(
                "Tile %d in %s: expected %d neighbours, parsed %d",
                tile_num,
                filename,
                n_neighbours,
                len(neighbours),
            )
        connections[tile_num] = neighbours

    if connections:
        logger.info(
            "Parsed net connectivity for %d tile(s) from %s", len(connections), filename
        )
    return connections


def _parse_templates(
    lines: list[str], non_prim_idx: int | None, filename: str = ""
) -> dict[int, MolTemplate]:
    """Parse per-molecule-type atom/bond blocks after the 'Non primitive data' header."""
    if non_prim_idx is None:
        return {}

    templates: dict[int, MolTemplate] = {}
    i = non_prim_idx + 3  # skip "Non primitive data" + abc line + angles line

    while i < len(lines):
        parts = lines[i].strip().split()

        # Molecule header: FORMULA  TYPE  N_ATOMS/N_BONDS
        if len(parts) == 3 and "/" in parts[2]:
            try:
                formula = parts[0]
                mol_type = int(parts[1])
                n_atoms, n_bonds = map(int, parts[2].split("/"))
            except ValueError:
                i += 1
                continue

            i += 1
            atoms, i = _parse_atoms(lines, i, n_atoms)

            while i < len(lines) and not lines[i].strip():
                i += 1

            bonds, i = _parse_bonds(lines, i, n_bonds)

            templates[mol_type] = MolTemplate(
                formula=formula, mol_type=mol_type, atoms=atoms, bonds=bonds
            )
        else:
            i += 1

    if templates:
        logger.info("Parsed %d molecule template(s) from %s", len(templates), filename)

    return templates


def _parse_atoms(lines: list[str], i: int, n_atoms: int) -> tuple[list[MolAtom], int]:
    atoms: list[MolAtom] = []
    for _ in range(n_atoms):
        if i >= len(lines):
            break
        parts = lines[i].strip().split()
        if len(parts) >= 5:
            atoms.append(
                MolAtom(
                    symbol=parts[1],
                    frac=np.array(
                        [float(parts[2]), float(parts[3]), float(parts[4])], dtype=np.float64
                    ),
                )
            )
        i += 1
    return atoms, i


def _infer_bonds(atoms: list[MolAtom], cryst: Crystallography) -> list[tuple[int, int]]:
    """Infer intramolecular bonds via covalent-radius cutoff: dist < (r_A + r_B) * 1.3."""
    if not atoms:
        return []
    cart = cryst.frac_to_cart(np.array([a.frac for a in atoms]))
    radii = np.array([get_cov_radius(a.symbol) for a in atoms])
    bonds: list[tuple[int, int]] = []
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            if np.linalg.norm(cart[i] - cart[j]) < (radii[i] + radii[j]) * 1.3:
                bonds.append((i, j))
    logger.debug("Inferred %d bonds for %d atoms", len(bonds), len(atoms))
    return bonds


def _parse_bonds(lines: list[str], i: int, n_bonds: int) -> tuple[list[tuple[int, int]], int]:
    bonds: list[tuple[int, int]] = []
    for _ in range(n_bonds):
        if i >= len(lines):
            break
        parts = lines[i].strip().split()
        if len(parts) == 2:
            try:
                bonds.append((int(parts[0]) - 1, int(parts[1]) - 1))
            except ValueError:
                pass
        i += 1
    return bonds, i
