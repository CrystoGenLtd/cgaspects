from pathlib import Path
from dataclasses import dataclass, field
import logging

import numpy as np

from ..gui.utils.crystallography import Cell, Crystallography
from ..utils.periodic_table import get_cov_radius

logger = logging.getLogger("CGA:Structure")


@dataclass
class MolAtom:
    symbol: str
    frac: np.ndarray  # fractional coordinates, shape (3,)


@dataclass
class MolTemplate:
    formula: str
    mol_type: int
    atoms: list[MolAtom]
    bonds: list[tuple[int, int]]  # 0-based atom index pairs


@dataclass
class Structure:
    """All data parsed from a CrystalGrower structure file."""

    filepath: Path
    cell: Cell | None
    cryst: Crystallography | None
    templates: dict[int, MolTemplate] = field(default_factory=dict)

    @classmethod
    def from_file(cls, file_path: str | Path) -> "Structure":
        """Parse a CrystalGrower structure file, extracting all data in one pass."""
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

        if cryst is not None:
            for tmpl in templates.values():
                if not tmpl.bonds:
                    tmpl.bonds = _infer_bonds(tmpl.atoms, cryst)

        return cls(filepath=file_path, cell=cell, cryst=cryst, templates=templates)

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
