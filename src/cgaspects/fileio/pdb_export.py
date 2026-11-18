"""Export viewport data as PDB files.

``write_docking_pdb`` writes docking atom-mode data; ``write_visual_data_pdb``
writes any ``VisualData`` (e.g. a sliced crystal) as centroids or expanded atoms.


Each docking centroid is expanded using the corresponding molecular template
(fractional → Cartesian via crystallography), following the same geometry
as ``VisualisationWidget._update_docking_atom_view``.

Shell IDs:
    30 = central molecule  → chain A, residue name "CEN"
    31 = 1st shell         → chain B, residue name "SH1"
    32 = 2nd shell         → chain C, residue name "SH2"
"""

from __future__ import annotations

import logging
import string
from pathlib import Path
from typing import Optional

import numpy as np

LOG = logging.getLogger("CGA:pdb_export")

# PDB ATOM record format (fixed-width, 80 cols)
# "ATOM  {serial:5d} {name:^4s}{alt:1s}{resname:3s} {chain:1s}{resseq:4d}{icode:1s}   "
# "{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{bfac:6.2f}          {element:>2s}  \n"
_ATOM_FMT = (
    "{record:<6s}{serial:5d} {name:^4s}{alt:1s}{resname:3s} "
    "{chain:1s}{resseq:4d}{icode:1s}   "
    "{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{bfac:6.2f}          {element:>2s}  \n"
)

# Fixed-width field limits: 5-digit serials, 4-digit residue numbers, %8.3f coords.
PDB_MAX_ATOMS = 99_999
_MAX_RESSEQ = 9_999
_COORD_MIN, _COORD_MAX = -999.999, 9999.999
# Residue numbers roll over into the next chain ID every 9999 residues.
_CHAIN_IDS = string.ascii_uppercase + string.ascii_lowercase + string.digits

_SHELL_CHAIN = {30: "A", 31: "B", 32: "C"}
_SHELL_RESNAME = {30: "CEN", 31: "SH1", 32: "SH2"}


def write_docking_pdb(
    filepath: Path | str,
    docking_data,
    mol_cart_templates: dict,
    a_axis: float = 1.0,
    crystallography=None,
) -> Path:
    """Write docking atom-mode coordinates to a PDB file.

    Parameters
    ----------
    filepath:
        Output ``.pdb`` path.
    docking_data:
        ``DockingData`` instance from ``xyz_file.py``.
    mol_cart_templates:
        Dict built by ``VisualisationWidget._precompute_mol_templates``.
        Keys are mol_type integers; each value has keys:
        ``cart`` (N,3), ``centroid`` (3,), ``symbols`` list[str], ``bonds`` list[(i,j)].
    a_axis:
        Length of the crystallographic a-axis in Å. Docking coords are in units
        of this length.  Defaults to 1.0 (no scaling).
    crystallography:
        Optional crystallography object with a ``cell`` having at least ``a, b, c,
        alpha, beta, gamma``. ``sgroup`` and ``z`` are used if present, otherwise
        defaulting to ``"P 1"`` and ``1`` respectively.

    Returns
    -------
    Path
        Resolved path to the written file.
    """
    filepath = Path(filepath)

    coords_xyz = docking_data.coords.astype(np.float64)
    cart_positions = (coords_xyz * a_axis).astype(np.float64)
    mol_types = docking_data.mol_types
    shells = docking_data.shells

    serial = 1
    res_seq = 1
    conect_lines: list[str] = []

    with filepath.open("w", encoding="utf-8") as f:
        _write_cryst1(f, crystallography)

        for i, (mol_type, centroid_pos, shell) in enumerate(
            zip(mol_types, cart_positions, shells)
        ):
            tmpl = mol_cart_templates.get(mol_type)
            if tmpl is None:
                LOG.warning("No template for mol_type %d at centroid %d — skipping", mol_type, i)
                continue

            chain = _SHELL_CHAIN.get(shell, "X")
            resname = _SHELL_RESNAME.get(shell, "UNK")
            symbols: list[str] = tmpl["symbols"]
            cart: np.ndarray = tmpl["cart"]        # (M, 3)
            centroid_tmpl: np.ndarray = tmpl["centroid"]  # (3,)

            atom_positions = cart + (centroid_pos - centroid_tmpl)
            first_serial_in_residue = serial

            for j, (sym, pos) in enumerate(zip(symbols, atom_positions)):
                # Atom name: pad single-char element to column 14 (1-indexed) convention
                name = f"{sym:<2s}" if len(sym) == 1 else sym[:4]
                f.write(
                    _ATOM_FMT.format(
                        record="ATOM",
                        serial=serial,
                        name=name,
                        alt="",
                        resname=resname,
                        chain=chain,
                        resseq=res_seq,
                        icode="",
                        x=float(pos[0]),
                        y=float(pos[1]),
                        z=float(pos[2]),
                        occ=1.00,
                        bfac=float(shell),
                        element=sym[:2],
                    )
                )
                serial += 1

            # CONECT records for this molecule
            for a1, a2 in tmpl.get("bonds", []):
                s1 = first_serial_in_residue + a1
                s2 = first_serial_in_residue + a2
                conect_lines.append(f"CONECT{s1:5d}{s2:5d}\n")

            res_seq += 1

        for line in conect_lines:
            f.write(line)

        f.write("END\n")

    LOG.info("Docking PDB written: %s  (%d residues)", filepath, res_seq - 1)
    return filepath.resolve()


def _write_cryst1(f, crystallography) -> None:
    """Write the CRYST1 record when a crystallography with a cell is available."""
    if crystallography is None or crystallography.cell is None:
        return
    cell = crystallography.cell
    sgroup = getattr(cell, "sgroup", "P 1")
    z = getattr(cell, "z", 1)
    f.write(
        f"CRYST1{cell.a:9.3f}{cell.b:9.3f}{cell.c:9.3f}"
        f"{cell.alpha:7.2f}{cell.beta:7.2f}{cell.gamma:7.2f}"
        f"  {sgroup:<11s}{z:4d}\n"
    )


def _resname(mol_type: int) -> str:
    return f"M{mol_type:02d}" if 0 <= mol_type < 100 else "MOL"


def write_visual_data_pdb(
    filepath: Path | str,
    vd,
    crystallography=None,
    expand_atoms: bool = False,
) -> tuple[Path, int]:
    """Write a ``VisualData`` (e.g. the visible, sliced crystal) to a PDB file.

    One residue per molecule; residue names are ``M<mol_type>``. With
    *expand_atoms* each centroid is expanded through its molecular template into
    ATOM records plus CONECT bonds; otherwise each centroid is written as a
    single ``X`` pseudo-atom HETATM record.

    Coordinates are written as stored in *vd* — Cartesian Å when a structure
    file is loaded, XYZ units otherwise.

    Raises
    ------
    ValueError
        If the data exceeds the PDB fixed-width limits (atom count or
        coordinate range).

    Returns
    -------
    (Path, int)
        Resolved path to the written file and the number of atoms written.
    """
    filepath = Path(filepath)
    centroids = vd.centroids.astype(np.float64)
    mol_types = vd.mol_types

    if expand_atoms:
        has_tmpl = np.isin(mol_types, list(vd.templates))
        if not has_tmpl.all():
            missing = sorted(set(mol_types[~has_tmpl].tolist()))
            LOG.warning("No template for mol_types %s — those molecules are skipped", missing)
        centroids, mol_types = centroids[has_tmpl], mol_types[has_tmpl]

    # Per-type coordinate bounds: centroid extent plus template atom offsets.
    n_atoms, lo, hi = 0, 0.0, 0.0
    for mt in np.unique(mol_types):
        c = centroids[mol_types == mt]
        if expand_atoms:
            offsets = vd.templates[mt]["cart"] - vd.templates[mt]["centroid"]
        else:
            offsets = np.zeros((1, 3))
        n_atoms += len(c) * len(offsets)
        lo = min(lo, float((c.min(axis=0) + offsets.min(axis=0)).min()))
        hi = max(hi, float((c.max(axis=0) + offsets.max(axis=0)).max()))

    if n_atoms > PDB_MAX_ATOMS:
        raise ValueError(
            f"{n_atoms:,} atoms exceeds the PDB limit of {PDB_MAX_ATOMS:,}. "
            "Slice to a smaller region and try again."
        )
    if lo < _COORD_MIN or hi > _COORD_MAX:
        raise ValueError(
            f"Coordinates span {lo:.1f} to {hi:.1f}, outside the PDB range "
            f"{_COORD_MIN} to {_COORD_MAX}."
        )

    serial = 1
    conect_lines: list[str] = []

    with filepath.open("w", encoding="utf-8") as f:
        _write_cryst1(f, crystallography)

        for i, (pos, mol_type) in enumerate(zip(centroids, mol_types)):
            chain_idx, res_idx = divmod(i, _MAX_RESSEQ)
            fields = dict(
                alt="",
                resname=_resname(int(mol_type)),
                chain=_CHAIN_IDS[chain_idx],
                resseq=res_idx + 1,
                icode="",
                occ=1.00,
                bfac=0.00,
            )

            if not expand_atoms:
                f.write(
                    _ATOM_FMT.format(
                        record="HETATM", serial=serial, name="X", element="",
                        x=float(pos[0]), y=float(pos[1]), z=float(pos[2]), **fields,
                    )
                )
                serial += 1
                continue

            tmpl = vd.templates[mol_type]
            atom_positions = tmpl["cart"] + (pos - tmpl["centroid"])
            first_serial = serial
            for sym, apos in zip(tmpl["symbols"], atom_positions):
                name = f"{sym:<2s}" if len(sym) == 1 else sym[:4]
                f.write(
                    _ATOM_FMT.format(
                        record="ATOM", serial=serial, name=name, element=sym[:2],
                        x=float(apos[0]), y=float(apos[1]), z=float(apos[2]), **fields,
                    )
                )
                serial += 1
            for a1, a2 in tmpl.get("bonds", []):
                conect_lines.append(f"CONECT{first_serial + a1:5d}{first_serial + a2:5d}\n")

        f.writelines(conect_lines)
        f.write("END\n")

    LOG.info("PDB written: %s  (%d atoms)", filepath, serial - 1)
    return filepath.resolve(), serial - 1
