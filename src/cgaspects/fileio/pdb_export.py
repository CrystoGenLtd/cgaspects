"""Export docking atom-mode data as a PDB file.

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
        # CRYST1 record
        if crystallography is not None:
            cell = crystallography.cell
            sgroup = getattr(cell, "sgroup", "P 1")
            z = getattr(cell, "z", 1)
            f.write(
                f"CRYST1{cell.a:9.3f}{cell.b:9.3f}{cell.c:9.3f}"
                f"{cell.alpha:7.2f}{cell.beta:7.2f}{cell.gamma:7.2f}"
                f"  {sgroup:<11s}{z:4d}\n"
            )

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
