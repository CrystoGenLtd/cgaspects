"""Crystal net file reader — parses CrystoGen (formerly CrystalGrower) .net interaction files.

Adapted from cg-solventmaps/scripts/python/cg_net.py.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

LOG = logging.getLogger("CG-NET")


@dataclass
class Interaction:
    serial: int = field(repr=False)
    id: int
    mol_type: str
    molecule_info: str = field(repr=False)
    r: float
    energy: float | str = field(default=None)

    def add_energy(self, energy: float | str):
        if self.energy is None:
            self.energy = energy
        else:
            raise ValueError("Use 'modify_energy' to modify energy value.")

    def modify_id(self, idx: int):
        self.id = idx

    def modify_energy(self, energy: float):
        self.energy = energy

    def __eq__(self, other):
        if not isinstance(other, Interaction):
            return NotImplemented
        return self.id == other.id and self.mol_type == other.mol_type and self.r == other.r


@dataclass
class Molecule:
    serial: int
    label: str
    interactions: list[Interaction] = field(default_factory=list)

    @property
    def energies(self) -> np.ndarray:
        added = []
        energies = []
        for interaction in self.interactions:
            if interaction.serial in added:
                continue
            energies.append(interaction.energy)
            added.append(interaction.serial)
        return np.array(energies)

    @property
    def unique_energies(self) -> np.ndarray:
        added = []
        energies = []
        for interaction in self.interactions:
            if interaction in added:
                continue
            energies.append(interaction.energy)
            added.append(interaction)
        return np.array(energies)

    @property
    def n_interactions(self) -> int:
        return len(self.interactions)

    def add_interaction(self, interaction: Interaction):
        self.interactions.append(interaction)

    def add_energy(self, energy: float | str):
        if isinstance(energy, str):
            try:
                energy = float(energy)
            except ValueError:
                if "_" in energy:
                    try:
                        _ = int(energy.split("_")[-1])
                    except ValueError:
                        LOG.error("Cannot parse energy placeholder: %s", energy)
                        raise
                else:
                    LOG.error("Cannot parse energy: %s", energy)
                    raise

        for interaction in self.interactions:
            if interaction.energy is None:
                lowest_unoccupied_id = interaction.id
                break
        else:
            return

        for interaction in self.interactions:
            if interaction.id == lowest_unoccupied_id:
                interaction.add_energy(energy)

    def group_interactions(self, using="r"):
        grouping_dict = {}
        for interaction in self.interactions:
            group_val = getattr(interaction, using)
            if group_val not in grouping_dict:
                grouping_dict[group_val] = {"count": 0, "total_energy": 0}
            grouping_dict[group_val]["count"] += 1
            grouping_dict[group_val]["total_energy"] += interaction.energy

        group_dict_keys = list(grouping_dict.keys())
        for r, data in grouping_dict.items():
            average_energy = data["total_energy"] / data["count"]
            idx = group_dict_keys.index(r) + 1
            for interaction in self.interactions:
                if getattr(interaction, using) == r:
                    interaction.modify_energy(average_energy)
                    interaction.modify_id(idx)


class CGNet:
    def __init__(self, filename: str | Path):
        self.filename: Path = Path(filename)
        self.molecules: list[Molecule] = []
        self.interaction_order_counter: int = 1

    @property
    def energies(self) -> dict:
        return {mol.label: mol.energies for mol in self.molecules}

    @property
    def unique_energies(self) -> dict:
        return {mol.label: mol.unique_energies for mol in self.molecules}

    def parse(self):
        with open(self.filename, "r", encoding="utf-8") as fh:
            lines = fh.readlines()

        mol_serial = 0
        reading_interactions = False
        reading_energies = False
        initial = True
        molecule = None

        for line in lines:
            if re.match(r"^\d+:", line):
                reading_interactions = True
                match = re.match(r"^(\d+):\[(\d[A-Za-z])\](.*?)R=([\d\.]+)", line)
                if match:
                    idx, label, molecule_info, r = match.groups()
                    idx = int(idx)
                    r = float(r)

                    if reading_energies or initial:
                        molecule = Molecule(mol_serial, label)
                        self.molecules.append(molecule)
                        mol_serial += 1
                        reading_energies = False
                        initial = False

                    interaction = Interaction(
                        serial=self.interaction_order_counter,
                        id=idx,
                        mol_type=label,
                        molecule_info=molecule_info,
                        r=r,
                    )
                    self.interaction_order_counter += 1
                    if molecule is not None:
                        molecule.add_interaction(interaction)
            else:
                reading_energies = True
                reading_interactions = False
                try:
                    if molecule is not None:
                        molecule.add_energy(line.strip())
                except (ValueError, UnboundLocalError):
                    continue

        LOG.info("Parsed %d molecules from %s", len(self.molecules), self.filename.name)

    def group_net(self, using="r"):
        for molecule in self.molecules:
            molecule.group_interactions(using)

    def __repr__(self) -> str:
        return f"CGNet({self.filename.name!r}, {len(self.molecules)} molecules)"
