import argparse
import json
import logging
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from cgaspects.analysis.shape_analysis import CrystalShape
from cgaspects.utils.cg_net import CGNet
from cgaspects.utils.data_structures import shape_info_tuple
from cgaspects.fileio.log_setup import setup_logging

LOG = logging.getLogger("SOL-MAP")
log_dict = {"basic": "DEBUG", "console": "INFO"}
setup_logging(**log_dict)


class SolventScreen:
    def __init__(self, folderpath: Path, solvent_dict: Path, lmax=20):
        self.folderpath = Path(folderpath)
        self.read_shapes(self.folderpath)

        self.xyz_info = None
        self.wulff_info = None
        self.cda_info = None
        self.occ_info = None
        self.owf_info = None

        self.lmax = lmax
        self.crystal: CrystalShape = CrystalShape()
        self.solvent_dict = solvent_dict
        self.csv = None

        self.set_xyz_info()
        self.set_occ_info()
        # self.set_owf_info()
        # self.set_wulff_info()

    def read_shapes(self, folderpath: Path = None):

        if folderpath is None:
            folderpath = self.folderpath

        self.xyz_list = list(folderpath.rglob("*.XYZ"))
        self.wulff_list = list(folderpath.rglob("SHAPE*"))
        self.cda_list = list(folderpath.rglob("*simulation_parameters.txt"))
        self.occs_list = list(folderpath.rglob("*.*.stdout"))
        self.owf_list = list(folderpath.rglob("*.owf.json"))

        message = (
            f"Found:\n"
            f" {len(self.xyz_list)} XYZs\n"
            f" {len(self.wulff_list)} Wulff shapes\n"
            f" {len(self.cda_list)} CDA files\n"
            f" {len(self.occs_list)} OCC outputs\n"
            f" {len(self.owf_list)} wavefunction files"
        )

        LOG.info(message)

    def set_occ_info(self):
        self.occ_info = defaultdict(list)

        for output in self.occs_list:
            solubility = []
            output = Path(output)
            solvent = output.name.split(".")[-2]
            LOG.debug("Checking: %s", solvent)
            with open(output, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for line in lines:
                if line.startswith("solubility (g/L)"):
                    solubility.append(float(line.split()[-1]))

            if solubility:
                LOG.debug("Solubility in %s: %s", solvent, solubility)
                self.occ_info[solvent].append(solubility)
            else:
                LOG.warning("Solubility was not found for %s!", solvent)

        LOG.info("Solvent Information successfully read!\n%s", self.occ_info)

    def set_owf_info(self):
        self.owf_info = defaultdict(list)

        for owf in self.owf_list:
            solubility = []
            output = Path(owf)
            solvent = output.name.split(".")[-2]
            LOG.debug("OWF for form: %s", solvent)
            with open(output, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for line in lines:
                if line.startswith("solubility (g/L)"):
                    solubility.append(float(line.split()[-1]))

            if solubility:
                LOG.info("Solubility in %s: %s", solvent, solubility)
                self.occ_info[solvent].append(solubility)
            else:
                LOG.warning("Solubility was not found for %s!", solvent)

    def set_xyz_info(self):
        self.xyz_info = self.get_shape_info(
            self.xyz_list, self.solvent_dict, get_energy=True
        )

    def set_wulff_info(self):
        self.xyz_info = self.get_shape_info(self.wulff_list, self.solvent_dict)

    def set_cda_info(self, directions, get_energy=True):
        ar_dict = defaultdict(list)

        for cda in self.cda_list:
            solvent = str(Path(cda).parent).rsplit("_", maxsplit=1)[-1]
            add = True
            with open(cda, "r", encoding="utf-8") as sim_file:
                lines = sim_file.readlines()

            for line in lines:
                try:
                    if line.startswith("Size of crystal at frame output"):
                        frame = lines.index(line) + 1
                except NameError:
                    continue

            len_info_lines = lines[frame:]
            for len_line in len_info_lines:
                for direction in directions:
                    if len_line.startswith(direction):
                        ar_dict[direction].append(float(len_line.split(" ")[-2]))
                        if solvent not in list(ar_dict["solvent"]):
                            ar_dict["solvent"].append(solvent)

            if get_energy:

                netfile = Path(cda).parent / "net.txt"
                net = CGNet(netfile)
                net.parse()
                energies = net.unique_energies

                print(f"{netfile.parent.name:>50s}  --> {len(energies)}")
                for i, energy in enumerate(energies):
                    i += 1
                    if np.isnan(energy):
                        continue

                    ar_dict[f"Int_{i}"].append(energy)

        LOG.debug("Selected Directions =", *directions)

        for key, value in ar_dict.items():
            _str = f"{key:>20s}:  {len(value)}"
            LOG.debug(_str)

        # Convert dictionary values to NumPy arrays
        ar_arrays = {
            key: np.array(value) for key, value in ar_dict.items() if key in directions
        }

        # Debug log
        LOG.debug("ar_arrays: ", ar_arrays)

        # Aspect ratio calculation
        for i in range(len(directions) - 1):
            print(f"PAIR {i} {directions[i]} {directions[i + 1]}")

            aspect_ratio_key = f"AspectRatio_{directions[i]}/{directions[i+1]}"
            numerator = ar_arrays[directions[i]]
            denominator = ar_arrays[directions[i + 1]]

            # Avoid division by zero
            with np.errstate(divide="ignore", invalid="ignore"):
                aspect_ratio = np.where(
                    denominator != 0, numerator / denominator, np.inf
                )
                aspect_ratio = np.where(
                    denominator == 0, np.sign(numerator) * np.inf, aspect_ratio
                )

            ar_arrays[aspect_ratio_key] = aspect_ratio

        # Convert back to dictionary of lists for consistency with original format
        ar_dict.update({key: value.tolist() for key, value in ar_arrays.items()})

        self.cda_info = ar_dict

        return self.cda_info

    def get_shape_info(self, shapes, solvent_json, get_energy=False):
        shape_info = defaultdict(list)
        solvent = None
        with open(solvent_json, "r", encoding="utf-8") as f:
            sol_dict = json.load(f)

        for shape in shapes:
            print(shape)
            shape = Path(shape)
            self.crystal.set_xyz(shape)
            name_split = str(shape.parent.name).rsplit("_", maxsplit=1)
            print(name_split)
            solvent = name_split[-1] if name_split[0] == "solvent" else None
            if shape.name.startswith("SHAPE"):
                with open(shape, "r", encoding="utf-8") as s:
                    lines = s.readlines()

                if "water" in shape.name or "vacuum" in shape.name:
                    continue

                sol_line = lines[1]
                if sol_line.startswith("Solvent") and solvent is None:
                    solvent = (
                        sol_line.split(":")[-1]
                        .lstrip()
                        .replace("\n", "")
                        .replace("E-Z", "E/Z")
                        .replace("cis-trans", "cis/trans")
                    )

            if solvent not in sol_dict:
                LOG.debug("%s", sol_dict.keys())
                LOG.error(
                    "%s Couldn't find solvent! Please check your file structure.",
                    solvent,
                )
                continue

            LOG.info("%s : %s", solvent, shape)

            analysis: shape_info_tuple = self.crystal.get_zingg_analysis()
            shape_info["solvent"].append(solvent)
            shape_info["ar1"].append(analysis.aspect1)
            shape_info["ar2"].append(analysis.aspect2)
            shape_info["sa"].append(analysis.sa)
            shape_info["vol"].append(analysis.vol)
            shape_info["sa_vol"].append(analysis.sa_vol)

            params = sol_dict[solvent]
            shape_info["n"].append(params[0])
            shape_info["acidity"].append(params[1])
            shape_info["basicity"].append(params[2])
            shape_info["gamma"].append(params[3])
            shape_info["dielectric"].append(params[4])
            shape_info["aromatic"].append(params[5])
            shape_info["halogen"].append(params[6])

            if get_energy:
                netfile = shape.parent / "net.txt"
                net = CGNet(netfile)
                net.parse()
                energies = net.unique_energies_arr.flatten()

                print(f"{netfile.parent.name:>50s}  --> {len(energies)}")
                for i, energy in enumerate(energies):
                    i += 1
                    if np.isnan(energy):
                        continue

                    shape_info[f"Int_{i}"].append(energy)

        return shape_info


if __name__ == "__main__":

    path = Path("/Volumes/AlvinSSD/Crystal_Systems/Xemium/form1")
    sol_path = Path(
        "/Users/alvin/GitHub/crystalopt/src/crystalopt/solvent/solvent.json"
    )
    sol_screen = SolventScreen(path, solvent_dict=sol_path)
