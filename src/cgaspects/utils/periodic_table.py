# CPK-style colors (R, G, B) in [0, 1] range, van der Waals radii, covalent radii,
# and average atomic masses.
# vdW radii (Å): Bondi (1964) / Alvarez (2013).
# Covalent radii (Å) and masses: occ/element.h (Alvarez 2008 / IUPAC values).

PERIODIC_TABLE: dict[str, dict] = {
    "H":  {"color": (0.90, 0.90, 0.90), "radius": 1.20, "cov_radius": 0.23, "mass":   1.00794},
    "He": {"color": (0.85, 1.00, 1.00), "radius": 1.40, "cov_radius": 1.50, "mass":   4.002602},
    "Li": {"color": (0.80, 0.50, 1.00), "radius": 1.82, "cov_radius": 1.28, "mass":   6.941},
    "Be": {"color": (0.76, 1.00, 0.00), "radius": 1.53, "cov_radius": 0.96, "mass":   9.012182},
    "B":  {"color": (1.00, 0.71, 0.71), "radius": 1.92, "cov_radius": 0.83, "mass":  10.811},
    "C":  {"color": (0.30, 0.30, 0.30), "radius": 1.70, "cov_radius": 0.68, "mass":  12.0107},
    "N":  {"color": (0.18, 0.31, 0.97), "radius": 1.55, "cov_radius": 0.68, "mass":  14.0067},
    "O":  {"color": (1.00, 0.05, 0.05), "radius": 1.52, "cov_radius": 0.68, "mass":  15.9994},
    "F":  {"color": (0.56, 0.88, 0.31), "radius": 1.47, "cov_radius": 0.64, "mass":  18.998403},
    "Ne": {"color": (0.70, 0.89, 0.96), "radius": 1.54, "cov_radius": 1.50, "mass":  20.1797},
    "Na": {"color": (0.67, 0.36, 0.95), "radius": 2.27, "cov_radius": 1.66, "mass":  22.98977},
    "Mg": {"color": (0.54, 1.00, 0.00), "radius": 1.73, "cov_radius": 1.41, "mass":  24.305},
    "Al": {"color": (0.75, 0.65, 0.65), "radius": 1.84, "cov_radius": 1.21, "mass":  26.981538},
    "Si": {"color": (0.94, 0.78, 0.63), "radius": 2.10, "cov_radius": 1.20, "mass":  28.0855},
    "P":  {"color": (1.00, 0.50, 0.00), "radius": 1.80, "cov_radius": 1.05, "mass":  30.973761},
    "S":  {"color": (1.00, 1.00, 0.19), "radius": 1.80, "cov_radius": 1.02, "mass":  32.065},
    "Cl": {"color": (0.12, 0.94, 0.12), "radius": 1.75, "cov_radius": 0.99, "mass":  35.453},
    "Ar": {"color": (0.50, 0.82, 0.89), "radius": 1.88, "cov_radius": 1.51, "mass":  39.948},
    "K":  {"color": (0.56, 0.25, 0.83), "radius": 2.75, "cov_radius": 2.03, "mass":  39.0983},
    "Ca": {"color": (0.24, 1.00, 0.00), "radius": 2.31, "cov_radius": 1.76, "mass":  40.078},
    "Sc": {"color": (0.90, 0.90, 0.90), "radius": 2.16, "cov_radius": 1.70, "mass":  44.95591},
    "Ti": {"color": (0.75, 0.76, 0.78), "radius": 1.87, "cov_radius": 1.60, "mass":  47.867},
    "V":  {"color": (0.65, 0.65, 0.67), "radius": 1.79, "cov_radius": 1.53, "mass":  50.9415},
    "Cr": {"color": (0.54, 0.60, 0.78), "radius": 1.89, "cov_radius": 1.39, "mass":  51.9961},
    "Mn": {"color": (0.61, 0.48, 0.78), "radius": 1.97, "cov_radius": 1.61, "mass":  54.938049},
    "Fe": {"color": (0.88, 0.40, 0.20), "radius": 1.94, "cov_radius": 1.52, "mass":  55.845},
    "Co": {"color": (0.94, 0.56, 0.63), "radius": 1.92, "cov_radius": 1.26, "mass":  58.9332},
    "Ni": {"color": (0.31, 0.82, 0.31), "radius": 1.84, "cov_radius": 1.24, "mass":  58.6934},
    "Cu": {"color": (0.78, 0.50, 0.20), "radius": 1.86, "cov_radius": 1.32, "mass":  63.546},
    "Zn": {"color": (0.49, 0.50, 0.69), "radius": 2.10, "cov_radius": 1.22, "mass":  65.409},
    "Ga": {"color": (0.76, 0.56, 0.56), "radius": 1.87, "cov_radius": 1.22, "mass":  69.723},
    "Ge": {"color": (0.40, 0.56, 0.56), "radius": 2.11, "cov_radius": 1.17, "mass":  72.64},
    "As": {"color": (0.74, 0.50, 0.89), "radius": 1.85, "cov_radius": 1.21, "mass":  74.9216},
    "Se": {"color": (1.00, 0.63, 0.00), "radius": 1.90, "cov_radius": 1.22, "mass":  78.96},
    "Br": {"color": (0.65, 0.16, 0.16), "radius": 1.85, "cov_radius": 1.21, "mass":  79.904},
    "Kr": {"color": (0.36, 0.72, 0.82), "radius": 2.02, "cov_radius": 1.50, "mass":  83.798},
    "Rb": {"color": (0.44, 0.18, 0.69), "radius": 3.03, "cov_radius": 2.20, "mass":  85.4678},
    "Sr": {"color": (0.00, 1.00, 0.00), "radius": 2.49, "cov_radius": 1.95, "mass":  87.62},
    "I":  {"color": (0.58, 0.00, 0.58), "radius": 1.98, "cov_radius": 1.40, "mass": 126.90447},
    "Cs": {"color": (0.34, 0.09, 0.56), "radius": 3.43, "cov_radius": 2.44, "mass": 132.90545},
    "Ba": {"color": (0.00, 0.79, 0.00), "radius": 2.68, "cov_radius": 2.15, "mass": 137.327},
    "Pb": {"color": (0.34, 0.35, 0.38), "radius": 2.02, "cov_radius": 1.46, "mass": 207.2},
}

_DEFAULT = {"color": (0.70, 0.70, 0.70), "radius": 1.70, "cov_radius": 1.50, "mass": 0.0}


def get_atom_color(symbol: str) -> tuple[float, float, float]:
    return PERIODIC_TABLE.get(symbol, _DEFAULT)["color"]


def get_atom_radius(symbol: str) -> float:
    return PERIODIC_TABLE.get(symbol, _DEFAULT)["radius"]


def get_cov_radius(symbol: str) -> float:
    return PERIODIC_TABLE.get(symbol, _DEFAULT)["cov_radius"]


def get_atom_mass(symbol: str) -> float:
    return PERIODIC_TABLE.get(symbol, _DEFAULT)["mass"]
