# CPK-style colors (R, G, B) in [0, 1] range and van der Waals radii in Angstroms.
# Radii from Bondi (1964) / Alvarez (2013). Colors follow common molecular-graphics conventions.

PERIODIC_TABLE: dict[str, dict] = {
    "H":  {"color": (0.90, 0.90, 0.90), "radius": 1.20},
    "He": {"color": (0.85, 1.00, 1.00), "radius": 1.40},
    "Li": {"color": (0.80, 0.50, 1.00), "radius": 1.82},
    "Be": {"color": (0.76, 1.00, 0.00), "radius": 1.53},
    "B":  {"color": (1.00, 0.71, 0.71), "radius": 1.92},
    "C":  {"color": (0.30, 0.30, 0.30), "radius": 1.70},
    "N":  {"color": (0.18, 0.31, 0.97), "radius": 1.55},
    "O":  {"color": (1.00, 0.05, 0.05), "radius": 1.52},
    "F":  {"color": (0.56, 0.88, 0.31), "radius": 1.47},
    "Ne": {"color": (0.70, 0.89, 0.96), "radius": 1.54},
    "Na": {"color": (0.67, 0.36, 0.95), "radius": 2.27},
    "Mg": {"color": (0.54, 1.00, 0.00), "radius": 1.73},
    "Al": {"color": (0.75, 0.65, 0.65), "radius": 1.84},
    "Si": {"color": (0.94, 0.78, 0.63), "radius": 2.10},
    "P":  {"color": (1.00, 0.50, 0.00), "radius": 1.80},
    "S":  {"color": (1.00, 1.00, 0.19), "radius": 1.80},
    "Cl": {"color": (0.12, 0.94, 0.12), "radius": 1.75},
    "Ar": {"color": (0.50, 0.82, 0.89), "radius": 1.88},
    "K":  {"color": (0.56, 0.25, 0.83), "radius": 2.75},
    "Ca": {"color": (0.24, 1.00, 0.00), "radius": 2.31},
    "Fe": {"color": (0.88, 0.40, 0.20), "radius": 2.04},
    "Co": {"color": (0.94, 0.56, 0.63), "radius": 2.00},
    "Ni": {"color": (0.31, 0.82, 0.31), "radius": 1.97},
    "Cu": {"color": (0.78, 0.50, 0.20), "radius": 1.96},
    "Zn": {"color": (0.49, 0.50, 0.69), "radius": 2.01},
    "Br": {"color": (0.65, 0.16, 0.16), "radius": 1.85},
    "I":  {"color": (0.58, 0.00, 0.58), "radius": 1.98},
    "Pb": {"color": (0.34, 0.35, 0.38), "radius": 2.02},
}

_DEFAULT = {"color": (0.70, 0.70, 0.70), "radius": 1.70}


def get_atom_color(symbol: str) -> tuple[float, float, float]:
    return PERIODIC_TABLE.get(symbol, _DEFAULT)["color"]


def get_atom_radius(symbol: str) -> float:
    return PERIODIC_TABLE.get(symbol, _DEFAULT)["radius"]
