"""Helpers for deriving identifiers from CrystoGen output file names."""

import re
from pathlib import Path

# Suffixes appended to file stems by the toolchain. Ordered longest-first so the
# more specific tag is stripped before its shorter substring (e.g.
# ``_ovito_CGVisualiser`` before ``_CGVisualiser``).
_KNOWN_SUFFIXES = ("_ovito_CGVisualiser", "_CGVisualiser", "_CGAspects")


def simulation_id_from_path(path) -> str:
    """Derive the simulation identifier from an XYZ/size file path.

    Any known toolchain suffix (``_CGAspects``, ``_ovito_CGVisualiser``, ...) is
    stripped first. If what remains ends in ``_<number>`` that number is the
    simulation id; otherwise the remainder is returned as-is (e.g. a solvent
    name such as ``m-cresol`` from ``m-cresol_CGAspects.XYZ``).

    Examples
    --------
    ``run_3_CGAspects``              -> ``"3"``
    ``run_3_ovito_CGVisualiser``    -> ``"3"``
    ``run_5``                       -> ``"5"``
    ``m-cresol_CGAspects``          -> ``"m-cresol"``
    ``1-decanol_CGAspects``         -> ``"1-decanol"``
    ``water_ovito_CGVisualiser``    -> ``"water"``

    This is the single source of truth shared across analysis workflows
    (aspect-ratio and growth-rate dataframe construction) and plot-click
    resolution, so the value stored in a "Simulation Number" column always
    matches the value derived later from the same file.
    """
    stem = Path(path).stem
    lowered = stem.lower()
    for suffix in _KNOWN_SUFFIXES:
        # match case-insensitively: the visualiser tag appears as both
        # "_ovito_CGVisualiser" and "_ovito_CGvisualiser" in the wild
        if lowered.endswith(suffix.lower()):
            stem = stem[: -len(suffix)]
            break
    match = re.search(r"_(\d+)$", stem)
    return match.group(1) if match else stem
