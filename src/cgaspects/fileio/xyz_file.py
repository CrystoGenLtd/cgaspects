import numpy as np
from typing import Tuple, List, Optional, Union, Iterable, Callable
from pathlib import Path
import trimesh
import logging
from collections import namedtuple


from dataclasses import dataclass, field

LOG = logging.getLogger("CGA:xyz_file")


def parse_xyz_file(filepath: Path, progress_callback=None) -> List[Tuple[str, np.ndarray]]:
    """Convert provided CG xyz file into list of arrays by reading line by line.

    Parameters
    ----------
    filepath : str
        Path to the .XYZ file to read.
    progress_callback : function, optional
        Callback function taking (position, total) for e.g. progress tracking

    Returns
    -------
    list of tuple of :obj:`np.ndarray`
        List of (N) comment lines, along with :obj:`np.ndarray` arrays of contents
    """
    frames = []
    # current_position = 0

    # make a callback function for progress
    def callback(pos, tot):
        if progress_callback is not None:
            progress_callback(pos, tot)

    num_frames = 1

    with filepath.open() as file:
        while True:
            header = file.readline()
            # Break at the end of file
            if not header:
                break

            section_line_count = int(header.strip())

            comment = file.readline().strip()
            if progress_callback is not None:
                num_frames = int(comment.split("//")[1])

            values = np.loadtxt(file, max_rows=section_line_count)
            frames.append((comment, np.array(values)))

            callback(len(frames), num_frames)

    return frames


def read_XYZ(filepath, progress_callback=None):
    """Read in shape data and generates a np arrary.
    Supported formats:
        .XYZ
        .txt (.xyz format)
        .stl
    """
    filepath = Path(filepath)
    LOG.debug(filepath)
    xyz = None
    xyz_movie = {}

    LOG.debug("reading file from %s", filepath.name)
    suffix = filepath.suffix

    if suffix == ".XYZ":
        LOG.debug("XYZ: File read!")
        xyzs = parse_xyz_file(filepath, progress_callback=progress_callback)
        if len(xyzs) > 0:
            xyz = xyzs[0][1]
        if len(xyzs) > 1:
            xyz_movie = {i: x[1] for i, x in enumerate(xyzs)}

    elif suffix == ".txt":
        LOG.debug("xyz: File read!")
        xyz = np.loadtxt(filepath, skiprows=2)
    elif suffix == ".stl":
        LOG.debug("stl: File read!")
        xyz = trimesh.load(filepath)
    else:
        LOG.warning("Invalid suffix when reading XYZ file %s", suffix)

    return xyz, xyz_movie


ShapeMetrics = namedtuple(
    "ShapeMetrics",
    [
        "x",
        "y",
        "z",
        "pc1",
        "pc2",
        "pc3",
        "aspect1",
        "aspect2",
        "surface_area",
        "volume",
        "surface_area_to_volume_ratio",
        "shape",
    ],
)


@dataclass
class Frame:
    """Single crystal shape frame."""

    raw: np.ndarray  # raw array with labels etc.
    comment: Optional[str] = None  # optional comment line (e.g. XYZ file)

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, idx: int | slice) -> np.ndarray:
        return self.coords[idx]

    def __iter__(self) -> Iterable[np.ndarray]:
        return iter(self.coords)

    @property
    def coords(self) -> np.ndarray:
        """Return the coordinate array (Nx3)."""
        return self.raw[:, 3:6] if self.raw.shape[1] >= 6 else self.raw

    @coords.setter
    def coords(self, xyz: np.ndarray) -> None:
        """Set the coordinate array (Nx3)."""
        xyz = np.asarray(xyz, dtype=float)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"Expected Nx3 array, got {xyz.shape}")
        if self.raw.shape[1] >= 6:
            self.raw[:, 3:6] = xyz
        else:
            self.raw = xyz


@dataclass
class Frames:
    """Container for multiple frames. Behaves like a list of Frame objects."""

    _frames: list[Frame] = field(default_factory=list)

    # --- core list-like behaviour ---
    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, idx: Union[int, slice]) -> Union[Frame, "Frames"]:
        if isinstance(idx, slice):
            return Frames(self._frames[idx])
        return self._frames[idx]

    def __iter__(self) -> Iterable[Frame]:
        return iter(self._frames)

    def append(self, frame: Frame) -> None:
        self._frames.append(frame)

    def extend(self, frames: Iterable[Frame]) -> None:
        self._frames.extend(frames)

    # --- convenience views ---
    @property
    def coords(self) -> dict[int, np.ndarray]:
        """All frame coordinates as dict {index: coords}."""
        return {i: f.coords for i, f in enumerate(self._frames)}

    @property
    def raw_coords(self) -> dict[int, np.ndarray]:
        """All frame coordinates as dict {index: coords}."""
        return {i: f.raw for i, f in enumerate(self._frames)}

    @property
    def comments(self) -> dict[int, Optional[str]]:
        """All frame comments as dict {index: comment}."""
        return {i: f.comment for i, f in enumerate(self._frames)}

    def get_coords(self, idx: int) -> Optional[np.ndarray]:
        """Convenience: coords for a single frame."""
        if -len(self._frames) <= idx < len(self._frames):
            return self._frames[idx].coords
        return None

    def get_raw_coords(self, idx: int) -> Optional[np.ndarray]:
        """Convenience: coords for a single frame."""
        if -len(self._frames) <= idx < len(self._frames):
            return self._frames[idx].raw
        return None


@dataclass
class CrystalCloud:
    """Base class for handling crystal point cloud data from various file formats."""

    filepath: Path
    frames: Frames = field(default_factory=Frames)
    xyz: Optional[np.ndarray] = None

    # ---- Core container behaviour ----
    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> Frame:
        return self.frames[idx]

    def __iter__(self):
        return iter(self.frames)

    # ---- Convenience views ----
    @property
    def movie(self) -> dict[int, np.ndarray]:
        """Return all frames as dict {index: coords}."""
        return self.frames.coords

    @property
    def empty(self) -> bool:
        """Return True if the crystal has no point data."""
        return self.xyz is None or self.xyz.size == 0

    @property
    def coords(self) -> Optional[np.ndarray]:
        """Return the coordinates of the last frame (index -1)."""
        return self.frames.get_coords(-1)

    # ---- Parsing helpers ----
    @staticmethod
    def parse_xyz_file(
        filepath: Path,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        clean: bool = True,
    ) -> Frames:
        """Parse multi-frame XYZ into Frames container."""
        frames = Frames()

        with filepath.open("r", encoding="utf-8") as file:
            frame_idx = 0
            while True:
                header = file.readline()
                if not header:
                    break

                try:
                    n_atoms = int(header.strip())
                except ValueError as e:
                    raise ValueError(f"Invalid XYZ header at frame {frame_idx}: {e}")

                comment = file.readline().strip()

                if n_atoms == 0:
                    raw = np.empty((0, 0), dtype=float)
                else:
                    try:
                        raw = np.loadtxt(file, max_rows=n_atoms, dtype=float, ndmin=2)
                    except ValueError as e:
                        if clean:
                            raw_text = Path(filepath).read_text(encoding="utf-8").replace("*", "0")
                            Path(filepath).write_text(raw_text, encoding="utf-8")
                            return CrystalCloud.parse_xyz_file(filepath, progress_callback, clean=False)
                        raise e

                frames.append(Frame(raw=raw, comment=comment))
                frame_idx += 1

                if progress_callback:
                    try:
                        total_frames = int(comment.split("//")[1])
                    except Exception:
                        total_frames = frame_idx
                    progress_callback(frame_idx, total_frames)

        return frames

    @staticmethod
    def normalise_verts(verts, center=True):
        if verts.size == 0:
            return verts
        if center:
            verts = verts - np.mean(verts, axis=0)
        norm = np.linalg.norm(verts, axis=1).max()
        if norm == 0:
            return verts
        verts /= norm
        return verts

    # ---- Loader ----
    @classmethod
    def from_file(
        cls,
        filepath: Path,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        normalise=True,
    ) -> "CrystalCloud":
        """Factory method to create CrystalShape from .XYZ, .txt, .stl, .glb."""
        filepath = Path(filepath)

        if filepath.suffix == ".XYZ":
            frames = cls.parse_xyz_file(filepath, progress_callback)
            xyz = frames.get_coords(0)

        elif filepath.suffix == ".txt":
            arr = np.genfromtxt(filepath, skip_header=2, dtype=float, invalid_raise=False)
            # Filter only numeric columns (drop any all-NaN columns)
            arr = arr[:, ~np.isnan(arr).all(axis=0)]
            # Use last 3 numeric columns as coordinates if more than 3 exist
            coords = arr[:, -3:] if arr.shape[1] >= 3 else arr
            frames = Frames([Frame(raw=coords, comment="txt-file")])
            print(frames)
            xyz = coords

        elif filepath.suffix in {".stl", ".glb"}:
            mesh = trimesh.load(filepath)
            coords = mesh.vertices
            frames = Frames([Frame(raw=coords, comment="mesh-file")])
            xyz = coords

        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}.")

        if xyz is not None and normalise:
            xyz = cls.normalise_verts(xyz)

        return cls(filepath=filepath, frames=frames, xyz=xyz)

    def get_raw_frame_coords(self, frame_idx: int = 0) -> Optional[np.ndarray]:
        """Get coordinates for a specific frame."""
        return self.frames.get_raw_coords(frame_idx)

    def get_frame_coords(self, frame_idx: int = 0) -> Optional[np.ndarray]:
        """Get coordinates for a specific frame."""
        return self.frames.get_coords(frame_idx)

    def get_all_frame_coords(self) -> dict[int, np.ndarray]:
        """Get coordinates for all frames."""
        return self.frames.coords

    def get_all_raw_frame_coords(self) -> dict[int, np.ndarray]:
        """Get coordinates for all frames."""
        return self.frames.raw_coords


@dataclass
class DockingData:
    """Docking site data from a CrystalGrower ``*_docking.XYZ`` file.

    Column layout (0-indexed in the raw array):
      0 : replicate index
      1 : molecule type
      2 : coordination shell  (30 = central, 31 = 1st shell, 32 = 2nd shell)
      3 : x coordinate
      4 : y coordinate
      5 : z coordinate
      6 : site number
    """

    filepath: Path
    raw: np.ndarray  # shape (N, 7)

    # Shell IDs
    SHELL_CENTRAL = 30
    SHELL_FIRST = 31
    SHELL_SECOND = 32

    # Default RGB colours per shell (values in [0, 1])
    SHELL_COLORS: dict = field(
        default_factory=lambda: {
            30: (1.0, 0.35, 0.0),   # orange-red: central molecule
            31: (0.0, 0.85, 0.2),   # green: 1st coordination shell
            32: (0.15, 0.45, 1.0),  # blue: 2nd coordination shell
        }
    )

    @property
    def coords(self) -> np.ndarray:
        return self.raw[:, 3:6]

    @property
    def shells(self) -> np.ndarray:
        return self.raw[:, 2].astype(int)

    @property
    def mol_types(self) -> np.ndarray:
        return self.raw[:, 1].astype(int)

    @property
    def site_numbers(self) -> np.ndarray:
        return self.raw[:, 6].astype(int)

    @property
    def empty(self) -> bool:
        return self.raw is None or self.raw.size == 0

    @classmethod
    def from_file(cls, filepath: Path) -> "DockingData":
        """Parse a CrystalGrower ``*_docking.XYZ`` file."""
        filepath = Path(filepath)
        with filepath.open("r", encoding="utf-8") as f:
            n_atoms = int(f.readline().strip())
            f.readline()  # skip "docking file" comment line
            raw = np.loadtxt(f, max_rows=n_atoms, dtype=float)
        if raw.ndim == 1:
            raw = raw[np.newaxis, :]
        return cls(filepath=filepath, raw=raw)

    def colored_points(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(coords, colors)`` arrays for rendering, coloured by shell.

        Both arrays are float32.  ``colors`` has shape (N, 3) with RGB in [0, 1].
        """
        coords = self.coords.astype(np.float32)
        colors = np.zeros((len(coords), 3), dtype=np.float32)
        for shell_id, color in self.SHELL_COLORS.items():
            mask = self.shells == shell_id
            colors[mask] = color
        return coords, colors
