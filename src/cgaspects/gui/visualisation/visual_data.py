from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


@dataclass
class VisualData:
    """Unified centroid data for all render modes in the 3D viewport.

    All centroid positions are stored in Cartesian Ångströms, centred at the
    world origin so that camera orbit always circles the displayed object.

    Use the class-method constructors (from_xyz, from_docking, from_checkpoint)
    rather than instantiating directly.
    """

    centroids: np.ndarray  # (N, 3) float32 – Cartesian Å, centred
    mol_types: np.ndarray  # (N,) int

    source: Literal["xyz", "docking", "checkpoint"] = "xyz"

    # mol_type → precomputed template dict (from _precompute_mol_templates)
    # {"cart": (M,3), "centroid": (3,), "colors": (M,3),
    #  "radii": (M,), "bonds": [(i,j)], "symbols": [str]}
    templates: dict[int, dict] | None = None

    # Raw per-source backing array (N, K); column layout depends on source:
    #   xyz        – the full original xyz input array; columns by native position
    #   docking    – col 0: shells
    #   checkpoint – col 0: tile_indices
    _raw: np.ndarray | None = field(default=None, repr=False)

    # ------------------------------------------------------------------ metadata

    @property
    def mol_numbers(self) -> np.ndarray | None:
        if self.source == "xyz" and self._raw is not None and self._raw.shape[1] > 1:
            return self._raw[:, 1]
        return None

    @property
    def layers(self) -> np.ndarray | None:
        if self.source == "xyz" and self._raw is not None and self._raw.shape[1] > 2:
            return self._raw[:, 2]
        return None

    @property
    def site_numbers(self) -> np.ndarray | None:
        if self.source == "xyz" and self._raw is not None and self._raw.shape[1] > 6:
            return self._raw[:, 6]
        return None

    @property
    def energies(self) -> np.ndarray | None:
        if self.source == "xyz" and self._raw is not None and self._raw.shape[1] > 7:
            return self._raw[:, 7]
        return None

    @property
    def shells(self) -> np.ndarray | None:
        if self.source == "docking" and self._raw is not None:
            return self._raw[:, 0]
        return None

    @property
    def tile_indices(self) -> np.ndarray | None:
        if self.source == "checkpoint" and self._raw is not None:
            return self._raw[:, 0]
        return None

    # ------------------------------------------------------------------ template builder

    @staticmethod
    def _build_cart_templates(mol_templates: dict, crystallography) -> dict[int, dict]:
        """Convert fractional MolTemplate coords to Cartesian and compute per-type GPU data."""
        from ...utils.periodic_table import get_atom_color, get_atom_radius

        result: dict[int, dict] = {}
        for mol_type, tmpl in mol_templates.items():
            if not tmpl.atoms:
                continue
            frac = np.array([a.frac for a in tmpl.atoms], dtype=np.float64)
            cart = crystallography.frac_to_cart(frac).astype(np.float32)
            centroid = cart.mean(axis=0)
            colors = np.array([get_atom_color(a.symbol) for a in tmpl.atoms], dtype=np.float32)
            radii = np.array([get_atom_radius(a.symbol) for a in tmpl.atoms], dtype=np.float32)
            result[mol_type] = {
                "cart": cart,
                "centroid": centroid,
                "colors": colors,
                "radii": radii,
                "bonds": tmpl.bonds,
                "symbols": [a.symbol for a in tmpl.atoms],
            }

        return result

    # ------------------------------------------------------------------ counts

    @property
    def n_centroids(self) -> int:
        return len(self.centroids)

    @property
    def n_atoms(self) -> int | None:
        """Total expanded atom count, or None when no templates are loaded."""
        if not self.templates:
            return None
        count = sum(
            len(self.templates[mt]["cart"]) for mt in self.mol_types if mt in self.templates
        )
        return count or None

    def display_count(self) -> tuple[int | None, str]:
        """(count, label) for the crystal-info panel.

        Returns atom count + "Atoms" when templates are available,
        otherwise centroid count + "Points".
        """
        atoms = self.n_atoms
        if atoms:
            return atoms, "Atoms"
        n = self.n_centroids
        return (n if n else None), "Points"

    # ------------------------------------------------------------------ constructors

    @classmethod
    def from_xyz(
        cls,
        xyz: np.ndarray,
        mol_templates: dict | None = None,
        crystallography=None,
    ) -> "VisualData":
        """Build from the standard (N, 7+) XYZ centroid array.

        If crystallography is provided its a-axis scales the centroid columns
        from XYZ units to Cartesian Å.
        """
        templates = (
            cls._build_cart_templates(mol_templates, crystallography)
            if mol_templates and crystallography
            else None
        )
        centroids = xyz[:, 3:6].astype(np.float64)
        if crystallography is not None and crystallography.cell is not None:
            a = float(crystallography.cell.a)
            if a > 1e-10 and a != 1.0:
                centroids *= a
        return cls(
            centroids=centroids.astype(np.float32),
            mol_types=xyz[:, 0].astype(int),
            source="xyz",
            templates=templates,
            _raw=xyz,
        )

    @classmethod
    def from_docking(
        cls,
        docking_data,
        mol_templates: dict | None = None,
        crystallography=None,
    ) -> "VisualData":
        """Build from DockingData.

        Docking coords are stored in XYZ units; the a-axis from crystallography
        converts them to Cartesian Å.
        """
        templates = (
            cls._build_cart_templates(mol_templates, crystallography)
            if mol_templates and crystallography
            else None
        )
        a = float(crystallography.cell.a) if crystallography and crystallography.cell else 1.0
        coords = (docking_data.coords.astype(np.float64) * a).astype(np.float32)
        return cls(
            centroids=coords,
            mol_types=docking_data.mol_types.copy(),
            source="docking",
            templates=templates,
            _raw=docking_data.shells.copy().reshape(-1, 1),
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint,
        crystallography,
        mol_templates: dict | None = None,
    ) -> "VisualData":
        """Build from a Checkpoint grid by expanding occupied cells to Cartesian Å."""
        templates = (
            cls._build_cart_templates(mol_templates, crystallography)
            if mol_templates and crystallography
            else None
        )
        centroid_chunks: list[np.ndarray] = []
        mol_type_chunks: list[np.ndarray] = []
        tile_chunks: list[np.ndarray] = []

        for t in range(checkpoint.n_tiles):
            indices = np.argwhere(checkpoint.data[..., t]).astype(float)
            if not len(indices):
                continue
            cart = crystallography.frac_to_cart(indices).astype(np.float32)
            mol_type = t + 1
            centroid_chunks.append(cart)
            mol_type_chunks.append(np.full(len(cart), mol_type, dtype=int))
            tile_chunks.append(np.full(len(cart), t, dtype=int))

        if not centroid_chunks:
            return cls(
                centroids=np.zeros((0, 3), dtype=np.float32),
                mol_types=np.zeros(0, dtype=int),
                source="checkpoint",
                templates=templates,
            )

        centroids = np.vstack(centroid_chunks)
        center = centroids.mean(axis=0)
        return cls(
            centroids=(centroids - center).astype(np.float32),
            mol_types=np.concatenate(mol_type_chunks),
            source="checkpoint",
            templates=templates,
            _raw=np.concatenate(tile_chunks).reshape(-1, 1),
        )

    # ------------------------------------------------------------------ colour getters

    def colors_uniform(self, rgb) -> np.ndarray:
        """(N, 3) float32 – solid colour for every centroid."""
        return np.tile(np.asarray(rgb, dtype=np.float32), (self.n_centroids, 1))

    def colors_by_z(self) -> np.ndarray:
        """(N, 3) float32 – blue-to-red gradient along Z (checkpoint layer view)."""
        z = self.centroids[:, 2]
        t = (z - z.min()) / max(float(z.max() - z.min()), 1e-9)
        zeros = np.zeros(self.n_centroids, dtype=np.float32)
        return np.stack([t.astype(np.float32), zeros, (1.0 - t).astype(np.float32)], axis=1)

    def colors_by_shell(
        self,
        shell_map: dict[int, tuple],
        overrides: dict[int, tuple] | None = None,
    ) -> np.ndarray:
        """(N, 3) float32 – colours for docking data, keyed by coordination shell."""
        colors = np.zeros((self.n_centroids, 3), dtype=np.float32)
        effective = {**shell_map, **(overrides or {})}
        if self.shells is not None:
            for sid, rgb in effective.items():
                colors[self.shells == sid] = rgb
        return colors

    def colors_by_array(
        self,
        values: np.ndarray,
        colormap_fn,
        min_val: float | None = None,
        max_val: float | None = None,
    ) -> np.ndarray:
        """(N, 3) float32 – colormap applied to an arbitrary 1-D values array."""
        v = values.astype(float)
        lo = min_val if min_val is not None else float(v.min())
        hi = max_val if max_val is not None else float(v.max())
        t = (v - lo) / max(hi - lo, 1e-9)
        return colormap_fn(t.astype(np.float32))[:, :3].astype(np.float32)

    def colors_by_tile(self, palette: np.ndarray) -> np.ndarray:
        """(N, 3) float32 – per-tile palette colours (checkpoint atoms view)."""
        colors = np.zeros((self.n_centroids, 3), dtype=np.float32)
        if self.tile_indices is not None:
            for t_idx in np.unique(self.tile_indices):
                colors[self.tile_indices == t_idx] = palette[t_idx % len(palette)]
        return colors

    # ------------------------------------------------------------------ vertex builders

    def sphere_vertices(
        self,
        colors: np.ndarray,
        selected_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """(N, 7) float32 vertex array [x,y,z, r,g,b, sel] for sphere/point rendering."""
        n = self.n_centroids
        sel = (
            selected_mask.astype(np.float32).reshape(n, 1)
            if selected_mask is not None
            else np.zeros((n, 1), dtype=np.float32)
        )
        return np.concatenate([self.centroids, colors, sel], axis=1).astype(np.float32)

    def atom_vertices(
        self,
        centroid_colors: np.ndarray | None,
        use_atom_colors: bool = False,
        color_overrides: dict[str, tuple] | None = None,
        radius_overrides: dict[str, float] | None = None,
        bond_radius: float = 0.1,
        selected_indices: set | None = None,
        slice_planes: list | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Build GPU vertex arrays for atom and bond rendering.

        Returns
        -------
        atom_arr : (M, 8) float32 – [x,y,z, r,g,b, sel, vdw_radius]
        bond_arr : (B, 10) float32 – [x0..z0, x1..z1, r,g,b, cyl_radius] or None
        """
        if not self.templates:
            return np.zeros((0, 8), dtype=np.float32), None

        # Build slice keep-mask once (True = keep centroid).
        if slice_planes:
            slice_keep = np.array(
                [not _centroid_clipped(c.astype(np.float64), slice_planes) for c in self.centroids],
                dtype=bool,
            )
        else:
            slice_keep = None

        global_indices = np.arange(self.n_centroids)
        atom_chunks: list[np.ndarray] = []
        bond_chunks: list[np.ndarray] = []

        for mol_type, tmpl in self.templates.items():
            type_mask = self.mol_types == mol_type
            keep = type_mask & slice_keep if slice_keep is not None else type_mask
            if not keep.any():
                continue

            K = int(keep.sum())
            A = len(tmpl["cart"])
            type_global_idx = global_indices[keep]
            centroids_k = self.centroids[keep]  # (K, 3)
            offsets = centroids_k - tmpl["centroid"]  # (K, 3)
            # atom positions: (K, A, 3) → (K*A, 3)
            atom_pos_3d = tmpl["cart"][np.newaxis, :, :] + offsets[:, np.newaxis, :]
            atom_pos_3d = atom_pos_3d.astype(np.float32)  # (K, A, 3)
            atom_pos = atom_pos_3d.reshape(K * A, 3)  # (K*A, 3)

            if use_atom_colors:
                base_colors, radii_arr = _resolve_overrides(tmpl, color_overrides, radius_overrides)
                # broadcast template colors across all K molecules
                colors_3d = (
                    np.broadcast_to(base_colors[np.newaxis, :, :], (K, A, 3))
                    .copy()
                    .astype(np.float32)
                )
            else:
                cc = centroid_colors[keep]  # (K, 3)
                colors_3d = (
                    np.broadcast_to(cc[:, np.newaxis, :], (K, A, 3)).copy().astype(np.float32)
                )
                radii_arr = tmpl["radii"].copy()
                if radius_overrides:
                    for j, sym in enumerate(tmpl["symbols"]):
                        if sym in radius_overrides:
                            radii_arr[j] = radius_overrides[sym]

            radii_arr = np.asarray(radii_arr, dtype=np.float32)
            colors_flat = colors_3d.reshape(K * A, 3)  # (K*A, 3)
            radii_ka = np.tile(radii_arr, K).reshape(K * A, 1)  # (K*A, 1)

            if selected_indices:
                sel_k = np.isin(type_global_idx, list(selected_indices)).astype(np.float32)
            else:
                sel_k = np.zeros(K, dtype=np.float32)
            sel_ka = np.repeat(sel_k, A).reshape(K * A, 1)  # (K*A, 1)

            atom_chunks.append(np.hstack([atom_pos, colors_flat, sel_ka, radii_ka]))

            for a1, a2 in tmpl["bonds"]:
                if a1 >= A or a2 >= A:
                    continue
                p1 = atom_pos_3d[:, a1, :]  # (K, 3)
                p2 = atom_pos_3d[:, a2, :]  # (K, 3)
                mid = (p1 + p2) * 0.5  # (K, 3)
                c1 = colors_3d[:, a1, :]  # (K, 3)
                c2 = colors_3d[:, a2, :]  # (K, 3)
                br = np.full((K, 1), bond_radius, dtype=np.float32)
                bond_chunks.append(np.hstack([p1, mid, c1, br]))
                bond_chunks.append(np.hstack([mid, p2, c2, br]))

        if not atom_chunks:
            return np.zeros((0, 8), dtype=np.float32), None

        atom_arr = np.vstack(atom_chunks).astype(np.float32)
        bond_arr = np.vstack(bond_chunks).astype(np.float32) if bond_chunks else None
        return atom_arr, bond_arr


# ------------------------------------------------------------------ module helpers


def _centroid_clipped(pos: np.ndarray, slice_planes: list) -> bool:
    """Return True if the centroid position should be excluded by any slice plane."""
    for normal, origin, two_sided, thickness in slice_planes:
        d = float(np.dot(pos - origin, normal))
        if two_sided and abs(d) > thickness / 2.0:
            return True
        if not two_sided and d < -thickness:
            return True
    return False


def _resolve_overrides(
    tmpl: dict,
    color_overrides: dict | None,
    radius_overrides: dict | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (colors, radii) for a template with user overrides applied."""
    colors = tmpl["colors"].copy()
    radii = tmpl["radii"].copy()
    if color_overrides or radius_overrides:
        for j, sym in enumerate(tmpl["symbols"]):
            if color_overrides and sym in color_overrides:
                colors[j] = color_overrides[sym]
            if radius_overrides and sym in radius_overrides:
                radii[j] = radius_overrides[sym]
    return colors, radii
