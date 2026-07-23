import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from pathlib import Path
import tempfile
import os

from cgaspects.fileio.xyz_file import CrystalCloud, Frame, Frames
from cgaspects.analysis.shape_analysis import ShapeAnalyser


class TestCrystalCloud(unittest.TestCase):
    """Test suite for CrystalCloud class - file I/O and data container."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_frame_initialization(self):
        """Test Frame dataclass initialization."""
        coords = np.array([[1, 2, 3], [4, 5, 6]])
        frame = Frame(raw=coords, comment="test frame")

        self.assertEqual(len(frame), 2)
        np.testing.assert_array_equal(frame.coords, coords)
        self.assertEqual(frame.comment, "test frame")

    def test_frames_container(self):
        """Test Frames container behaves like a list."""
        frames = Frames()
        frame1 = Frame(raw=np.array([[1, 2, 3]]))
        frame2 = Frame(raw=np.array([[4, 5, 6]]))

        frames.append(frame1)
        frames.append(frame2)

        self.assertEqual(len(frames), 2)
        self.assertEqual(frames[0], frame1)
        self.assertEqual(frames[1], frame2)

    def test_from_file_txt(self):
        """Test loading from .txt file."""
        txt_path = Path(self.temp_dir) / "test.txt"
        test_data = "2\ntest comment\nC 1.0 2.0 3.0\nC 4.0 5.0 6.0"
        txt_path.write_text(test_data)

        crystal = CrystalCloud.from_file(txt_path, normalise=False)

        self.assertEqual(len(crystal.frames), 1)
        self.assertIsNotNone(crystal.coords)
        self.assertEqual(crystal.coords.shape, (2, 3))

    def test_from_file_xyz_single_frame(self):
        """Test loading single frame XYZ file."""
        xyz_path = Path(self.temp_dir) / "test.XYZ"
        xyz_content = (
            "3\nFrame 0\n1.0 0.0 0.0 1.0 2.0 3.0\n2.0 0.0 0.0 4.0 5.0 6.0\n3.0 0.0 0.0 7.0 8.0 9.0"
        )
        xyz_path.write_text(xyz_content)

        crystal = CrystalCloud.from_file(xyz_path, normalise=False)

        self.assertEqual(len(crystal.frames), 1)
        self.assertEqual(crystal.coords.shape, (3, 3))
        np.testing.assert_array_equal(
            crystal.coords, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        )

    def test_from_file_xyz_multi_frame(self):
        """Test loading multi-frame XYZ file."""
        xyz_path = Path(self.temp_dir) / "test_multi.XYZ"
        xyz_content = (
            "2\nFrame 0 // 2\n1.0 0.0 0.0 1.0 2.0 3.0\n2.0 0.0 0.0 4.0 5.0 6.0\n"
            "2\nFrame 1 // 2\n1.0 0.0 0.0 7.0 8.0 9.0\n2.0 0.0 0.0 10.0 11.0 12.0"
        )
        xyz_path.write_text(xyz_content)

        crystal = CrystalCloud.from_file(xyz_path, normalise=False)
        print(crystal)

        self.assertEqual(len(crystal.frames), 2)
        self.assertEqual(len(crystal.movie), 2)

        # Test accessing different frames
        frame0_coords = crystal.get_frame_coords(0)
        frame1_coords = crystal.get_frame_coords(1)

        np.testing.assert_array_equal(frame0_coords, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        np.testing.assert_array_equal(frame1_coords, [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])

    def test_normalise_verts(self):
        """Test vertex normalization."""
        verts = np.array([[1, 2, 3], [-1, -2, -3], [4, 5, 6]])
        normalized_verts = CrystalCloud.normalise_verts(verts, center=True)

        # Check if vertices are centered around origin
        self.assertAlmostEqual(np.mean(normalized_verts), 0, places=10)

        # Check if maximum norm is 1
        self.assertAlmostEqual(np.max(np.linalg.norm(normalized_verts, axis=1)), 1.0, places=10)

    def test_normalise_verts_no_center(self):
        """Test vertex normalization without centering."""
        verts = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        normalized_verts = CrystalCloud.normalise_verts(verts, center=False)

        # Maximum norm should be 1
        self.assertAlmostEqual(np.max(np.linalg.norm(normalized_verts, axis=1)), 1.0, places=10)

    def test_crystal_cloud_iterator(self):
        """Test CrystalCloud iteration over frames."""
        xyz_path = Path(self.temp_dir) / "test_iter.XYZ"
        xyz_content = (
            "2\nFrame 0\n1.0 0.0 0.0 1.0 2.0 3.0\n2.0 0.0 0.0 4.0 5.0 6.0\n"
            "2\nFrame 1\n1.0 0.0 0.0 7.0 8.0 9.0\n2.0 0.0 0.0 10.0 11.0 12.0"
        )
        xyz_path.write_text(xyz_content)

        crystal = CrystalCloud.from_file(xyz_path, normalise=False)

        frame_count = 0
        for frame in crystal:
            self.assertIsInstance(frame, Frame)
            frame_count += 1

        self.assertEqual(frame_count, 2)

    def test_get_all_frame_coords(self):
        """Test retrieving all frame coordinates."""
        xyz_path = Path(self.temp_dir) / "test_all.XYZ"
        xyz_content = (
            "2\nFrame 0\n1.0 0.0 0.0 1.0 2.0 3.0\n2.0 0.0 0.0 4.0 5.0 6.0\n"
            "2\nFrame 1\n1.0 0.0 0.0 7.0 8.0 9.0\n2.0 0.0 0.0 10.0 11.0 12.0"
        )
        xyz_path.write_text(xyz_content)

        crystal = CrystalCloud.from_file(xyz_path, normalise=False)
        all_coords = crystal.get_all_frame_coords()

        self.assertEqual(len(all_coords), 2)
        self.assertIn(0, all_coords)
        self.assertIn(1, all_coords)


class TestShapeAnalyser(unittest.TestCase):
    """Test suite for ShapeAnalyser class - shape analysis functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyser = ShapeAnalyser(l_max=10)
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test ShapeAnalyser initialization."""
        self.assertEqual(self.analyser.l_max, 10)
        self.assertEqual(self.analyser.zingg_method, "svd")
        self.assertEqual(len(self.analyser.frame_metrics), 0)

    def test_get_shape_class(self):
        """Test shape classification based on aspect ratios."""
        self.assertEqual(ShapeAnalyser.get_shape_class(0.5, 0.5), "Lath")
        self.assertEqual(ShapeAnalyser.get_shape_class(0.5, 0.8), "Plate")
        self.assertEqual(ShapeAnalyser.get_shape_class(0.8, 0.8), "Block")
        self.assertEqual(ShapeAnalyser.get_shape_class(0.8, 0.5), "Needle")

    def test_get_sa_vol_ratio(self):
        """Test surface area to volume ratio calculation."""
        # Create a unit cube
        xyz = np.array(
            [
                [0, 0, 0],
                [3, 0, 0],
                [0, 3, 0],
                [0, 0, 3],
                [3, 3, 0],
                [0, 3, 3],
                [3, 0, 3],
                [3, 3, 3],
            ]
        )

        sa_vol_ratio = self.analyser.get_sa_vol_ratio(xyz)

        expected_sa = 54.0
        expected_vol = 27.0
        expected_ratio = expected_sa / expected_vol

        np.testing.assert_almost_equal(sa_vol_ratio[0], expected_sa, decimal=5)
        np.testing.assert_almost_equal(sa_vol_ratio[1], expected_vol, decimal=5)
        np.testing.assert_almost_equal(sa_vol_ratio[2], expected_ratio, decimal=5)

    def test_shape_info_cube(self):
        """A 2x2x2 cube classifies as a Block with unit aspect ratios."""
        # Create a simple cube
        xyz = np.array([
                [0, 0, 0], [2, 0, 0], [0, 2, 0], [2, 2, 0],
                [0, 0, 2], [2, 0, 2], [0, 2, 2], [2, 2, 2],
            ]
        )

        shape_metrics = self.analyser.shape_info(xyz, get_sa_vol=False)

        # A perfect cube has three equal PCA eigenvalues, so the principal axes
        # are degenerate and the reported extents are along an arbitrary rotated
        # basis (2*sqrt(2) on two axes). Aspect ratios and shape stay stable.
        self.assertAlmostEqual(shape_metrics.x, 2.0 * np.sqrt(2), places=5)
        self.assertAlmostEqual(shape_metrics.y, 2.0, places=5)
        self.assertAlmostEqual(shape_metrics.z, 2.0 * np.sqrt(2), places=5)
        self.assertAlmostEqual(shape_metrics.aspect1, 1.0, places=1)
        self.assertAlmostEqual(shape_metrics.aspect2, 1.0, places=1)
        self.assertEqual(shape_metrics.shape, "Block")

    def test_shape_info_plate(self):
        """A 2x2x1 slab classifies as a Plate."""
        # Create a simple plate
        xyz = np.array([
                [0, 0, 0], [2, 0, 0], [0, 2, 0], [2, 2, 0],
                [0, 0, 1], [2, 0, 1], [0, 2, 1], [2, 2, 1],
            ]
        )

        shape_metrics = self.analyser.shape_info(xyz, get_sa_vol=False)

        self.assertAlmostEqual(shape_metrics.x, 2.0, places=5)
        self.assertAlmostEqual(shape_metrics.y, 2.0, places=5)
        self.assertAlmostEqual(shape_metrics.z, 1.0, places=5)
        self.assertAlmostEqual(shape_metrics.aspect1, 0.5, places=1)
        self.assertAlmostEqual(shape_metrics.aspect2, 1.0, places=1)
        self.assertEqual(shape_metrics.shape, "Plate")

    def test_shape_info_needle(self):
        """A 2x2x5 rod classifies as a Needle."""
        # Create a simple needle
        xyz = np.array([
                [0, 0, 0], [2, 0, 0], [0, 2, 0], [2, 2, 0],
                [0, 0, 5], [2, 0, 5], [0, 2, 5], [2, 2, 5],
            ]
        )

        shape_metrics = self.analyser.shape_info(xyz, get_sa_vol=False)

        self.assertAlmostEqual(shape_metrics.x, 5.0, places=5)
        self.assertAlmostEqual(shape_metrics.y, 2.0, places=5)
        self.assertAlmostEqual(shape_metrics.z, 2.0, places=5)
        self.assertAlmostEqual(shape_metrics.aspect1, 1.0, places=1)
        self.assertAlmostEqual(shape_metrics.aspect2, 0.4, places=1)
        self.assertEqual(shape_metrics.shape, "Needle")

    def test_shape_info_lath(self):
        """A 2x5x1 elongated slab classifies as a Lath."""
        # Create a simple lath
        xyz = np.array([
                [0, 0, 0], [2, 0, 0], [0, 5, 0], [2, 5, 0],
                [0, 0, 1], [2, 0, 1], [0, 5, 1], [2, 5, 1],
            ]
        )

        shape_metrics = self.analyser.shape_info(xyz, get_sa_vol=False)

        self.assertAlmostEqual(shape_metrics.x, 5.0, places=5)
        self.assertAlmostEqual(shape_metrics.y, 2.0, places=5)
        self.assertAlmostEqual(shape_metrics.z, 1.0, places=5)
        self.assertAlmostEqual(shape_metrics.aspect1, 0.5, places=1)
        self.assertAlmostEqual(shape_metrics.aspect2, 0.4, places=1)
        self.assertEqual(shape_metrics.shape, "Lath")

    def test_shape_info_with_sa_vol(self):
        """Test shape information with surface area and volume."""
        xyz = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]
        )

        shape_metrics = self.analyser.shape_info(xyz, get_sa_vol=True)

        self.assertIsNotNone(shape_metrics.surface_area)
        self.assertIsNotNone(shape_metrics.volume)
        self.assertIsNotNone(shape_metrics.surface_area_to_volume_ratio)

    def test_analyse_crystal_single_frame(self):
        """Test analysing a single frame of a crystal."""
        xyz_path = Path(self.temp_dir) / "test.XYZ"
        xyz_content = "4\nTest crystal\n1.0 0.0 0.0 0.0 0.0 0.0\n2.0 0.0 0.0 2.0 0.0 0.0\n3.0 0.0 0.0 0.0 2.0 0.0\n4.0 0.0 0.0 0.0 0.0 2.0"
        xyz_path.write_text(xyz_content)

        crystal = CrystalCloud.from_file(xyz_path, normalise=False)
        self.analyser.analyse_crystal(crystal, frame_idx=0)

        self.assertIn(0, self.analyser.frame_metrics)
        metrics = self.analyser.get_frame_metrics(0)
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.shape, "Plate")

    def test_analyse_crystal_all_frames(self):
        """Test analysing all frames of a crystal."""
        xyz_path = Path(self.temp_dir) / "test_multi.XYZ"
        xyz_content = (
            "4\nFrame 0\n1.0 0.0 0.0 0.0 0.0 0.0\n2.0 0.0 0.0 2.0 0.0 0.0\n3.0 0.0 0.0 0.0 2.0 0.0\n4.0 0.0 0.0 0.0 0.0 2.0\n"
            "4\nFrame 1\n1.0 0.0 0.0 0.0 0.0 0.0\n2.0 0.0 0.0 3.0 0.0 0.0\n3.0 0.0 0.0 0.0 3.0 0.0\n4.0 0.0 0.0 0.0 0.0 3.0"
        )
        xyz_path.write_text(xyz_content)

        crystal = CrystalCloud.from_file(xyz_path, normalise=False)
        self.analyser.analyse_crystal(crystal)

        all_metrics = self.analyser.get_all_frame_metrics()
        self.assertEqual(len(all_metrics), 2)
        self.assertIn(0, all_metrics)
        self.assertIn(1, all_metrics)

    def test_zingg_method_bounding_box(self):
        """Test Zingg analysis using bounding box method."""
        analyser_bbox = ShapeAnalyser(l_max=10, zingg_method="bounding_box")
        xyz = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]])

        shape_metrics = analyser_bbox.shape_info(xyz, get_sa_vol=False)

        self.assertIsNotNone(shape_metrics.aspect1)
        self.assertIsNotNone(shape_metrics.aspect2)

    def test_invalid_zingg_method(self):
        """Test that invalid Zingg method raises ValueError."""
        analyser = ShapeAnalyser(l_max=10, zingg_method="invalid")
        xyz = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]])

        with self.assertRaises(ValueError):
            analyser.shape_info(xyz)

    def test_get_frame_metrics_nonexistent(self):
        """Test retrieving metrics for non-existent frame."""
        metrics = self.analyser.get_frame_metrics(999)
        self.assertIsNone(metrics)


class TestIntegration(unittest.TestCase):
    """Integration tests for CrystalCloud and ShapeAnalyser."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_full_workflow(self):
        """Test complete workflow from file loading to analysis."""
        # Create test XYZ file
        xyz_path = Path(self.temp_dir) / "test_workflow.XYZ"
        xyz_content = (
            "14\nCube crystal\n"
            "1.0 0.0 0.0 0.0 0.0 0.0\n"
            "2.0 0.0 0.0 1.0 0.0 0.0\n"
            "3.0 0.0 0.0 0.0 1.0 0.0\n"
            "4.0 0.0 0.0 0.0 0.0 1.0\n"
            "5.0 0.0 0.0 1.0 1.0 0.0\n"
            "6.0 0.0 0.0 0.0 1.0 1.0\n"
            "7.0 0.0 0.0 1.0 0.0 1.0\n"
            "8.0 0.0 0.0 1.0 1.0 1.0\n"
            "9.0 0.0 0.0 0.5 0.5 0.0\n"
            "10.0 0.0 0.0 0.5 0.5 1.0\n"
            "11.0 0.0 0.0 0.5 0.0 0.5\n"
            "12.0 0.0 0.0 0.5 1.0 0.5\n"
            "13.0 0.0 0.0 0.0 0.5 0.5\n"
            "14.0 0.0 0.0 1.0 0.5 0.5"
        )
        xyz_path.write_text(xyz_content)

        # Load crystal
        crystal = CrystalCloud.from_file(xyz_path, normalise=False)

        # Analyse crystal
        analyser = ShapeAnalyser(l_max=10, zingg_method="bounding_box")
        analyser.analyse_crystal(crystal, frame_idx=0)

        # Check results
        metrics = analyser.get_frame_metrics(0)
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.shape, "Block")
        self.assertIsNotNone(metrics.volume)
        self.assertIsNotNone(metrics.surface_area)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
