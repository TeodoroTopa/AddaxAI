"""Integration tests for postprocessing pipeline (move_files, file operations)."""

import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest

from addaxai.processing.postprocess import move_files


class TestMoveFiles:
    """Tests for move_files() function."""

    def test_move_files_basic_move(self) -> None:
        """Test basic file move to category directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup source and destination
            src_dir = os.path.join(tmpdir, "source")
            dst_dir = os.path.join(tmpdir, "dest")
            os.makedirs(src_dir)
            os.makedirs(dst_dir)

            # Create a test file
            test_file = os.path.join(src_dir, "test.jpg")
            Path(test_file).write_text("test content")

            # Move file
            result = move_files(
                file="test.jpg",
                detection_type="animal",
                file_placement=1,  # move
                max_detection_conf=0.9,
                sep_conf=False,
                dst_root=dst_dir,
                src_dir=src_dir,
                manually_checked=False,
            )

            # Verify result (normalize path separators for cross-platform)
            assert result.replace("\\", "/") == "animal/test.jpg"
            assert not os.path.exists(test_file)  # Original moved
            assert os.path.exists(os.path.join(dst_dir, "animal", "test.jpg"))

    def test_move_files_copy_instead_of_move(self) -> None:
        """Test file copy (file_placement=2) preserves original."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = os.path.join(tmpdir, "source")
            dst_dir = os.path.join(tmpdir, "dest")
            os.makedirs(src_dir)
            os.makedirs(dst_dir)

            test_file = os.path.join(src_dir, "test.jpg")
            Path(test_file).write_text("test content")

            result = move_files(
                file="test.jpg",
                detection_type="person",
                file_placement=2,  # copy
                max_detection_conf=0.85,
                sep_conf=False,
                dst_root=dst_dir,
                src_dir=src_dir,
                manually_checked=False,
            )

            # Verify result
            assert result.replace("\\", "/") == "person/test.jpg"
            assert os.path.exists(test_file)  # Original still exists
            assert os.path.exists(os.path.join(dst_dir, "person", "test.jpg"))

    def test_move_files_with_confidence_separation(self) -> None:
        """Test file separation by confidence bucket."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = os.path.join(tmpdir, "source")
            dst_dir = os.path.join(tmpdir, "dest")
            os.makedirs(src_dir)
            os.makedirs(dst_dir)

            test_file = os.path.join(src_dir, "test.jpg")
            Path(test_file).write_text("test content")

            result = move_files(
                file="test.jpg",
                detection_type="animal",
                file_placement=2,  # copy to preserve
                max_detection_conf=0.75,
                sep_conf=True,
                dst_root=dst_dir,
                src_dir=src_dir,
                manually_checked=False,
            )

            # Confidence 0.75 maps to "conf_0.7-0.8"
            assert result.replace("\\", "/") == "animal/conf_0.7-0.8/test.jpg"
            assert os.path.exists(os.path.join(dst_dir, "animal", "conf_0.7-0.8", "test.jpg"))

    def test_move_files_verified_creates_verified_dir(self) -> None:
        """Test manually_checked=True creates 'verified' subdirectory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = os.path.join(tmpdir, "source")
            dst_dir = os.path.join(tmpdir, "dest")
            os.makedirs(src_dir)
            os.makedirs(dst_dir)

            test_file = os.path.join(src_dir, "test.jpg")
            Path(test_file).write_text("test content")

            result = move_files(
                file="test.jpg",
                detection_type="animal",
                file_placement=2,  # copy
                max_detection_conf=0.9,
                sep_conf=True,
                dst_root=dst_dir,
                src_dir=src_dir,
                manually_checked=True,
            )

            # Verified files go to "verified" subdirectory
            assert result.replace("\\", "/") == "animal/verified/test.jpg"
            assert os.path.exists(os.path.join(dst_dir, "animal", "verified", "test.jpg"))

    def test_move_files_empty_no_confidence_separation(self) -> None:
        """Test empty detection skips confidence separation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = os.path.join(tmpdir, "source")
            dst_dir = os.path.join(tmpdir, "dest")
            os.makedirs(src_dir)
            os.makedirs(dst_dir)

            test_file = os.path.join(src_dir, "test.jpg")
            Path(test_file).write_text("test content")

            result = move_files(
                file="test.jpg",
                detection_type="empty",
                file_placement=2,
                max_detection_conf=0.0,
                sep_conf=True,  # Even with sep_conf=True
                dst_root=dst_dir,
                src_dir=src_dir,
                manually_checked=False,
            )

            # Empty bypasses confidence separation
            assert result.replace("\\", "/") == "empty/test.jpg"
            assert os.path.exists(os.path.join(dst_dir, "empty", "test.jpg"))

    def test_move_files_nested_path(self) -> None:
        """Test file with nested directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = os.path.join(tmpdir, "source")
            dst_dir = os.path.join(tmpdir, "dest")
            os.makedirs(src_dir)
            os.makedirs(dst_dir)

            # Create nested source structure
            nested = os.path.join(src_dir, "subfolder", "nested")
            os.makedirs(nested)
            test_file = os.path.join(nested, "test.jpg")
            Path(test_file).write_text("test content")

            result = move_files(
                file="subfolder/nested/test.jpg",
                detection_type="animal",
                file_placement=2,  # copy
                max_detection_conf=0.9,
                sep_conf=False,
                dst_root=dst_dir,
                src_dir=src_dir,
                manually_checked=False,
            )

            assert result.replace("\\", "/") == "animal/subfolder/nested/test.jpg"
            assert os.path.exists(os.path.join(dst_dir, "animal", "subfolder", "nested", "test.jpg"))

    def test_move_files_all_confidence_buckets(self) -> None:
        """Test confidence separation for all bucket ranges."""
        confidence_levels = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
        expected_dirs = [
            "conf_0.0-0.1", "conf_0.1-0.2", "conf_0.2-0.3", "conf_0.3-0.4",
            "conf_0.4-0.5", "conf_0.5-0.6", "conf_0.6-0.7", "conf_0.7-0.8",
            "conf_0.8-0.9", "conf_0.9-1.0",
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = os.path.join(tmpdir, "source")
            dst_dir = os.path.join(tmpdir, "dest")
            os.makedirs(src_dir)
            os.makedirs(dst_dir)

            for conf, expected_dir in zip(confidence_levels, expected_dirs):
                test_file = os.path.join(src_dir, f"test_{conf}.jpg")
                Path(test_file).write_text("test content")

                result = move_files(
                    file=f"test_{conf}.jpg",
                    detection_type="animal",
                    file_placement=2,
                    max_detection_conf=conf,
                    sep_conf=True,
                    dst_root=dst_dir,
                    src_dir=src_dir,
                    manually_checked=False,
                )

                assert result.replace("\\", "/") == f"animal/{expected_dir}/test_{conf}.jpg"
                assert os.path.exists(
                    os.path.join(dst_dir, "animal", expected_dir, f"test_{conf}.jpg")
                )


class TestFileOperationsWithFixtures:
    """Tests using actual fixture images."""

    def test_move_files_with_fixture_images(self) -> None:
        """Test move_files with actual fixture images."""
        fixtures_dir = os.path.join(os.path.dirname(__file__), "fixtures")
        images_dir = os.path.join(fixtures_dir, "images")

        # Skip if images don't exist
        if not os.path.isdir(images_dir):
            pytest.skip("Fixture images not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            dst_dir = os.path.join(tmpdir, "dest")
            os.makedirs(dst_dir)

            # Move fixture images
            result = move_files(
                file="test_animal.jpg",
                detection_type="animal",
                file_placement=2,  # copy to preserve
                max_detection_conf=0.92,
                sep_conf=False,
                dst_root=dst_dir,
                src_dir=images_dir,
                manually_checked=False,
            )

            assert result.replace("\\", "/") == "animal/test_animal.jpg"
            dest_file = os.path.join(dst_dir, result)
            assert os.path.exists(dest_file)
            # Verify it's a valid image file
            assert os.path.getsize(dest_file) > 0

    def test_move_multiple_fixture_images(self) -> None:
        """Test moving multiple fixture images of different types."""
        fixtures_dir = os.path.join(os.path.dirname(__file__), "fixtures")
        images_dir = os.path.join(fixtures_dir, "images")

        if not os.path.isdir(images_dir):
            pytest.skip("Fixture images not available")

        moves = [
            ("test_animal.jpg", "animal", 0.92),
            ("test_person.jpg", "person", 0.88),
            ("test_vehicle.jpg", "vehicle", 0.79),
            ("test_empty.jpg", "empty", 0.0),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            dst_dir = os.path.join(tmpdir, "dest")
            os.makedirs(dst_dir)

            for filename, detection_type, conf in moves:
                result = move_files(
                    file=filename,
                    detection_type=detection_type,
                    file_placement=2,
                    max_detection_conf=conf,
                    sep_conf=False,
                    dst_root=dst_dir,
                    src_dir=images_dir,
                    manually_checked=False,
                )

                expected = f"{detection_type}/{filename}"
                assert result.replace("\\", "/") == expected
                assert os.path.exists(os.path.join(dst_dir, result))
