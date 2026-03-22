"""Tests for export roundtrips (CSV, COCO format conversions)."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from addaxai.processing.export import csv_to_coco
from addaxai.schemas.validate import validate_recognition_output

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not available")
class TestExportRoundtrip:
    """Tests for exporting detection results to various formats."""

    def test_csv_to_coco_basic(self) -> None:
        """Test basic CSV to COCO conversion."""
        # Create minimal DataFrames
        detections_data = {
            "relative_path": ["image1.jpg", "image1.jpg", "image2.jpg"],
            "label": ["animal", "animal", "person"],
            "bbox_left": [10, 100, 50],
            "bbox_top": [20, 110, 60],
            "bbox_right": [90, 150, 200],
            "bbox_bottom": [80, 160, 250],
        }
        files_data = {
            "relative_path": ["image1.jpg", "image2.jpg"],
            "file_width": [300, 400],
            "file_height": [200, 500],
            "DateTimeOriginal": ["01/01/20 12:00:00", "02/01/20 14:30:00"],
        }

        detections_df = pd.DataFrame(detections_data)
        files_df = pd.DataFrame(files_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.json")
            csv_to_coco(detections_df, files_df, output_path, version="1.0")

            # Verify output file exists and is valid JSON
            assert os.path.exists(output_path)
            with open(output_path) as f:
                coco_data = json.load(f)

            # Verify COCO structure
            assert "images" in coco_data
            assert "annotations" in coco_data
            assert "categories" in coco_data
            assert "licenses" in coco_data
            assert "info" in coco_data

            # Verify counts
            assert len(coco_data["images"]) == 2
            assert len(coco_data["annotations"]) == 3
            assert len(coco_data["categories"]) == 2  # animal, person

    def test_csv_to_coco_empty_detections(self) -> None:
        """Test COCO conversion with file that has no detections."""
        detections_data = {
            "relative_path": ["image1.jpg"],
            "label": ["animal"],
            "bbox_left": [10],
            "bbox_top": [20],
            "bbox_right": [90],
            "bbox_bottom": [80],
        }
        files_data = {
            "relative_path": ["image1.jpg", "image2.jpg"],  # image2.jpg has no detections
            "file_width": [300, 400],
            "file_height": [200, 500],
            "DateTimeOriginal": ["01/01/20 12:00:00", "NA"],
        }

        detections_df = pd.DataFrame(detections_data)
        files_df = pd.DataFrame(files_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.json")
            csv_to_coco(detections_df, files_df, output_path)

            with open(output_path) as f:
                coco_data = json.load(f)

            # Both images should be in the output
            assert len(coco_data["images"]) == 2
            # Only one has annotations
            assert len(coco_data["annotations"]) == 1

    def test_csv_to_coco_multiple_categories(self) -> None:
        """Test COCO with multiple detection categories."""
        detections_data = {
            "relative_path": ["image1.jpg"] * 5,
            "label": ["animal", "person", "vehicle", "animal", "person"],
            "bbox_left": [10, 50, 100, 150, 200],
            "bbox_top": [20, 60, 110, 160, 210],
            "bbox_right": [40, 80, 150, 180, 230],
            "bbox_bottom": [50, 90, 160, 190, 240],
        }
        files_data = {
            "relative_path": ["image1.jpg"],
            "file_width": [500],
            "file_height": [500],
            "DateTimeOriginal": ["01/01/20 12:00:00"],
        }

        detections_df = pd.DataFrame(detections_data)
        files_df = pd.DataFrame(files_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.json")
            csv_to_coco(detections_df, files_df, output_path)

            with open(output_path) as f:
                coco_data = json.load(f)

            # Should have 3 unique categories
            assert len(coco_data["categories"]) == 3
            category_names = {cat["name"] for cat in coco_data["categories"]}
            assert category_names == {"animal", "person", "vehicle"}

            # All 5 annotations should be present
            assert len(coco_data["annotations"]) == 5

    def test_csv_to_coco_bbox_calculation(self) -> None:
        """Test that COCO bboxes are calculated correctly (x, y, width, height)."""
        detections_data = {
            "relative_path": ["image1.jpg"],
            "label": ["animal"],
            "bbox_left": [100],
            "bbox_top": [50],
            "bbox_right": [300],
            "bbox_bottom": [250],
        }
        files_data = {
            "relative_path": ["image1.jpg"],
            "file_width": [500],
            "file_height": [500],
            "DateTimeOriginal": ["01/01/20 12:00:00"],
        }

        detections_df = pd.DataFrame(detections_data)
        files_df = pd.DataFrame(files_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.json")
            csv_to_coco(detections_df, files_df, output_path)

            with open(output_path) as f:
                coco_data = json.load(f)

            annotation = coco_data["annotations"][0]
            bbox = annotation["bbox"]
            # COCO format is [x, y, width, height]
            assert bbox == [100, 50, 200, 200]  # width=300-100=200, height=250-50=200
            assert annotation["area"] == 40000  # 200 * 200

    def test_csv_to_coco_date_handling(self) -> None:
        """Test date conversion in COCO output."""
        detections_data = {
            "relative_path": ["image1.jpg", "image2.jpg"],
            "label": ["animal", "animal"],
            "bbox_left": [10, 20],
            "bbox_top": [20, 30],
            "bbox_right": [90, 100],
            "bbox_bottom": [80, 150],
        }
        files_data = {
            "relative_path": ["image1.jpg", "image2.jpg"],
            "file_width": [300, 400],
            "file_height": [200, 500],
            "DateTimeOriginal": ["15/03/20 14:30:45", "NA"],  # Invalid date becomes 'NA'
        }

        detections_df = pd.DataFrame(detections_data)
        files_df = pd.DataFrame(files_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.json")
            csv_to_coco(detections_df, files_df, output_path)

            with open(output_path) as f:
                coco_data = json.load(f)

            # First image should have formatted date
            assert coco_data["images"][0]["date_captured"] == "2020-03-15 14:30:45"
            # Second image should have NA
            assert coco_data["images"][1]["date_captured"] == "NA"

    def test_csv_to_coco_output_is_valid_json_schema(self) -> None:
        """Test that COCO output conforms to expected structure."""
        detections_data = {
            "relative_path": ["image1.jpg"],
            "label": ["animal"],
            "bbox_left": [10],
            "bbox_top": [20],
            "bbox_right": [90],
            "bbox_bottom": [80],
        }
        files_data = {
            "relative_path": ["image1.jpg"],
            "file_width": [300],
            "file_height": [200],
            "DateTimeOriginal": ["01/01/20 12:00:00"],
        }

        detections_df = pd.DataFrame(detections_data)
        files_df = pd.DataFrame(files_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.json")
            csv_to_coco(detections_df, files_df, output_path)

            with open(output_path) as f:
                coco_data = json.load(f)

            # Verify required COCO fields
            assert isinstance(coco_data, dict)
            assert "images" in coco_data
            assert "annotations" in coco_data
            assert "categories" in coco_data
            assert "licenses" in coco_data
            assert "info" in coco_data

            # Verify image structure
            for image in coco_data["images"]:
                assert "id" in image
                assert "width" in image
                assert "height" in image
                assert "file_name" in image

            # Verify annotation structure
            for annotation in coco_data["annotations"]:
                assert "id" in annotation
                assert "image_id" in annotation
                assert "category_id" in annotation
                assert "bbox" in annotation
                assert "area" in annotation
                assert len(annotation["bbox"]) == 4

            # Verify category structure
            for category in coco_data["categories"]:
                assert "id" in category
                assert "name" in category

    def test_csv_to_coco_version_in_info(self) -> None:
        """Test that version parameter is included in info section."""
        detections_data = {
            "relative_path": ["image1.jpg"],
            "label": ["animal"],
            "bbox_left": [10],
            "bbox_top": [20],
            "bbox_right": [90],
            "bbox_bottom": [80],
        }
        files_data = {
            "relative_path": ["image1.jpg"],
            "file_width": [300],
            "file_height": [200],
            "DateTimeOriginal": ["01/01/20 12:00:00"],
        }

        detections_df = pd.DataFrame(detections_data)
        files_df = pd.DataFrame(files_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.json")
            csv_to_coco(detections_df, files_df, output_path, version="2.5.1")

            with open(output_path) as f:
                coco_data = json.load(f)

            assert "AddaxAI (v2.5.1)" in coco_data["info"]["description"]
