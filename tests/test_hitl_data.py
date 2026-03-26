"""Tests for addaxai.hitl.data — HITL data processing functions (TDD red phase).

Tests verify:
  - verification_status reads XML verified attribute correctly
  - check_if_img_needs_converting parameterized with base_folder
  - fetch_confs_per_class parameterized with base_folder
  - update_json_from_img_list uses progress_callback instead of GUI dialog
"""

import json
import xml.etree.ElementTree as ET


# ─── verification_status ─────────────────────────────────────────────────────

class TestVerificationStatus:
    def test_importable(self):
        from addaxai.hitl.data import verification_status  # noqa: F401

    def test_verified_yes(self, tmp_path):
        from addaxai.hitl.data import verification_status
        xml_path = tmp_path / "test.xml"
        root = ET.Element("annotation", verified="yes")
        ET.SubElement(root, "filename").text = "img.jpg"
        ET.ElementTree(root).write(str(xml_path))
        assert verification_status(str(xml_path)) is True

    def test_verified_no(self, tmp_path):
        from addaxai.hitl.data import verification_status
        xml_path = tmp_path / "test.xml"
        root = ET.Element("annotation", verified="no")
        ET.ElementTree(root).write(str(xml_path))
        assert verification_status(str(xml_path)) is False

    def test_no_verified_attribute(self, tmp_path):
        from addaxai.hitl.data import verification_status
        xml_path = tmp_path / "test.xml"
        root = ET.Element("annotation")
        ET.ElementTree(root).write(str(xml_path))
        assert verification_status(str(xml_path)) is False


# ─── check_if_img_needs_converting ───────────────────────────────────────────

class TestCheckIfImgNeedsConverting:
    def test_importable(self):
        from addaxai.hitl.data import check_if_img_needs_converting  # noqa: F401

    def test_needs_converting_when_verified_not_json_updated(self, tmp_path):
        from addaxai.hitl.data import check_if_img_needs_converting
        base = tmp_path / "project"
        base.mkdir()
        temp_folder = base / "temp-folder"
        temp_folder.mkdir()
        img_path = str(base / "photo.jpg")

        # Create XML at temp-folder/photo.xml with verified=yes, no json_updated
        xml_path = temp_folder / "photo.xml"
        root = ET.Element("annotation", verified="yes")
        ET.SubElement(root, "filename").text = "photo.jpg"
        ET.ElementTree(root).write(str(xml_path))

        assert check_if_img_needs_converting(img_path, str(base)) is True

    def test_no_conversion_when_already_updated(self, tmp_path):
        from addaxai.hitl.data import check_if_img_needs_converting
        base = tmp_path / "project"
        base.mkdir()
        temp_folder = base / "temp-folder"
        temp_folder.mkdir()
        img_path = str(base / "photo.jpg")

        xml_path = temp_folder / "photo.xml"
        root = ET.Element("annotation", verified="yes", json_updated="yes")
        ET.ElementTree(root).write(str(xml_path))

        assert check_if_img_needs_converting(img_path, str(base)) is False

    def test_no_conversion_when_not_verified(self, tmp_path):
        from addaxai.hitl.data import check_if_img_needs_converting
        base = tmp_path / "project"
        base.mkdir()
        temp_folder = base / "temp-folder"
        temp_folder.mkdir()
        img_path = str(base / "photo.jpg")

        xml_path = temp_folder / "photo.xml"
        root = ET.Element("annotation")
        ET.ElementTree(root).write(str(xml_path))

        assert check_if_img_needs_converting(img_path, str(base)) is False


# ─── fetch_confs_per_class ───────────────────────────────────────────────────

class TestFetchConfsPerClass:
    def test_importable(self):
        from addaxai.hitl.data import fetch_confs_per_class  # noqa: F401

    def test_returns_confs_dict(self, tmp_path):
        from addaxai.hitl.data import fetch_confs_per_class
        base = tmp_path / "project"
        base.mkdir()

        # Create image_recognition_file.json with label map
        rec_json = {
            "detection_categories": {"1": "animal", "2": "person"},
            "images": [
                {"file": "img1.jpg", "detections": [
                    {"category": "1", "conf": 0.95},
                    {"category": "2", "conf": 0.80},
                ]},
                {"file": "img2.jpg", "detections": [
                    {"category": "1", "conf": 0.70},
                ]},
            ],
        }
        json_path = base / "image_recognition_file.json"
        json_path.write_text(json.dumps(rec_json))

        result = fetch_confs_per_class(str(json_path), str(base))
        assert "1" in result
        assert "2" in result
        assert result["1"] == [0.95, 0.70]
        assert result["2"] == [0.80]

    def test_empty_detections(self, tmp_path):
        from addaxai.hitl.data import fetch_confs_per_class
        base = tmp_path / "project"
        base.mkdir()

        rec_json = {
            "detection_categories": {"1": "animal"},
            "images": [{"file": "img1.jpg"}],
        }
        json_path = base / "image_recognition_file.json"
        json_path.write_text(json.dumps(rec_json))

        result = fetch_confs_per_class(str(json_path), str(base))
        assert result["1"] == []


# ─── update_json_from_img_list ───────────────────────────────────────────────

class TestUpdateJsonFromImgList:
    def test_importable(self):
        from addaxai.hitl.data import update_json_from_img_list  # noqa: F401

    def test_accepts_base_folder_param(self):
        """Signature includes base_folder (not a tkinter var)."""
        import inspect
        from addaxai.hitl.data import update_json_from_img_list
        params = inspect.signature(update_json_from_img_list).parameters
        assert "base_folder" in params

    def test_accepts_progress_callback(self):
        """Signature uses progress_callback (not patience_dialog)."""
        import inspect
        from addaxai.hitl.data import update_json_from_img_list
        params = inspect.signature(update_json_from_img_list).parameters
        assert "progress_callback" in params
        assert "patience_dialog" not in params
