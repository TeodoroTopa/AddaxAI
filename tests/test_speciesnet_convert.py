"""Tests for SpeciesNet JSON conversion — pure data transformation (TDD).

Tests verify:
  - reclassify_speciesnet_detections merges label maps correctly
  - reclassify_speciesnet_detections reclassifies animal detections above threshold
  - reclassify_speciesnet_detections preserves non-animal detections
  - reclassify_speciesnet_detections preserves detections below threshold
"""


# ─── reclassify_speciesnet_detections ────────────────────────────────────────

class TestReclassifySpeciesnetDetections:
    def test_importable(self):
        from addaxai.processing.speciesnet import reclassify_speciesnet_detections  # noqa: F401

    def test_reclassifies_animal_above_threshold(self):
        from addaxai.processing.speciesnet import reclassify_speciesnet_detections
        data = {
            "detection_categories": {"1": "animal", "2": "person"},
            "classification_categories": {"0": "deer", "1": "bear"},
            "images": [{
                "file": "img.jpg",
                "detections": [{
                    "category": "1",
                    "conf": 0.9,
                    "classifications": [["0", 0.85]],
                }],
            }],
        }
        result = reclassify_speciesnet_detections(data, cls_detec_thresh=0.5)
        det = result["images"][0]["detections"][0]
        assert det["conf"] == 0.85
        assert det["prev_conf"] == 0.9
        assert det["prev_category"] == "1"
        # deer should be in the detection categories
        assert "deer" in result["detection_categories"].values()

    def test_preserves_below_threshold(self):
        from addaxai.processing.speciesnet import reclassify_speciesnet_detections
        data = {
            "detection_categories": {"1": "animal", "2": "person"},
            "classification_categories": {"0": "deer"},
            "images": [{
                "file": "img.jpg",
                "detections": [{
                    "category": "1",
                    "conf": 0.3,
                    "classifications": [["0", 0.85]],
                }],
            }],
        }
        result = reclassify_speciesnet_detections(data, cls_detec_thresh=0.5)
        det = result["images"][0]["detections"][0]
        assert det["conf"] == 0.3
        assert det["category"] == "1"
        assert "prev_conf" not in det

    def test_preserves_non_animal(self):
        from addaxai.processing.speciesnet import reclassify_speciesnet_detections
        data = {
            "detection_categories": {"1": "animal", "2": "person"},
            "classification_categories": {"0": "deer"},
            "images": [{
                "file": "img.jpg",
                "detections": [{
                    "category": "2",
                    "conf": 0.9,
                }],
            }],
        }
        result = reclassify_speciesnet_detections(data, cls_detec_thresh=0.5)
        det = result["images"][0]["detections"][0]
        assert det["category"] == "2"
        assert det["conf"] == 0.9

    def test_saves_original_detection_categories(self):
        from addaxai.processing.speciesnet import reclassify_speciesnet_detections
        data = {
            "detection_categories": {"1": "animal"},
            "classification_categories": {"0": "deer"},
            "images": [{
                "file": "img.jpg",
                "detections": [{
                    "category": "1",
                    "conf": 0.9,
                    "classifications": [["0", 0.85]],
                }],
            }],
        }
        result = reclassify_speciesnet_detections(data, cls_detec_thresh=0.5)
        assert "detection_categories_original" in result
        assert result["detection_categories_original"] == {"1": "animal"}

    def test_no_classifications_key(self):
        from addaxai.processing.speciesnet import reclassify_speciesnet_detections
        data = {
            "detection_categories": {"1": "animal"},
            "classification_categories": {"0": "deer"},
            "images": [{
                "file": "img.jpg",
                "detections": [{"category": "1", "conf": 0.9}],
            }],
        }
        result = reclassify_speciesnet_detections(data, cls_detec_thresh=0.5)
        det = result["images"][0]["detections"][0]
        assert det["category"] == "1"
        assert det["conf"] == 0.9
