"""SpeciesNet output conversion — no GUI dependencies.

Pure data transformation for converting SpeciesNet classification results
into the AddaxAI detection format.
"""

from typing import Any, Dict


def reclassify_speciesnet_detections(
    data: Dict[str, Any],
    cls_detec_thresh: float,
) -> Dict[str, Any]:
    """Reclassify SpeciesNet detections into AddaxAI detection format.

    Merges classification categories into the detection label map and
    replaces animal detections (above *cls_detec_thresh*) with their
    highest classification result.

    Args:
        data:              Recognition JSON data with ``detection_categories``,
                           ``classification_categories``, and ``images``.
        cls_detec_thresh:  Minimum confidence for reclassification.

    Returns:
        The mutated *data* dict with updated detection categories and
        reclassified detections.
    """
    cls_label_map = data['classification_categories']
    det_label_map = data['detection_categories']
    inverted_cls_label_map = {v: k for k, v in cls_label_map.items()}
    inverted_det_label_map = {v: k for k, v in det_label_map.items()}

    # merge classification classes into detection label map
    for k in inverted_cls_label_map:
        if k not in inverted_det_label_map:
            inverted_det_label_map[k] = str(len(inverted_det_label_map) + 1)

    # reclassify animal detections above threshold
    for image in data['images']:
        if 'detections' in image and image['detections'] is not None:
            for detection in image['detections']:
                category_id = detection['category']
                category_conf = detection['conf']
                if (category_conf >= cls_detec_thresh
                        and det_label_map.get(category_id) == "animal"
                        and 'classifications' in detection):
                    highest = detection['classifications'][0]
                    class_idx = highest[0]
                    class_name = cls_label_map[class_idx]
                    detec_idx = inverted_det_label_map[class_name]
                    detection['prev_conf'] = detection['conf']
                    detection['prev_category'] = detection['category']
                    detection['conf'] = highest[1]
                    detection['category'] = str(detec_idx)

    # update detection categories
    data['detection_categories_original'] = data['detection_categories']
    data['detection_categories'] = {v: k for k, v in inverted_det_label_map.items()}

    return data
