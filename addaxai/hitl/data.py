"""HITL data processing functions — no GUI dependencies.

Pure data operations for the Human-in-the-Loop verification workflow:
reading/writing XML annotations, checking verification status,
extracting confidence values, and updating recognition JSON files.
"""

import json
import logging
import os
import xml.etree.ElementTree as ET
from typing import Callable, Dict, List, Optional

from addaxai.processing.annotations import (
    convert_xml_to_coco,
    indent_xml,
    return_xml_path,
)
from addaxai.utils.json_ops import check_json_paths, fetch_label_map_from_json

logger = logging.getLogger(__name__)


def verification_status(xml_path: str) -> bool:
    """Check if an XML annotation has been verified by the user.

    Args:
        xml_path: Path to the Pascal VOC XML annotation file.

    Returns:
        True if the annotation's ``verified`` attribute is ``"yes"``.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    try:
        return root.attrib['verified'] == 'yes'
    except (KeyError, AttributeError):
        return False


def check_if_img_needs_converting(img_file: str, base_folder: str) -> bool:
    """Check whether an image's annotation needs converting to JSON.

    An annotation needs converting when it has been verified by the user
    but the JSON recognition file has not yet been updated.

    Args:
        img_file:    Absolute path to the image file.
        base_folder: Project base folder (for locating the temp-folder XML).

    Returns:
        True if the annotation is verified but not yet updated in JSON.
    """
    root = ET.parse(return_xml_path(img_file, base_folder)).getroot()

    try:
        is_verified = root.attrib['verified'] == 'yes'
    except (KeyError, AttributeError):
        is_verified = False

    try:
        is_json_updated = root.attrib['json_updated'] == 'yes'
    except (KeyError, AttributeError):
        is_json_updated = False

    return is_verified and not is_json_updated


def fetch_confs_per_class(
    json_fpath: str,
    base_folder: str,
) -> Dict[str, List[float]]:
    """Read confidence values per detection category from a recognition JSON.

    Args:
        json_fpath:  Path to the recognition JSON file to read detections from.
        base_folder: Project base folder (for locating ``image_recognition_file.json``
                     to read the label map).

    Returns:
        Dict mapping category ID strings to lists of confidence floats.
    """
    label_map = fetch_label_map_from_json(
        os.path.join(base_folder, 'image_recognition_file.json'))
    confs: Dict[str, List[float]] = {key: [] for key in label_map}
    with open(json_fpath) as content:
        data = json.load(content)
        for image in data['images']:
            if 'detections' in image:
                for detection in image['detections']:
                    confs[detection["category"]].append(detection["conf"])
    return confs


def update_json_from_img_list(
    verified_images: List[str],
    inverted_label_map: Dict[str, str],
    recognition_file: str,
    base_folder: str,
    progress_callback: Optional[Callable[[int], None]] = None,
    current: int = 0,
) -> int:
    """Update recognition JSON with verified annotations from XML files.

    For each image in *verified_images*, reads the corresponding XML annotation,
    converts it to COCO format, and writes it back into the recognition JSON.

    Args:
        verified_images:   List of absolute image paths that need JSON update.
        inverted_label_map: Dict mapping class names to category IDs.
        recognition_file:  Path to the recognition JSON to update.
        base_folder:       Project base folder (for locating temp-folder XMLs
                           and resolving relative JSON paths).
        progress_callback: Optional callback called with the current counter
                           after each image is processed.
        current:           Starting value for the progress counter.

    Returns:
        The final value of *current* after all images have been processed.
    """
    # check if the json has relative paths
    json_paths_are_relative = (
        check_json_paths(recognition_file, base_folder) == "relative"
    )

    # read
    with open(recognition_file, "r") as f:
        data = json.load(f)

    # adjust
    for image in data['images']:
        image_path = image['file']
        if json_paths_are_relative:
            image_path = os.path.normpath(
                os.path.join(os.path.dirname(recognition_file), image_path))
        if image_path in verified_images:
            # update progress
            if progress_callback is not None:
                progress_callback(current)
            current += 1

            # read annotation and convert
            xml = return_xml_path(image_path, base_folder)
            coco, ver_status, new_class, inverted_label_map = convert_xml_to_coco(
                xml, inverted_label_map)
            image['manually_checked'] = ver_status
            if new_class:
                data['detection_categories'] = {
                    v: k for k, v in inverted_label_map.items()}
            if ver_status:
                image['detections'] = coco['detections']

                # mark xml as json-updated
                tree = ET.parse(xml)
                root = tree.getroot()
                root.set('json_updated', 'yes')
                indent_xml(root)
                tree.write(xml)

    # write
    logger.debug("Writing recognition file: %s", recognition_file)
    with open(recognition_file, "w") as json_file:
        json.dump(data, json_file, indent=1)

    return current
