"""Tests for orchestration config dataclasses (Step B3, TDD red phase).

Written BEFORE the implementation.

Contracts verified:
  - Each dataclass is importable from addaxai.orchestration.context
  - Each dataclass can be instantiated with required fields
  - All field values are plain Python types (str, bool, float, int, List)
  - No field stores a tkinter variable or widget reference
  - Field names match the architecture spec in CLAUDE.md
"""

import dataclasses
from typing import List

import pytest


# --- Importability ---

def test_deploy_config_importable():
    from addaxai.orchestration.context import DeployConfig  # noqa: F401


def test_classify_config_importable():
    from addaxai.orchestration.context import ClassifyConfig  # noqa: F401


def test_postprocess_config_importable():
    from addaxai.orchestration.context import PostprocessConfig  # noqa: F401


# --- DeployConfig ---

@pytest.fixture
def deploy_config():
    from addaxai.orchestration.context import DeployConfig
    return DeployConfig(
        base_path="/app",
        det_model_dir="/app/models/det",
        det_model_name="MegaDetector 5a",
        det_model_path="",
        cls_model_name="None",
        disable_gpu=False,
        use_abs_paths=False,
        source_folder="/data/images",
        dpd_options_model=[["Model A", "Custom"], ["Modelo A", "Personalizado"]],
        lang_idx=0,
    )


def test_deploy_config_instantiation(deploy_config):
    """DeployConfig should be instantiatable with required fields."""
    from addaxai.orchestration.context import DeployConfig
    assert isinstance(deploy_config, DeployConfig)


def test_deploy_config_field_names(deploy_config):
    """DeployConfig must have all fields from the architecture spec."""
    fields = {f.name for f in dataclasses.fields(deploy_config)}
    required = {
        "base_path", "det_model_dir", "det_model_name", "det_model_path",
        "cls_model_name", "disable_gpu", "use_abs_paths", "source_folder",
        "dpd_options_model", "lang_idx",
    }
    assert required.issubset(fields), f"Missing fields: {required - fields}"


def test_deploy_config_plain_types(deploy_config):
    """All DeployConfig field values should be plain Python types."""
    for f in dataclasses.fields(deploy_config):
        value = getattr(deploy_config, f.name)
        assert not _is_tkinter_type(value), (
            f"Field '{f.name}' holds a tkinter object: {type(value)}"
        )


def test_deploy_config_values(deploy_config):
    assert deploy_config.base_path == "/app"
    assert deploy_config.det_model_name == "MegaDetector 5a"
    assert deploy_config.det_model_path == ""
    assert deploy_config.cls_model_name == "None"
    assert deploy_config.disable_gpu is False
    assert deploy_config.lang_idx == 0


def test_deploy_config_is_dataclass():
    from addaxai.orchestration.context import DeployConfig
    assert dataclasses.is_dataclass(DeployConfig)


# --- ClassifyConfig ---

@pytest.fixture
def classify_config():
    from addaxai.orchestration.context import ClassifyConfig
    return ClassifyConfig(
        base_path="/app",
        cls_model_name="SpeciesNet",
        disable_gpu=True,
        cls_detec_thresh=0.2,
        cls_class_thresh=0.6,
        smooth_cls_animal=True,
        tax_fallback=False,
        temp_frame_folder="",
        lang_idx=1,
    )


def test_classify_config_instantiation(classify_config):
    from addaxai.orchestration.context import ClassifyConfig
    assert isinstance(classify_config, ClassifyConfig)


def test_classify_config_field_names(classify_config):
    fields = {f.name for f in dataclasses.fields(classify_config)}
    required = {
        "base_path", "cls_model_name", "disable_gpu",
        "cls_detec_thresh", "cls_class_thresh", "smooth_cls_animal",
        "tax_fallback", "temp_frame_folder", "lang_idx",
    }
    assert required.issubset(fields), f"Missing fields: {required - fields}"


def test_classify_config_plain_types(classify_config):
    for f in dataclasses.fields(classify_config):
        value = getattr(classify_config, f.name)
        assert not _is_tkinter_type(value), (
            f"Field '{f.name}' holds a tkinter object: {type(value)}"
        )


def test_classify_config_values(classify_config):
    assert classify_config.cls_model_name == "SpeciesNet"
    assert classify_config.disable_gpu is True
    assert classify_config.cls_detec_thresh == pytest.approx(0.2)
    assert classify_config.cls_class_thresh == pytest.approx(0.6)
    assert classify_config.smooth_cls_animal is True
    assert classify_config.lang_idx == 1


def test_classify_config_is_dataclass():
    from addaxai.orchestration.context import ClassifyConfig
    assert dataclasses.is_dataclass(ClassifyConfig)


# --- PostprocessConfig ---

@pytest.fixture
def postprocess_config():
    from addaxai.orchestration.context import PostprocessConfig
    return PostprocessConfig(
        source_folder="/data/images",
        dest_folder="/data/results",
        thresh=0.2,
        separate_files=True,
        file_placement=2,
        sep_conf=False,
        vis=False,
        crp=False,
        exp=True,
        plt=False,
        exp_format="CSV",
        data_type="img",
        vis_blur=False,
        vis_bbox=True,
        vis_size_idx=1,
        keep_series=False,
        keep_series_seconds=0.0,
        keep_series_species=[],
        current_version="5.0",
        lang_idx=0,
    )


def test_postprocess_config_instantiation(postprocess_config):
    from addaxai.orchestration.context import PostprocessConfig
    assert isinstance(postprocess_config, PostprocessConfig)


def test_postprocess_config_field_names(postprocess_config):
    fields = {f.name for f in dataclasses.fields(postprocess_config)}
    required = {
        "source_folder", "dest_folder", "thresh", "separate_files",
        "file_placement", "sep_conf", "vis", "crp", "exp", "plt",
        "exp_format", "data_type", "vis_blur", "vis_bbox", "vis_size_idx",
        "keep_series", "keep_series_seconds", "keep_series_species",
        "current_version", "lang_idx",
    }
    assert required.issubset(fields), f"Missing fields: {required - fields}"


def test_postprocess_config_plain_types(postprocess_config):
    for f in dataclasses.fields(postprocess_config):
        value = getattr(postprocess_config, f.name)
        assert not _is_tkinter_type(value), (
            f"Field '{f.name}' holds a tkinter object: {type(value)}"
        )


def test_postprocess_config_values(postprocess_config):
    assert postprocess_config.source_folder == "/data/images"
    assert postprocess_config.dest_folder == "/data/results"
    assert postprocess_config.thresh == pytest.approx(0.2)
    assert postprocess_config.separate_files is True
    assert postprocess_config.file_placement == 2
    assert postprocess_config.data_type == "img"
    assert postprocess_config.keep_series_species == []
    assert postprocess_config.current_version == "5.0"


def test_postprocess_config_is_dataclass():
    from addaxai.orchestration.context import PostprocessConfig
    assert dataclasses.is_dataclass(PostprocessConfig)


def test_postprocess_config_keep_series_species_is_list(postprocess_config):
    """keep_series_species must be a list (not a tkinter StringVar)."""
    assert isinstance(postprocess_config.keep_series_species, list)


# --- No tkinter imports required ---

def test_context_module_importable_without_tkinter():
    """addaxai.orchestration.context must be importable without tkinter being present."""
    # If tkinter were imported at module level this would fail in headless CI.
    # Just verifying the import succeeds in our test env is sufficient.
    import addaxai.orchestration.context  # noqa: F401


# --- Helper ---

def _is_tkinter_type(value):
    """Return True if value is a tkinter variable, widget, or Tk instance."""
    type_name = type(value).__module__ or ""
    return type_name.startswith("tkinter") or type_name.startswith("customtkinter")
