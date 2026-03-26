"""Tests for deploy helper functions extracted from start_deploy() (TDD).

Tests verify:
  - scan_media_presence detects images and videos in folders
  - scan_media_presence respects exclude_subs flag
  - build_deploy_options builds correct option lists for img and vid
  - scan_special_characters finds and catalogues special chars in paths
"""


# ─── scan_media_presence ─────────────────────────────────────────────────────

class TestScanMediaPresence:
    def test_importable(self):
        from addaxai.orchestration.deploy_helpers import scan_media_presence  # noqa: F401

    def test_finds_images(self, tmp_path):
        from addaxai.orchestration.deploy_helpers import scan_media_presence
        (tmp_path / "photo.jpg").write_text("")
        img, vid = scan_media_presence(
            str(tmp_path), check_img=True, check_vid=True, exclude_subs=False)
        assert img is True
        assert vid is False

    def test_finds_videos(self, tmp_path):
        from addaxai.orchestration.deploy_helpers import scan_media_presence
        (tmp_path / "clip.mp4").write_text("")
        img, vid = scan_media_presence(
            str(tmp_path), check_img=True, check_vid=True, exclude_subs=False)
        assert img is False
        assert vid is True

    def test_finds_both(self, tmp_path):
        from addaxai.orchestration.deploy_helpers import scan_media_presence
        (tmp_path / "photo.jpg").write_text("")
        (tmp_path / "clip.mp4").write_text("")
        img, vid = scan_media_presence(
            str(tmp_path), check_img=True, check_vid=True, exclude_subs=False)
        assert img is True
        assert vid is True

    def test_empty_folder(self, tmp_path):
        from addaxai.orchestration.deploy_helpers import scan_media_presence
        img, vid = scan_media_presence(
            str(tmp_path), check_img=True, check_vid=True, exclude_subs=False)
        assert img is False
        assert vid is False

    def test_exclude_subs_ignores_subdirs(self, tmp_path):
        from addaxai.orchestration.deploy_helpers import scan_media_presence
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "photo.jpg").write_text("")
        img, vid = scan_media_presence(
            str(tmp_path), check_img=True, check_vid=True, exclude_subs=True)
        assert img is False

    def test_recursive_finds_in_subdirs(self, tmp_path):
        from addaxai.orchestration.deploy_helpers import scan_media_presence
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "photo.jpg").write_text("")
        img, vid = scan_media_presence(
            str(tmp_path), check_img=True, check_vid=True, exclude_subs=False)
        assert img is True

    def test_check_img_false_skips_images(self, tmp_path):
        from addaxai.orchestration.deploy_helpers import scan_media_presence
        (tmp_path / "photo.jpg").write_text("")
        img, vid = scan_media_presence(
            str(tmp_path), check_img=False, check_vid=True, exclude_subs=False)
        assert img is False


# ─── build_deploy_options ────────────────────────────────────────────────────

class TestBuildDeployOptions:
    def test_importable(self):
        from addaxai.orchestration.deploy_helpers import build_deploy_options  # noqa: F401

    def test_simple_mode_img_options(self):
        from addaxai.orchestration.deploy_helpers import build_deploy_options
        img_opts, vid_opts = build_deploy_options(simple_mode=True)
        assert "--recursive" in img_opts
        assert "--output_relative_filenames" in img_opts

    def test_simple_mode_vid_options(self):
        from addaxai.orchestration.deploy_helpers import build_deploy_options
        _, vid_opts = build_deploy_options(simple_mode=True)
        assert "--recursive" in vid_opts
        assert "--time_sample=1" in vid_opts
        assert "--json_confidence_threshold=0.01" in vid_opts

    def test_advanced_mode_recursive(self):
        from addaxai.orchestration.deploy_helpers import build_deploy_options
        img_opts, vid_opts = build_deploy_options(
            simple_mode=False, exclude_subs=False)
        assert "--recursive" in img_opts
        assert "--recursive" in vid_opts

    def test_advanced_mode_no_recursive(self):
        from addaxai.orchestration.deploy_helpers import build_deploy_options
        img_opts, vid_opts = build_deploy_options(
            simple_mode=False, exclude_subs=True)
        assert "--recursive" not in img_opts
        assert "--recursive" not in vid_opts

    def test_checkpoint_options(self):
        from addaxai.orchestration.deploy_helpers import build_deploy_options
        img_opts, _ = build_deploy_options(
            simple_mode=False, use_checkpoints=True, checkpoint_freq="200")
        assert "--checkpoint_frequency=200" in img_opts

    def test_custom_image_size(self):
        from addaxai.orchestration.deploy_helpers import build_deploy_options
        img_opts, _ = build_deploy_options(
            simple_mode=False, custom_img_size="640")
        assert "--image_size=640" in img_opts

    def test_nth_frame_option(self):
        from addaxai.orchestration.deploy_helpers import build_deploy_options
        _, vid_opts = build_deploy_options(
            simple_mode=False, not_all_frames=True, nth_frame="2")
        assert "--time_sample=2" in vid_opts

    def test_timelapse_mode(self):
        from addaxai.orchestration.deploy_helpers import build_deploy_options
        _, vid_opts = build_deploy_options(
            simple_mode=True, timelapse_mode=True)
        assert "--include_all_processed_frames" in vid_opts

    def test_temp_frame_folder(self):
        from addaxai.orchestration.deploy_helpers import build_deploy_options
        _, vid_opts = build_deploy_options(
            simple_mode=True, temp_frame_folder="/tmp/frames")
        assert "--frame_folder=/tmp/frames" in vid_opts
        assert "--keep_extracted_frames" in vid_opts


# ─── scan_special_characters ─────────────────────────────────────────────────

class TestScanSpecialCharacters:
    def test_importable(self):
        from addaxai.orchestration.deploy_helpers import scan_special_characters  # noqa: F401

    def test_no_special_chars(self, tmp_path):
        from addaxai.orchestration.deploy_helpers import scan_special_characters
        (tmp_path / "normal_photo.jpg").write_text("")
        result = scan_special_characters(str(tmp_path))
        assert result["total_files"] == 0

    def test_returns_dict_with_total(self, tmp_path):
        from addaxai.orchestration.deploy_helpers import scan_special_characters
        result = scan_special_characters(str(tmp_path))
        assert "total_files" in result
        assert "paths" in result
