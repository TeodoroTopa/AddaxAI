"""Tests for addaxai.models.download — model download utilities (TDD).

Tests verify:
  - needs_update version comparison logic
  - get_download_info manifest parsing
  - fetch_manifest accepts current_version parameter
  - download_model_files accepts progress callback
  - download_and_extract_env accepts progress callbacks
"""


# ─── needs_update ────────────────────────────────────────��───────────────────

class TestNeedsUpdate:
    def test_importable(self):
        from addaxai.models.download import needs_update  # noqa: F401

    def test_lower_version_needs_update(self):
        from addaxai.models.download import needs_update
        assert needs_update("1.0.0", "2.0.0") is True

    def test_higher_version_no_update(self):
        from addaxai.models.download import needs_update
        assert needs_update("3.0.0", "2.0.0") is False

    def test_equal_version_no_update(self):
        from addaxai.models.download import needs_update
        assert needs_update("2.0.0", "2.0.0") is False

    def test_minor_version_needs_update(self):
        from addaxai.models.download import needs_update
        assert needs_update("2.0.0", "2.1.0") is True

    def test_patch_version_needs_update(self):
        from addaxai.models.download import needs_update
        assert needs_update("2.0.0", "2.0.1") is True

    def test_uneven_version_lengths(self):
        from addaxai.models.download import needs_update
        assert needs_update("2.0", "2.0.1") is True
        assert needs_update("2.0.1", "2.0") is False


# ─── get_download_info ───────────────────────────────────────────────────────

class TestGetDownloadInfo:
    def test_importable(self):
        from addaxai.models.download import get_download_info  # noqa: F401

    def test_returns_urls_and_size(self):
        from addaxai.models.download import get_download_info
        manifest = {
            "windows-env-test.zip": {
                "parts": ["part1.zip", "part2.zip"],
                "total_size": 1024,
            }
        }
        urls, size = get_download_info(manifest, "https://example.com", "windows-env-test.zip")
        assert urls == ["https://example.com/part1.zip", "https://example.com/part2.zip"]
        assert size == 1024


# ─── fetch_manifest ──────────────────────────────────────────────────────────

class TestFetchManifest:
    def test_importable(self):
        from addaxai.models.download import fetch_manifest  # noqa: F401

    def test_accepts_current_version_param(self):
        import inspect
        from addaxai.models.download import fetch_manifest
        params = inspect.signature(fetch_manifest).parameters
        assert "current_version" in params
        assert "platform_prefix" in params


# ─── download_model_files ────────────────────────────────────────────────────

class TestDownloadModelFiles:
    def test_importable(self):
        from addaxai.models.download import download_model_files  # noqa: F401

    def test_accepts_progress_callback(self):
        import inspect
        from addaxai.models.download import download_model_files
        params = inspect.signature(download_model_files).parameters
        assert "progress_callback" in params


# ─── download_and_extract_env ────────────────────────────────────────────────

class TestDownloadAndExtractEnv:
    def test_importable(self):
        from addaxai.models.download import download_and_extract_env  # noqa: F401

    def test_accepts_progress_callbacks(self):
        import inspect
        from addaxai.models.download import download_and_extract_env
        params = inspect.signature(download_and_extract_env).parameters
        assert "download_progress_callback" in params
        assert "extraction_progress_callback" in params
