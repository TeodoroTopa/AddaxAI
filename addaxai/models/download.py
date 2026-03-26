"""Model and environment download utilities — no GUI dependencies.

Pure download/extraction logic for models and conda environments.
All user interaction (confirmation dialogs, progress windows) is
handled via injected callbacks.
"""

import logging
import os
import platform
from typing import Callable, Dict, List, Optional, Tuple

try:
    import requests  # type: ignore[import-untyped]
except ImportError:  # unit-test env without requests
    requests = None  # type: ignore[assignment]

try:
    from tqdm import tqdm  # type: ignore[import-untyped]
except ImportError:  # unit-test env without tqdm
    tqdm = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Standard browser-like headers to avoid download blocks.
_DOWNLOAD_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:66.0) "
                  "Gecko/20100101 Firefox/66.0",
    "Accept-Encoding": "*",
    "Connection": "keep-alive",
}


def needs_update(current_version: str, required_version: str) -> bool:
    """Check whether *current_version* is older than *required_version*.

    Both versions are dot-separated numeric strings (e.g. ``"2.1.0"``).
    Shorter version strings are zero-padded before comparison.

    Returns:
        True if *current_version* is strictly lower than *required_version*.
    """
    current_parts = list(map(int, current_version.split('.')))
    required_parts = list(map(int, required_version.split('.')))

    while len(current_parts) < len(required_parts):
        current_parts.append(0)
    while len(required_parts) < len(current_parts):
        required_parts.append(0)

    for cur, req in zip(current_parts, required_parts):
        if cur < req:
            return True
        elif cur > req:
            return False

    return False


def fetch_manifest(
    platform_prefix: str,
    current_version: str,
) -> Tuple[Dict, str]:
    """Fetch the platform manifest from GitHub Release assets.

    Tries the pinned version first, then falls back to ``latest``.

    Args:
        platform_prefix: ``"windows"`` or ``"macos"``.
        current_version: App version string (e.g. ``"5.0"``).

    Returns:
        Tuple of (manifest_dict, base_url) where *base_url* is the URL
        prefix for downloading the assets listed in the manifest.

    Raises:
        RuntimeError: If no manifest could be fetched.
    """
    repo = "PetervanLunteren/AddaxAI"
    manifest_name = f"{platform_prefix}-manifest.json"
    for url in [
        f"https://github.com/{repo}/releases/download/v{current_version}/{manifest_name}",
        f"https://github.com/{repo}/releases/latest/download/{manifest_name}",
    ]:
        try:
            resp = requests.get(
                url, timeout=60, headers=_DOWNLOAD_HEADERS, allow_redirects=True)
            resp.raise_for_status()
            return resp.json(), url.rsplit("/", 1)[0]
        except Exception:
            continue
    raise RuntimeError(f"Could not fetch {manifest_name} from GitHub releases")


def get_download_info(
    manifest: Dict,
    base_url: str,
    asset_key: str,
) -> Tuple[List[str], int]:
    """Return (list_of_urls, total_size) for a given asset key in *manifest*."""
    entry = manifest[asset_key]
    urls = [f"{base_url}/{part}" for part in entry["parts"]]
    return urls, entry["total_size"]


def download_model_files(
    model_dir: str,
    download_info: List[Tuple[str, str]],
    progress_callback: Optional[Callable[[float], None]] = None,
) -> str:
    """Download model weight files to *model_dir*.

    Args:
        model_dir:         Destination directory for the model files.
        download_info:     List of ``(url, relative_filename)`` pairs.
        progress_callback: Called with a float in ``[0.0, 1.0]`` representing
                           overall download progress.

    Returns:
        Path to the last downloaded file.

    Raises:
        requests.HTTPError: If any download fails.
    """
    # compute total size
    total_size = 0
    for download_url, _ in download_info:
        response = requests.get(
            download_url, stream=True, timeout=30, headers=_DOWNLOAD_HEADERS)
        response.raise_for_status()
        total_size += int(response.headers.get('content-length', 0))

    progress_bar = tqdm(total=total_size, unit='B', unit_scale=True)
    file_path = ""
    for download_url, fname in download_info:
        file_path = os.path.normpath(os.path.join(model_dir, fname))
        dir_name = os.path.dirname(file_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        response = requests.get(
            download_url, stream=True, timeout=30, headers=_DOWNLOAD_HEADERS)
        response.raise_for_status()

        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))
                    if progress_callback is not None and total_size > 0:
                        progress_callback(progress_bar.n / total_size)
    progress_bar.close()
    logger.info("Download successful. File saved at: %s", file_path)
    return file_path


def download_and_extract_env(
    env_name: str,
    env_dir: str,
    current_version: str,
    download_progress_callback: Optional[Callable[[float], None]] = None,
    extraction_progress_callback: Optional[Callable[[float], None]] = None,
) -> str:
    """Download and extract a conda environment archive.

    Args:
        env_name:                     Environment name (e.g. ``"pytorch"``).
        env_dir:                      Base directory for environments.
        current_version:              App version string for manifest lookup.
        download_progress_callback:   Called with download progress ``[0.0, 1.0]``.
        extraction_progress_callback: Called with extraction progress ``[0.0, 1.0]``.

    Returns:
        Path to the downloaded/extracted environment.

    Raises:
        RuntimeError: On Linux (environments installed during setup).
        requests.HTTPError: If download fails.
    """
    if os.name == 'nt':
        platform_prefix = "windows"
        asset_key = f"windows-env-{env_name}.zip"
        filename = f"{env_name}.zip"
    elif platform.system() == 'Darwin':
        platform_prefix = "macos"
        asset_key = f"macos-env-{env_name}.tar.xz"
        filename = f"{env_name}.tar.xz"
    else:
        raise RuntimeError("Linux environments are installed during setup")

    manifest, base_url = fetch_manifest(platform_prefix, current_version)
    part_urls, total_size = get_download_info(manifest, base_url, asset_key)

    # download
    progress_bar = tqdm(total=total_size, unit='B', unit_scale=True)
    file_path = os.path.join(env_dir, filename)
    with open(file_path, 'wb') as f:
        for part_url in part_urls:
            response = requests.get(
                part_url, stream=True, timeout=60, headers=_DOWNLOAD_HEADERS)
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))
                    if download_progress_callback is not None and total_size > 0:
                        download_progress_callback(progress_bar.n / total_size)
    progress_bar.close()
    logger.info("Download successful. File saved at: %s", file_path)

    # extract
    _extract_archive(file_path, env_dir, extraction_progress_callback)

    return file_path


def _extract_archive(
    file_path: str,
    dest_dir: str,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> None:
    """Extract a zip or tar.xz archive and remove it afterwards."""
    if file_path.endswith(".tar.xz"):
        import tarfile
        with tarfile.open(file_path, "r:xz") as tar:
            total_files = len(tar.getnames())
            progress_bar = tqdm(total=total_files, unit='file', desc="Extracting")
            for member in tar:
                tar.extract(member, path=dest_dir)
                progress_bar.update(1)
                if progress_callback is not None and total_files > 0:
                    progress_callback(progress_bar.n / total_files)
            progress_bar.close()
    elif file_path.endswith(".zip"):
        import zipfile
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            total_files = len(zip_ref.namelist())
            progress_bar = tqdm(total=total_files, unit='file', desc="Extracting")
            for zip_entry in zip_ref.namelist():
                zip_ref.extract(zip_entry, path=dest_dir)
                progress_bar.update(1)
                if progress_callback is not None and total_files > 0:
                    progress_callback(progress_bar.n / total_files)
            progress_bar.close()
    else:
        raise ValueError(f"Unsupported archive format: {file_path}")

    logger.info("Extraction successful. Files extracted to: %s", dest_dir)

    # clean up archive
    try:
        os.remove(file_path)
        logger.info("Removed archive: %s", file_path)
    except Exception as e:
        logger.warning("Error removing file: %s", e)
