"""Standard event names emitted by AddaxAI.

These are string constants — not an enum — so they can be used as dict keys
and in f-strings without .value access. Grouped by feature area.

Events are emitted by orchestration logic (deploy_model, classify_detections, etc.)
and listened to by UI components to update progress, handle errors, etc.
"""

# ============================================================================
# Deployment — running MegaDetector on a folder
# ============================================================================
DEPLOY_STARTED = "deploy.started"
"""Emitted when deployment begins."""

DEPLOY_PROGRESS = "deploy.progress"
"""Emitted during deployment. kwargs: pct (0-100), message (str)."""

DEPLOY_IMAGE_COMPLETE = "deploy.image_complete"
"""Emitted when one image finishes detection. kwargs: image_path, index, total."""

DEPLOY_ERROR = "deploy.error"
"""Emitted on detection error. kwargs: message (str), exc (Exception or None)."""

DEPLOY_CANCELLED = "deploy.cancelled"
"""Emitted when user cancels deployment."""

DEPLOY_FINISHED = "deploy.finished"
"""Emitted when deployment completes. kwargs: results_path (str)."""

# ============================================================================
# Classification — running species classifier on detections
# ============================================================================
CLASSIFY_STARTED = "classify.started"
"""Emitted when classification begins."""

CLASSIFY_PROGRESS = "classify.progress"
"""Emitted during classification. kwargs: pct (0-100), message (str)."""

CLASSIFY_ERROR = "classify.error"
"""Emitted on classification error. kwargs: message (str)."""

CLASSIFY_FINISHED = "classify.finished"
"""Emitted when classification completes. kwargs: results_path (str)."""

# ============================================================================
# Postprocessing — separating files by detection type
# ============================================================================
POSTPROCESS_STARTED = "postprocess.started"
"""Emitted when postprocessing begins."""

POSTPROCESS_PROGRESS = "postprocess.progress"
"""Emitted during postprocessing. kwargs: pct (0-100), message (str)."""

POSTPROCESS_ERROR = "postprocess.error"
"""Emitted on postprocessing error. kwargs: message (str)."""

POSTPROCESS_FINISHED = "postprocess.finished"
"""Emitted when postprocessing completes."""

# ============================================================================
# Model management — downloading and setting up models
# ============================================================================
MODEL_DOWNLOAD_STARTED = "model.download_started"
"""Emitted when model download begins. kwargs: model_name (str)."""

MODEL_DOWNLOAD_PROGRESS = "model.download_progress"
"""Emitted during download. kwargs: pct (0-100)."""

MODEL_DOWNLOAD_FINISHED = "model.download_finished"
"""Emitted when download completes. kwargs: model_name (str)."""

MODEL_DOWNLOAD_ERROR = "model.download_error"
"""Emitted on download error. kwargs: model_name (str), message (str)."""
