# Model Adapter Template

This directory is a **template** for creating a new classification model adapter for AddaxAI.

## Quick Start

1. **Copy this directory** and rename it to your model's name:
   ```bash
   cp -r _template my-model-v1.0
   ```

2. **Edit `classify_detections.py`**:
   - Replace `TODO: Import your model framework` with actual imports
   - Replace `TODO: Load the model` with your model loading logic
   - Replace `TODO: Classify detections` with your classification logic
   - Keep the argument parsing and JSON I/O structure intact

3. **Create `variables.json`** in `AddaxAI_files/AddaxAI/models/cls/my-model-v1.0/`:
   ```json
   {
     "model_type": "my-model",
     "framework": "pytorch",
     "model_fname": "model_checkpoint.pth",
     "all_classes": ["lion", "zebra", "giraffe", "elephant"],
     "var_cls_detec_thresh": 0.5,
     "var_cls_detec_thresh_default": 0.5,
     "var_cls_class_thresh": 0.5,
     "var_cls_class_thresh_default": 0.5,
     "developer": "Your Name",
     "description": "Species classifier trained on African wildlife",
     "info_url": "https://github.com/yourusername/my-model"
   }
   ```

4. **Place model weights** in `AddaxAI_files/AddaxAI/models/cls/my-model-v1.0/`:
   - E.g., `model_checkpoint.pth` (name must match `model_fname` in `variables.json`)

5. **Test locally**:
   ```bash
   python classify_detections.py \
     /path/to/AddaxAI_files \
     /path/to/model_checkpoint.pth \
     0.5 \
     0.5 \
     False \
     /path/to/recognition.json \
     None \
     False \
     0
   ```

6. **Launch AddaxAI GUI** and select your model from the dropdown

## Understanding the Arguments

Your `classify_detections.py` receives 9 positional arguments via `sys.argv`:

| Arg | Name | Type | Meaning |
|-----|------|------|---------|
| 1 | `AddaxAI_files` | str | Base path (e.g., `/home/user/AddaxAI_files`) |
| 2 | `cls_model_fpath` | str | Path to checkpoint (e.g., `...models/cls/my-model/weights.pth`) |
| 3 | `cls_detec_thresh` | float | Skip detections below this confidence (0.0-1.0) |
| 4 | `cls_class_thresh` | float | Only report classifications above this (0.0-1.0) |
| 5 | `smooth_bool` | "True"/"False" | Smooth predictions across image burst/sequence |
| 6 | `json_path` | str | Path to recognition JSON (read + write) |
| 7 | `temp_frame_folder` | str | Temp dir for video frames, or "None" |
| 8 | `cls_tax_fallback` | "True"/"False" | Use taxonomy fallback for uncertain predictions |
| 9 | `cls_tax_levels_idx` | int | Taxonomy level (0=species, 1=genus, etc.) |

## Input JSON Structure

Your script receives a **recognition JSON** with this structure:

```json
{
  "images": [
    {
      "file": "IMG_001.JPG",
      "detections": [
        {
          "category": "1",
          "conf": 0.92,
          "bbox": [0.1, 0.2, 0.5, 0.8]
        }
      ]
    }
  ],
  "detection_categories": {
    "1": "animal",
    "2": "person",
    "3": "vehicle"
  }
}
```

**Your job:** Add `"classifications"` to each detection.

## Output JSON Structure

After classification, each detection should have this structure:

```json
{
  "category": "1",
  "conf": 0.92,
  "bbox": [0.1, 0.2, 0.5, 0.8],
  "classifications": [
    ["lion", 0.91],
    ["tiger", 0.07],
    ["leopard", 0.02]
  ]
}
```

**Format:** Each classification is `[species_name, confidence]`, sorted by confidence descending.

## variables.json Reference

This file is **required** and must live at:
```
AddaxAI_files/AddaxAI/models/cls/my-model-v1.0/variables.json
```

| Field | Required | Type | Purpose |
|-------|----------|------|---------|
| `model_type` | ✅ | string | Model identifier (e.g., "my-model", "SpeciesNet") |
| `framework` | ✅ | string | ML framework: "pytorch", "tensorflow", or "onnx" |
| `model_fname` | ✅ | string | Checkpoint filename (relative to model dir) |
| `all_classes` | ✅ | array | List of all species this model can classify |
| `var_cls_detec_thresh` | ✅ | number | Default detection threshold (0.0-1.0) |
| `var_cls_detec_thresh_default` | ✅ | number | (same as above) |
| `var_cls_class_thresh` | ✅ | number | Default classification threshold |
| `var_cls_class_thresh_default` | ✅ | number | (same as above) |
| `developer` | ✓ | string | Your name or organization |
| `description` | ✓ | string | Short description of the model |
| `info_url` | ✓ | string | Link to model docs or GitHub |
| `var_smooth_cls_animal` | ✓ | bool | Whether to smooth across bursts |
| `var_tax_levels_idx` | ✓ | int | Taxonomy depth (0=species) |
| `full_image_cls` | ✓ | bool | If true, classify entire image not crops |
| `taxon_mapping_csv` | ✓ | string | URL to taxonomy mapping CSV (downloaded automatically) |

## Example: PyTorch Model

```python
import torch
from torchvision import transforms
from your_model import YourModel

# ....

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YourModel.load_checkpoint(cls_model_fpath)
model = model.to(device)
model.eval()

# Preprocess
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Classify crops
for image_entry in images:
    image_path = image_entry["file"]
    if not os.path.isabs(image_path):
        image_path = os.path.join(AddaxAI_files, image_path)

    img = Image.open(image_path).convert("RGB")

    for detection in image_entry.get("detections", []):
        if detection["conf"] < cls_detec_thresh:
            continue

        bbox = detection["bbox"]
        x_min, y_min, x_max, y_max = bbox
        w, h = img.size
        crop = img.crop((int(x_min*w), int(y_min*h), int(x_max*w), int(y_max*h)))

        # Classify
        x = transform(crop).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]

        # Format output
        sorted_probs = sorted(zip(model.classes, probs.tolist()), key=lambda x: -x[1])
        classifications = [
            [name, float(conf)] for name, conf in sorted_probs
            if conf >= cls_class_thresh
        ]

        if classifications:
            detection["classifications"] = classifications
```

## Example: TensorFlow Model

```python
import tensorflow as tf
import numpy as np

# ....

# Load model
model = tf.keras.models.load_model(cls_model_fpath)
class_names = [...]  # Load from somewhere

# Classify crops
for image_entry in images:
    image_path = image_entry["file"]
    if not os.path.isabs(image_path):
        image_path = os.path.join(AddaxAI_files, image_path)

    img = Image.open(image_path).convert("RGB")

    for detection in image_entry.get("detections", []):
        if detection["conf"] < cls_detec_thresh:
            continue

        bbox = detection["bbox"]
        x_min, y_min, x_max, y_max = bbox
        w, h = img.size
        crop = img.crop((int(x_min*w), int(y_min*h), int(x_max*w), int(y_max*h)))

        # Preprocess
        crop = crop.resize((224, 224))
        x = np.expand_dims(np.array(crop, dtype=np.float32) / 255.0, axis=0)

        # Classify
        logits = model(x)[0]
        probs = tf.nn.softmax(logits).numpy()

        # Format output
        sorted_idx = np.argsort(-probs)
        classifications = [
            [class_names[i], float(probs[i])] for i in sorted_idx
            if probs[i] >= cls_class_thresh
        ]

        if classifications:
            detection["classifications"] = classifications
```

## Error Handling

- **Exit 0** on success (or if no detections to classify)
- **Exit 1** on fatal error (missing image file, model load failure, etc.)
- **Print warnings to stdout** (captured in AddaxAI log) for non-fatal issues
- Always wrap in try/except to catch unexpected errors

## Testing Your Adapter

1. Create a test recognition JSON with a few sample detections
2. Run your script manually with test arguments
3. Verify the output JSON has `classifications` fields with correct format
4. Check `AddaxAI_files/addaxai.log` for any warnings

## Conda Environment (Optional)

If your model requires specific packages, create a conda environment:

```bash
conda create -n my-model-env python=3.8 pytorch::pytorch torchvision torchaudio -c pytorch
```

Then specify in `variables.json`:

```json
{
  "env": "my-model-env",
  ...
}
```

AddaxAI will activate this environment before running your script.

## Troubleshooting

**"ModuleNotFoundError: No module named 'your_model'"**
- Check your imports are correct
- Verify the package is installed in the conda environment (if using one)

**"Model not loading"**
- Verify `model_fname` in `variables.json` matches the actual checkpoint name
- Check file permissions
- Print debug info in your script (goes to log)

**"No classifications in output"**
- Check `cls_class_thresh` is reasonable (default 0.5)
- Verify your model is actually outputting logits/probabilities
- Check for exceptions in the log file

## Publishing Your Adapter

Once working locally:

1. Create a GitHub repo with your adapter code
2. Open an issue on the [main AddaxAI repo](https://github.com/PetervanLunteren/AddaxAI) proposing to add your model
3. Submit a PR with:
   - Your `classify_detections.py` in `classification_utils/model_types/your-model/`
   - A `variables.json` template
   - License (recommend CC-BY or CC0 for animal classifiers)
