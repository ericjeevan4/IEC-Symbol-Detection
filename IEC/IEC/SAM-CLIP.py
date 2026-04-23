#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ================================
# Cell 1: Setup SAM + CLIP Environment
# ================================

# Install required packages
get_ipython().system('pip install -U      torch torchvision torchaudio      opencv-python      matplotlib      pillow      numpy      git+https://github.com/facebookresearch/segment-anything.git      ftfy regex tqdm')

# Install CLIP
get_ipython().system('pip install git+https://github.com/openai/CLIP.git')

print("✅ Packages installed")

# ----------------
# Load SAM
# ----------------
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"
SAM_TYPE = "vit_h"

# Download SAM checkpoint if not present
import os
if not os.path.exists(SAM_CHECKPOINT):
    get_ipython().system('wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth')

device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[SAM_TYPE](
    checkpoint=SAM_CHECKPOINT
).to(device)

mask_generator = SamAutomaticMaskGenerator(
    sam,
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    min_mask_region_area=100  # important for small symbols
)

print("✅ SAM model loaded")

# ----------------
# Load CLIP
# ----------------
import clip

clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

print("✅ CLIP model loaded")
print("🔥 Device:", device)
print("💾 GPU:", torch.cuda.get_device_name(0))


# In[2]:


# ================================
# Cell 2a: Upload P&ID Image
# ================================

from ipywidgets import FileUpload
from IPython.display import display

uploader = FileUpload(accept='image/*', multiple=False)
print("📤 Upload your P&ID image:")
display(uploader)


# In[3]:


# ================================
# Cell 2b: Load & Display Image
# ================================

import io
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Handle both widget formats safely
if isinstance(uploader.value, tuple):
    uploaded_file = uploader.value[0]
else:
    uploaded_file = list(uploader.value.values())[0]

# Load image bytes
image_bytes = uploaded_file['content']

# Convert to PIL and OpenCV
pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
image = np.array(pil_image)
image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# Display image
plt.figure(figsize=(8, 8))
plt.imshow(image)
plt.axis("off")
plt.title("Uploaded P&ID Image")
plt.show()

print("✅ Image loaded successfully")
print("🖼️ Image shape:", image.shape)


# In[4]:


# ================================
# Cell 3: SAM Mask Generation → Bounding Boxes
# ================================

import numpy as np
import cv2
import matplotlib.pyplot as plt

print("🧠 Running SAM mask generator... (this may take ~30–60 seconds)")

masks = mask_generator.generate(image_bgr)

print(f"✅ Total masks detected by SAM: {len(masks)}")

# Convert masks to bounding boxes
sam_boxes = []

for m in masks:
    y_indices, x_indices = np.where(m["segmentation"])
    if len(x_indices) == 0 or len(y_indices) == 0:
        continue

    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()

    # Filter extremely small noise boxes
    if (x_max - x_min) < 8 or (y_max - y_min) < 8:
        continue

    sam_boxes.append({
        "bbox": [int(x_min), int(y_min), int(x_max), int(y_max)],
        "area": m["area"]
    })

print(f"📦 Bounding boxes after filtering: {len(sam_boxes)}")

# ----------------
# Visualize ALL boxes
# ----------------
vis = image.copy()

for box in sam_boxes:
    x1, y1, x2, y2 = box["bbox"]
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)

plt.figure(figsize=(10, 8))
plt.imshow(vis)
plt.axis("off")
plt.title("SAM Detected Shapes (All Bounding Boxes)")
plt.show()


# In[6]:


# ================================
# Cell 4 (FIXED): SAM + CLIP Classification
# ================================

import torch
import numpy as np
import cv2
import json
from PIL import Image

# --------------------------------
# Improved P&ID symbol prompts
# --------------------------------
CLASSES = [
    "valve symbol",
    "control valve symbol",
    "pump symbol",
    "compressor symbol",
    "heat exchanger symbol",
    "cyclone separator symbol",
    "conveyor symbol",
    "bucket elevator symbol",
    "roller conveyor symbol",
    "screw conveyor symbol",
    "chimney symbol",
    "flare symbol",
    "spray nozzle symbol",
    "feeder symbol",
    "palletizer symbol",
    "press machine symbol",
    "crane symbol",
    "magnet symbol",
    "screening symbol"
]

TEXT_PROMPTS = [
    f"black and white industrial schematic symbol of a {c}"
    for c in CLASSES
]

text_tokens = clip.tokenize(TEXT_PROMPTS).to(device)

results = []

print("🔍 Running CLIP classification...")

for box in sam_boxes:
    x1, y1, x2, y2 = box["bbox"]
    crop = image[y1:y2, x1:x2]

    if crop.shape[0] < 12 or crop.shape[1] < 12:
        continue

    crop_pil = Image.fromarray(crop)
    crop_input = clip_preprocess(crop_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        image_feat = clip_model.encode_image(crop_input)
        text_feat = clip_model.encode_text(text_tokens)

        image_feat /= image_feat.norm(dim=-1, keepdim=True)
        text_feat /= text_feat.norm(dim=-1, keepdim=True)

        similarity = (image_feat @ text_feat.T)[0]
        best_idx = similarity.argmax().item()
        confidence = similarity[best_idx].item()

    # MUCH LOWER threshold
    if confidence < 0.05:
        continue

    results.append({
        "label": CLASSES[best_idx],
        "confidence": round(float(confidence), 3),
        "bbox": [x1, y1, x2, y2]
    })

print(f"✅ Classified symbols: {len(results)}")

with open("sam_clip_pid_detections.json", "w") as f:
    json.dump(results, f, indent=2)

print("📁 Saved to sam_clip_pid_detections.json")


# In[7]:


# ================================
# Cell 5: Visualization of SAM + CLIP Results
# ================================

import cv2
import json
import matplotlib.pyplot as plt
import numpy as np

# Load results
with open("sam_clip_pid_detections.json", "r") as f:
    results = json.load(f)

print(f"📦 Visualizing {len(results)} classified symbols")

vis = image.copy()

for det in results:
    x1, y1, x2, y2 = det["bbox"]
    label = det["label"]
    conf = det["confidence"]

    # Color by confidence
    if conf >= 0.15:
        color = (0, 255, 0)   # strong
    elif conf >= 0.10:
        color = (255, 255, 0) # medium
    else:
        color = (255, 0, 0)   # weak

    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)

    text = f"{label} ({conf:.2f})"
    cv2.putText(
        vis,
        text,
        (x1, max(y1 - 4, 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        color,
        1
    )

plt.figure(figsize=(12, 8))
plt.imshow(vis)
plt.axis("off")
plt.title("SAM + CLIP Classified P&ID Symbols")
plt.show()


# In[8]:


# ================================
# Cell 6: Extract & Save Symbol Crops
# ================================

import os
import json
import cv2
import numpy as np

# Load detection results
with open("sam_clip_pid_detections.json", "r") as f:
    detections = json.load(f)

# Output directory
OUTPUT_DIR = "pid_symbol_crops"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"📦 Saving symbol crops to: {OUTPUT_DIR}")

count = 0

for idx, det in enumerate(detections):
    label = det["label"].replace(" ", "_")
    x1, y1, x2, y2 = det["bbox"]

    # Create label-specific folder
    label_dir = os.path.join(OUTPUT_DIR, label)
    os.makedirs(label_dir, exist_ok=True)

    # Crop image
    crop = image[y1:y2, x1:x2]

    # Skip invalid crops
    if crop.size == 0:
        continue

    # Save image
    filename = f"{label}_{idx:04d}.png"
    filepath = os.path.join(label_dir, filename)
    cv2.imwrite(filepath, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

    count += 1

print(f"✅ Extracted and saved {count} symbol images")


# In[ ]:




