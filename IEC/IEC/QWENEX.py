#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ================================
# Cell 1 (FIX): Stable environment for Qwen2.5-VL
# ================================

# Step 1: Remove broken versions
get_ipython().system('pip uninstall -y numpy transformers scipy huggingface-hub')

# Step 2: Install COMPATIBLE versions
get_ipython().system('pip install      numpy==1.26.4      scipy==1.12.0      transformers==4.57.6      huggingface-hub==0.36.0      accelerate      torch      torchvision      pillow      opencv-python')

print("✅ Environment fixed. Please RESTART kernel after this cell.")


# In[3]:


# ================================
# Cell 1c (FINAL): Load Qwen2.5-VL-7B-Instruct correctly
# ================================

import torch
from transformers import AutoProcessor, AutoModel

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

print("🔄 Loading processor...")
processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

print("🔄 Loading model (FP16, GPU)...")
model = AutoModel.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,          # correct arg (torch_dtype is deprecated)
    device_map="auto",
    trust_remote_code=True
)

model.eval()

print("✅ Model loaded successfully!")
print("🔥 GPU:", torch.cuda.get_device_name(0))
print("💾 VRAM:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2), "GB")


# In[19]:


# ================================
# Cell 2 (FIXED): Upload & Load P&ID Image (AIKOSH)
# ================================

from ipywidgets import FileUpload
from PIL import Image
import matplotlib.pyplot as plt
from IPython.display import display
import io

# Create upload widget
uploader = FileUpload(accept='image/*', multiple=False)

print("📤 Please upload your P&ID image:")
display(uploader)


# In[20]:


# ================================
# Cell 2b (FIXED): Load & Display Image
# ================================

import io
from PIL import Image
import matplotlib.pyplot as plt

# Handle both tuple and dict cases safely
if isinstance(uploader.value, tuple):
    uploaded_file = uploader.value[0]
else:
    uploaded_file = list(uploader.value.values())[0]

# Extract image bytes
image_bytes = uploaded_file['content']

# Convert to PIL Image
pid_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

# Display image
plt.figure(figsize=(10, 10))
plt.imshow(pid_image)
plt.axis("off")
plt.title("Uploaded P&ID Diagram")
plt.show()

print("✅ Image loaded successfully!")
print("🖼️ Image size:", pid_image.size)


# In[10]:


# ================================
# Cell 3 (FINAL): Correct Qwen2.5-VL inference model
# ================================

import torch
from transformers import AutoProcessor
from transformers.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

print("🔄 Loading processor...")
processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

print("🔄 Loading Qwen2.5-VL inference model...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

model.eval()
print("✅ Inference model ready!")


# In[21]:


# ================================
# Cell 3b (FINAL): Correct Qwen2.5-VL multimodal inference
# ================================

import torch

# Define conversation in Qwen chat format
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {
                "type": "text",
                "text": (
                    "You are a P&ID diagram analysis system.\n\n"
                    "TASK:\n"
                    "Detect ALL P&ID symbols/components in the image.\n\n"
                    "OUTPUT RULES (STRICT):\n"
                    "- Output ONLY valid JSON\n"
                    "- No explanations\n"
                    "- No markdown\n"
                    "- No comments\n\n"
                    "FORMAT:\n"
                    "[\n"
                    "  {\n"
                    "    \"label\": \"<component_name>\",\n"
                    "    \"bbox\": [x_min, y_min, x_max, y_max]\n"
                    "  }\n"
                    "]\n\n"
                    "LABEL GUIDELINES:\n"
                    "valve, control_valve, pump, heat_exchanger, evaporator,\n"
                    "compressor, transmitter, indicator, flow_meter, tank, pipe\n\n"
                    "IMPORTANT:\n"
                    "- Coordinates must be pixel coordinates\n"
                    "- Detect every visible symbol including instruments (FT, LT, LIC, AT, etc.)"
                )
            }
        ]
    }
]

# Apply Qwen chat template (THIS IS CRITICAL)
text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# Prepare inputs (CORRECT multimodal binding)
inputs = processor(
    text=text,
    images=pid_image,
    return_tensors="pt"
).to(model.device)

# Generate output
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=False
    )

# Decode output
output_text = processor.decode(
    output_ids[0],
    skip_special_tokens=True
)

print("🔍 RAW MODEL OUTPUT:")
print(output_text)


# In[17]:


# ================================
# Cell 4 (CORRECT): Parse JSON directly from model output
# ================================

import json
import re
import ast

# Use the model output variable directly
raw_text = output_text

print("🔍 Raw text length:", len(raw_text))

# 1️⃣ Try extracting JSON inside ```json block
match = re.search(r"```json\s*(\[[\s\S]*?\])\s*```", raw_text, re.IGNORECASE)

if match:
    json_str = match.group(1)
    print("✅ JSON found inside ```json``` block")
else:
    print("⚠️ No ```json``` block found, trying generic JSON extraction")

    # 2️⃣ Generic JSON array extraction
    match = re.search(r"\[\s*\{[\s\S]*?\}\s*\]", raw_text)

    if not match:
        raise ValueError("❌ JSON array not found in model output")

    json_str = match.group(0)

# 3️⃣ Parse JSON
try:
    detections = json.loads(json_str)
except json.JSONDecodeError:
    detections = ast.literal_eval(json_str)

print(f"✅ Parsed {len(detections)} raw detections")

# 4️⃣ Remove duplicate bounding boxes
seen = set()
clean_detections = []

for det in detections:
    bbox_key = tuple(det["bbox"])
    if bbox_key not in seen:
        seen.add(bbox_key)
        clean_detections.append(det)

print(f"🧹 Cleaned detections: {len(clean_detections)}")

for d in clean_detections:
    print(d)


# In[22]:


# ================================
# Cell 5: Draw Bounding Boxes on P&ID Image
# ================================

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Convert PIL image to OpenCV format
img = np.array(pid_image)
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

for det in clean_detections:
    x1, y1, x2, y2 = det["bbox"]
    label = det["label"]

    # Draw bounding box
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw label background
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x1, y1 - h - 6), (x1 + w + 4, y1), (0, 255, 0), -1)

    # Put label text
    cv2.putText(
        img,
        label,
        (x1 + 2, y1 - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1
    )

# Display result
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("P&ID Symbols Detected with Bounding Boxes")
plt.show()


# In[ ]:




