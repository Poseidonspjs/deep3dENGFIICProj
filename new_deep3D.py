# deep3d_demo.py
import os
import urllib.request
import mxnet as mx
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from images2gif import writeGif

# -----------------------------
# Download pretrained model if missing
# -----------------------------
MODEL_URL = "http://homes.cs.washington.edu/~jxie/download/deep3d-0050.params"
MODEL_PREFIX = "deep3d"
MODEL_FILE = "deep3d-0050.params"
MODEL_EPOCH = 50

if not os.path.exists(MODEL_FILE):
    print("Downloading pretrained Deep3D model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
    print("Download complete.")

# -----------------------------
# Load MXNet model
# -----------------------------
ctx = mx.gpu() if mx.context.num_gpus() > 0 else mx.cpu()
model = mx.model.FeedForward.load(MODEL_PREFIX, MODEL_EPOCH, ctx)

# -----------------------------
# Load input image
# -----------------------------
IMAGE_FILE = "demo.jpg"
shape = (384, 160)  # resize to Deep3D expected input

img = cv2.imread(IMAGE_FILE)
if img is None:
    raise FileNotFoundError(f"Image file '{IMAGE_FILE}' not found.")

raw_shape = (img.shape[1], img.shape[0])
img_resized = cv2.resize(img, shape)
plt.imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

# -----------------------------
# Prepare input for MXNet
# -----------------------------
X = img_resized.astype(np.float32).transpose((2, 0, 1))
X = X.reshape((1,) + X.shape)
test_iter = mx.io.NDArrayIter({'left': X, 'left0': X})

# -----------------------------
# Predict right view
# -----------------------------
Y = model.predict(test_iter)
right = np.clip(Y.squeeze().transpose((1, 2, 0)), 0, 255).astype(np.uint8)

# Convert to PIL images for GIF
left_pil = Image.fromarray(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
right_pil = Image.fromarray(cv2.cvtColor(right, cv2.COLOR_BGR2RGB))

# -----------------------------
# Save GIF
# -----------------------------
writeGif("demo.gif", [left_pil, right_pil], duration=0.08)
print("GIF saved as 'demo.gif'.")
