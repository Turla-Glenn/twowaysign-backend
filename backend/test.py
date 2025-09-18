import base64
import cv2
import numpy as np

test_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8Xw8AAoMBgIFuF+0AAAAASUVORK5CYII="

# Strip extra characters if any
clean_b64 = test_b64[:88]
print("Clean length:", len(clean_b64))  # ✅ Should be 88

# Decode Base64
img_bytes = base64.b64decode(clean_b64)
img_array = np.frombuffer(img_bytes, np.uint8)
img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

if img is None:
    print("❌ Failed to decode")
else:
    print("✅ Image decoded successfully", img.shape)
