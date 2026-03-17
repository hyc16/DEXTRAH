from ultralytics import SAM
import numpy as np
import torch

print("torch.cuda.is_available() =", torch.cuda.is_available())
print("torch.cuda.device_count() =", torch.cuda.device_count())

model = SAM("sam2.1_l.pt")

img = np.zeros((512, 512, 3), dtype=np.uint8)
results = model(img, device=0, imgsz=512)

print("inference ok")
print("num results =", len(results))