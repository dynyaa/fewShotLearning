from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import cv2
import numpy as np
import os

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def encode_video(video_path):
    frames = extract_frames(video_path)
    embeddings = []
    for frame in frames:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = processor(images=image, return_tensors="pt", padding=True)
        outputs = model.get_image_features(**inputs)
        embeddings.append(outputs.detach().cpu().numpy())
    return np.array(embeddings)

locations = {
    "Bottom Center": r"C:\Users\robot\Desktop\fewshot_Dataset\Bottom Center",
    "Top Right": r"C:\Users\robot\Desktop\fewshot_Dataset\Top Right",
    "Bottom Left": r"C:\Users\robot\Desktop\fewshot_Dataset\Bottom Left",
    "Bottom Right": r"C:\Users\robot\Desktop\fewshot_Dataset\Bottom Right",
    "Center": r"C:\Users\robot\Desktop\fewshot_Dataset\Center",
    "Left": r"C:\Users\robot\Desktop\fewshot_Dataset\Left",
    "Right": r"C:\Users\robot\Desktop\fewshot_Dataset\Right",
    "Top Center": r"C:\Users\robot\Desktop\fewshot_Dataset\Top Center",
    "Top Left": r"C:\Users\robot\Desktop\fewshot_Dataset\Top Left"
}

for location, folder_path in locations.items():
    all_embeddings = []
    for video_name in os.listdir(folder_path):
        video_path = os.path.join(folder_path, video_name)
        embeddings = encode_video(video_path)
        all_embeddings.append(embeddings)
    
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    save_path = os.path.join(r"C:\Users\robot\Desktop\embeddings", f"{location}_embeddings.npy")
    np.save(save_path, all_embeddings)
    print(f"Saved embeddings for {location} at {save_path}")
