import torch
import numpy as np
import cv2
import json
import os
from model import SkeletonTransformer
from preprocess import extract_pose_from_video, normalize_pose, TARGET_FRAMES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path):
    model = SkeletonTransformer(
        input_dim=132,
        seq_len=30,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        num_classes=6,
        dropout=0.1,
    ).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model


def load_label_map(label_map_path):
    with open(label_map_path, "r") as f:
        label_map = json.load(f)
    id_to_name = {v: k for k, v in label_map.items()}
    return id_to_name


def predict_video(video_path, model, id_to_name):
    pose_seq = extract_pose_from_video(video_path)
    if pose_seq is None:
        return None, None
    norm_seq = np.array([normalize_pose(f) for f in pose_seq])
    input_tensor = torch.FloatTensor(norm_seq).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(input_tensor)
        prob = torch.softmax(out, dim=1)
        conf, pred_id = torch.max(prob, dim=1)
    return id_to_name[pred_id.item()], conf.item()


if __name__ == "__main__":
    model = load_model("/home/shi_chou_chu_jin/cv-course/work13/data/best_model.pth")
    id_to_name = load_label_map(
        "/home/shi_chou_chu_jin/cv-course/work13/data/preprocessed/label_map.json"
    )
    test_video = "/home/shi_chou_chu_jin/cv-course/work13/data/badminton_dataset/forehand_clear/100.mp4"
    if os.path.exists(test_video):
        cls, conf = predict_video(test_video, model, id_to_name)
        if cls:
            print(f"Predicted class: {cls}")
            print(f"Confidence: {conf:.4f}")
    else:
        print(f"视频不存在: {test_video}")
