import cv2
import mediapipe as mp
import numpy as np
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import json

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

TARGET_FRAMES = 30
KEYPOINT_DIM = 33 * 4

CLASS_MAP = {
    "forehand_drive": 0,
    "forehand_lift": 1,
    "forehand_net_shot": 2,
    "forehand_clear": 3,
    "backhand_drive": 4,
    "backhand_net_shot": 5,
}


def normalize_pose(keypoints_2d):
    keypoints = keypoints_2d.reshape(-1, 4)
    left_hip = keypoints[23, :2]
    right_hip = keypoints[24, :2]
    hip_center = (left_hip + right_hip) / 2
    left_shoulder = keypoints[11, :2]
    right_shoulder = keypoints[12, :2]
    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
    if shoulder_width > 1e-6:
        keypoints[:, :2] = (keypoints[:, :2] - hip_center) / shoulder_width
    else:
        keypoints[:, :2] = keypoints[:, :2] - hip_center
    return keypoints.flatten()


def uniform_sample_sequence(seq, target_len):
    current_len = len(seq)
    if current_len == target_len:
        return seq
    indices = np.linspace(0, current_len - 1, target_len).astype(int)
    return seq[indices]


def extract_pose_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    all_poses = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            frame_pose = []
            for lm in landmarks:
                frame_pose.extend([lm.x, lm.y, lm.z, lm.visibility])
            all_poses.append(np.array(frame_pose))
        else:
            all_poses.append(np.zeros(KEYPOINT_DIM))
    cap.release()
    if len(all_poses) == 0:
        return None
    all_poses = np.array(all_poses)
    return uniform_sample_sequence(all_poses, TARGET_FRAMES)


def process_dataset(data_root, output_root):
    X, y = [], []
    for class_name, class_id in CLASS_MAP.items():
        class_path = os.path.join(data_root, class_name)
        if not os.path.exists(class_path):
            print(f"跳过缺失目录: {class_path}")
            continue
        videos = [
            f
            for f in os.listdir(class_path)
            if f.endswith((".mp4", ".avi", ".mov", ".mkv"))
        ]
        print(f"处理 {class_name} ({class_id}): {len(videos)} 个视频")
        for vid in tqdm(videos, desc=class_name):
            video_path = os.path.join(class_path, vid)
            pose_seq = extract_pose_from_video(video_path)
            if pose_seq is not None:
                norm_seq = np.array([normalize_pose(f) for f in pose_seq])
                X.append(norm_seq)
                y.append(class_id)
    X = np.array(X)
    y = np.array(y)
    print(f"总计有效样本: {len(X)}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    os.makedirs(output_root, exist_ok=True)
    np.save(os.path.join(output_root, "X_train.npy"), X_train)
    np.save(os.path.join(output_root, "y_train.npy"), y_train)
    np.save(os.path.join(output_root, "X_test.npy"), X_test)
    np.save(os.path.join(output_root, "y_test.npy"), y_test)
    with open(os.path.join(output_root, "label_map.json"), "w") as f:
        json.dump(CLASS_MAP, f, indent=2)
    print(f"数据保存至 {output_root}")
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")


if __name__ == "__main__":
    DATA_ROOT = "/home/shi_chou_chu_jin/cv-course/work13/data/badminton_dataset"
    OUTPUT_ROOT = "/home/shi_chou_chu_jin/cv-course/work13/data/preprocessed"
    process_dataset(DATA_ROOT, OUTPUT_ROOT)
