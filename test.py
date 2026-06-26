import cv2
import numpy as np
import glob
import os

# ================= 配置参数 =================
images_path = "/home/shi_chou_chu_jin/cv-course/work14/photo/*.jpg"
pattern_size = (9, 6)
square_size = 20

# ================= 定义输出目录 =================
output_dir = "/home/shi_chou_chu_jin/cv-course/work14/result"
os.makedirs(output_dir, exist_ok=True)  # 主目录
os.makedirs(os.path.join(output_dir, "corners"), exist_ok=True)  # 角点图目录
os.makedirs(os.path.join(output_dir, "undistort"), exist_ok=True)  # 去畸变图目录

# ================= 准备世界坐标 =================
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = (
    np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(-1, 2) * square_size
)

objpoints = []
imgpoints = []
image_files = glob.glob(images_path)

print(f"找到图片数量: {len(image_files)}")
if len(image_files) < 15:
    print("警告:图片数量少于15张,标定结果可能不稳定。")

# ================= 第一步：循环找角点 =================
for fname in image_files:
    img = cv2.imread(fname)
    if img is None:
        print(f"图片读取失败: {fname}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    ret, corners = cv2.findChessboardCorners(
        gray, pattern_size, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if ret == True:
        objpoints.append(objp)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (21, 21), (-1, -1), criteria)
        imgpoints.append(corners2)

        # 保存画了角点的图片
        img_corners = img.copy()
        cv2.drawChessboardCorners(img_corners, pattern_size, corners2, ret)
        save_corner_path = os.path.join(
            output_dir, "corners", f"corners_{os.path.basename(fname)}"
        )
        cv2.imwrite(save_corner_path, img_corners)
        # print(f"角点检测成功: {os.path.basename(fname)}")
    else:
        print(f"角点检测失败: {os.path.basename(fname)}")

# ================= 第二步：核心标定 =================
if len(objpoints) > 0:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    print("\n====== 标定结果 ======")
    print(f"重投影误差 (RMS): {ret:.4f} pixels")
    print("\n内参矩阵 K:\n", mtx)
    print("\n畸变参数 D (k1, k2, p1, p2, k3):\n", dist.flatten())

    # ===== 第三步：为所有图片生成去畸变图 =====
    print("\n生成所有图片的去畸变图")
    for fname in image_files:
        img = cv2.imread(fname)
        if img is None:
            continue

        h, w = img.shape[:2]
        # 获取优化的新相机矩阵
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        # 去畸变
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

        # 生成文件名
        base_name = os.path.splitext(os.path.basename(fname))[0]
        # 保存去畸变图
        undist_path = os.path.join(output_dir, "undistort", f"{base_name}_undist.jpg")

        cv2.imwrite(undist_path, dst)

    print(f"\n去畸变图已保存到: {os.path.join(output_dir, 'undistort')}")

else:
    print("没有任何一张图片成功检测到角点。")
