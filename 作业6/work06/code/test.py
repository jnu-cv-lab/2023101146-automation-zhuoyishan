import cv2
import numpy as np

img1_path = "/home/shi_chou_chu_jin/cv-course/work06/photo/box.png"  # 模板图
img2_path = "/home/shi_chou_chu_jin/cv-course/work06/photo/box_in_scene.png"  # 场景图

img1_color = cv2.imread(img1_path)  # 读取彩色模板图
img2_color = cv2.imread(img2_path)  # 读取彩色场景图

if img1_color is None or img2_color is None:
    print("错误:找不到图片")
    exit()

# ORB特征检测需使用灰度图，将模板图转为灰度图
img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)

h1, w1 = img1.shape[:2]  # 模板图的高和宽，用于过滤边缘点

# 任务1:ORB特征检测
# 1、使用ORB检测关键点与描述子
print("\n任务1:ORB特征检测")

# 创建ORB检测器
orb = cv2.ORB_create(
    nfeatures=1000,  # 检测1000个特征点
    scaleFactor=1.2,  # 尺度金字塔缩放系数
    nlevels=8,  # 金字塔层数
    edgeThreshold=31,  # 边缘阈值
    firstLevel=0,  # 金字塔起始层
    WTA_K=2,  # 计算描述子所用的点数量
    scoreType=cv2.ORB_HARRIS_SCORE,  # 用HARRIS评分对特征点排序
    patchSize=31,  # 计算描述子的窗口大小
    fastThreshold=20,  # FAST角点检测阈值
)

# 2、使用detectAndCompute()得到关键点和描述子
# kp1:关键点信息（位置、方向、大小）
# des1:描述子（特征的数字指纹）
kp1, des1 = orb.detectAndCompute(img1, None)  # 对模板图进行:检测关键点，计算描述子
kp2, des2 = orb.detectAndCompute(img2, None)  # 对场景图进行:检测关键点，计算描述子

# 在模板图上画出关键点
img1_kp = cv2.drawKeypoints(
    img1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)
# 在场景图上画出关键点
img2_kp = cv2.drawKeypoints(
    img2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

cv2.imwrite("/home/shi_chou_chu_jin/cv-course/work06/photo/task1_box_keypoints.jpg", img1_kp)
cv2.imwrite("/home/shi_chou_chu_jin/cv-course/work06/photo/task1_scene_keypoints.jpg", img2_kp)

# 模板图关键点数量
print(f"模板图关键点:{len(kp1)}")
# 输出场景图关键点数量
print(f"场景图关键点:{len(kp2)}")
# 输出描述子维度
print(f"ORB描述子维度:{des1.shape[1]}")

# 任务2:ORB特征匹配
# 对ORB描述子进行特征匹配
print("\n任务2:ORB特征匹配")

# 创建暴力匹配器BFMatcher
# ORB使用NORM_HAMMING
bf = cv2.BFMatcher(cv2.NORM_HAMMING)
# 进行KNN匹配，每个特征找k=2个最佳匹配
matches = bf.knnMatch(des1, des2, k=2)

# Lowe比率测试:过滤错误匹配
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# 过滤模板图边缘点
filtered_matches = []
for match in good_matches:
    x, y = kp1[match.queryIdx].pt
    if 0.15 * w1 < x < 0.85 * w1 and 0.15 * h1 < y < 0.85 * h1:
        filtered_matches.append(match)
good_matches = filtered_matches

# 按照匹配距离从小到大排序
good_matches = sorted(good_matches, key=lambda x: x.distance)

# 输出原始匹配数
print(f"总匹配数:{len(matches)}")
# 输出过滤后的优质匹配数
print(f"过滤后优质匹配数:{len(good_matches)}")

# 画出前50个匹配结果
img_matches = cv2.drawMatches(
    img1,
    kp1,
    img2,
    kp2,
    good_matches[:50],
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)
cv2.imwrite(
    "/home/shi_chou_chu_jin/cv-course/work06/photo/task2_orb_match_result.jpg", img_matches
)

# 任务3:RANSAC剔除错误匹配
# 用RANSAC计算单应矩阵，剔除错误匹配
print("\n任务3:RANSAC去噪")

# 至少需要4个点才能计算单应矩阵
if len(good_matches) < 4:
    print("有效匹配点不足，无法计算单应矩阵")
    exit()

# 提取模板图匹配点坐标，转为float32格式
pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
# 提取场景图匹配点坐标
pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# 用RANSAC方法计算单应矩阵H
H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 2.0)

# 将mask展平成列表
matchesMask = mask.ravel().tolist()
# 统计内点（正确匹配）数量
inlier_num = sum(matchesMask)
# 计算内点比例
inlier_ratio = inlier_num / len(good_matches)

# 画出RANSAC过滤后的内点匹配（绿色线）
img_ransac = cv2.drawMatches(
    img1,
    kp1,
    img2,
    kp2,
    good_matches,
    None,
    matchColor=(0, 255, 0),
    matchesMask=matchesMask,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)
cv2.imwrite(
    "/home/shi_chou_chu_jin/cv-course/work06/photo/task3_ransac_result.jpg", img_ransac
)

# 输出单应矩阵
print("单应矩阵 H:")
print(H)
# 输出总匹配数
print(f"优质匹配数:{len(good_matches)}")
# 输出内点数
print(f"RANSAC内点数:{inlier_num}")
# 输出内点比例
print(f"内点比例:{inlier_ratio:.2%}")

# 任务4:目标定位
# 用单应矩阵定位目标
print("\n任务4:目标定位")

# 获取模板图的高和宽
h, w = img1_color.shape[:2]
# 定义模板图的四个角点（左上、左下、右下、右上）
pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

# 用单应矩阵将四个角点投影到场景图中
dst = cv2.perspectiveTransform(pts, H)

# 在彩色场景图上画红色边框
img_result = img2_color.copy()
# 画出四边形边框
img_result = cv2.polylines(
    img_result, [np.int32(dst)], True, (0, 0, 255), 4, cv2.LINE_AA
)

cv2.imwrite(
    "/home/shi_chou_chu_jin/cv-course/work06/photo/task4_final_detection.jpg", img_result
)

# 任务6:参数对比实验
# 对比nfeatures=500/1000/2000
print("\n任务6:nfeatures参数对比")


# 定义函数:输入特征点数量n，返回实验数据
def run_orb(n):
    # 创建ORB检测器
    orb = cv2.ORB_create(nfeatures=n, scaleFactor=1.2, edgeThreshold=31, patchSize=31)
    # 检测特征点和描述子
    kp1_t, des1_t = orb.detectAndCompute(img1, None)
    kp2_t, des2_t = orb.detectAndCompute(img2, None)

    # 匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches_t = bf.knnMatch(des1_t, des2_t, k=2)

    # Lowe过滤
    good_t = []
    for m, n in matches_t:
        if m.distance < 0.70 * n.distance:
            good_t.append(m)

    # 边缘过滤
    filtered_t = []
    for match in good_t:
        x, y = kp1_t[match.queryIdx].pt
        if 0.15 * w1 < x < 0.85 * w1 and 0.15 * h1 < y < 0.85 * h1:
            filtered_t.append(match)
    good_t = filtered_t
    good_t = sorted(good_t, key=lambda x: x.distance)

    if len(good_t) < 4:
        return len(kp1_t), len(kp2_t), len(good_t), 0, 0.0, False

    # 坐标转换
    pts1_t = np.float32([kp1_t[m.queryIdx].pt for m in good_t]).reshape(-1, 1, 2)
    pts2_t = np.float32([kp2_t[m.trainIdx].pt for m in good_t]).reshape(-1, 1, 2)

    # RANSAC计算
    H_t, mask_t = cv2.findHomography(pts1_t, pts2_t, cv2.RANSAC, 2.0)
    inl_t = sum(mask_t.ravel()) if H_t is not None else 0
    ratio_t = inl_t / len(good_t) if len(good_t) > 0 else 0

    # 判断是否定位成功
    success = ratio_t > 0.3
    return len(kp1_t), len(kp2_t), len(good_t), inl_t, ratio_t, success


# 三组实验
n500 = run_orb(500)
n1000 = run_orb(1000)
n2000 = run_orb(2000)

# 对比表格
print("| nfeatures | 模板点数 | 场景点数 | 匹配数 | 内点数 | 内点比例 | 定位成功 |")
print("|-----------|----------|----------|--------|--------|----------|----------|")
print(
    f"| 500       | {n500[0]:<8} | {n500[1]:<8} | {n500[2]:<6} | {n500[3]:<6} | {n500[4]:<8.2%} | {'是' if n500[5] else '否'} |"
)
print(
    f"| 1000      | {n1000[0]:<8} | {n1000[1]:<8} | {n1000[2]:<6} | {n1000[3]:<6} | {n1000[4]:<8.2%} | {'是' if n1000[5] else '否'} |"
)
print(
    f"| 2000      | {n2000[0]:<8} | {n2000[1]:<8} | {n2000[2]:<6} | {n2000[3]:<6} | {n2000[4]:<8.2%} | {'是' if n2000[5] else '否'} |"
)

# SIFT匹配
# SIFT特征匹配
print("\nSIFT匹配:")
# 创建SIFT检测器
sift = cv2.SIFT_create()
# 检测特征点和描述子
kp_sift1, des_sift1 = sift.detectAndCompute(img1, None)
kp_sift2, des_sift2 = sift.detectAndCompute(img2, None)

# SIFT使用NORM_L2匹配
bf_sift = cv2.BFMatcher(cv2.NORM_L2)
matches_sift = bf_sift.knnMatch(des_sift1, des_sift2, k=2)

# Lowe比率测试
good = []
for m, n in matches_sift:
    if m.distance < 0.75 * n.distance:
        good.append(m)
good = sorted(good, key=lambda x: x.distance)

# 计算单应矩阵
if len(good) >= 4:
    pts_sift1 = np.float32([kp_sift1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_sift2 = np.float32([kp_sift2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H_sift, mask_sift = cv2.findHomography(pts_sift1, pts_sift2, cv2.RANSAC, 2.0)
    inl_sift = sum(mask_sift.ravel()) if H_sift is not None else 0
    ratio_sift = inl_sift / len(good) if len(good) > 0 else 0
else:
    inl_sift = 0
    ratio_sift = 0.0

# SIFT结果
print(f"SIFT匹配数:{len(good)}，内点:{inl_sift}，比例:{ratio_sift:.2%}")
