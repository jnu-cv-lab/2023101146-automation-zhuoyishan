import cv2
import numpy as np

photo_path = f"/home/shi_chou_chu_jin/cv-course/work05/photo"
test_jpg = f"{photo_path}/test.jpg"


# 测试图
def make_test_img(size=600):
    img = np.ones((size, size, 3), np.uint8) * 255
    cv2.rectangle(img, (100, 100), (350, 350), (0, 0, 255), 3)
    cv2.circle(img, (450, 220), 70, (0, 255, 0), 3)
    # 平行线+垂直线
    for y in [120, 170, 220, 270]:
        cv2.line(img, (50, y), (550, y), (255, 0, 0), 2)
    for x in [120, 170, 220, 270]:
        cv2.line(img, (x, 50), (x, 550), (0, 0, 0), 2)
    cv2.line(img, (420, 100), (420, 350), (255, 0, 255), 3)
    cv2.line(img, (300, 220), (500, 220), (255, 0, 255), 3)
    return img


ori = make_test_img()
h, w = ori.shape[:2]

# 三种变换
# 相似变换
M_sim = cv2.getRotationMatrix2D((w / 2, h / 2), 20, 0.8)
sim = cv2.warpAffine(ori, M_sim, (w, h))

# 仿射变换
pts1 = np.float32([[50, 50], [500, 50], [50, 500]])
pts2 = np.float32([[100, 100], [450, 80], [80, 450]])
M_aff = cv2.getAffineTransform(pts1, pts2)
aff = cv2.warpAffine(ori, M_aff, (w, h))

# 透视变换
pts1_p = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
pts2_p = np.float32([[50, 50], [w - 50, 30], [30, h - 50], [w - 30, h - 30]])
M_per = cv2.getPerspectiveTransform(pts1_p, pts2_p)
per = cv2.warpPerspective(ori, M_per, (w, h))

row1 = np.hstack((ori, sim))
row2 = np.hstack((aff, per))
merge_all = np.vstack((row1, row2))
cv2.imwrite(f"{photo_path}/photo_1.jpg", merge_all)
print("测视图保存")

# 平面图形校正
img = cv2.imread(test_jpg)
img_preview = img.copy()
if img is None:
    print("找不到图片")
    exit()

H, W = img.shape[:2]

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (9, 9), 0)
edge = cv2.Canny(blur, 30, 120)
edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

# 纸轮廓
cnts = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
box = None

for c in cnts:
    approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
    if len(approx) == 4 and cv2.contourArea(approx) > W * H * 0.2:
        box = approx.reshape(4, 2).astype(np.float32)
        break


def sort_pts(p):
    rect = np.zeros((4, 2), np.float32)
    s = p.sum(1)
    rect[0] = p[np.argmin(s)]
    rect[2] = p[np.argmax(s)]
    d = np.diff(p, 1)
    rect[1] = p[np.argmin(d)]
    rect[3] = p[np.argmax(d)]
    return rect


if box is not None:
    box = sort_pts(box)
    cv2.polylines(img_preview, [box.astype(np.int32)], True, (0, 0, 255), 4)
    for x, y in box:
        cv2.circle(img_preview, (int(x), int(y)), 8, (255, 0, 0), -1)
else:
    box = np.float32(
        [
            [W * 0.05, H * 0.05],
            [W * 0.95, H * 0.05],
            [W * 0.95, H * 0.95],
            [W * 0.05, H * 0.95],
        ]
    )
    cv2.polylines(img_preview, [box.astype(np.int32)], True, (0, 0, 255), 4)

cv2.imwrite(f"{photo_path}/photo_2.jpg", img_preview)

tw, th = 700, int(700 * 1.414)
dst = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]])
M = cv2.getPerspectiveTransform(box, dst)
res = cv2.warpPerspective(img, M, (tw, th))
cv2.imwrite(f"{photo_path}/photo_3.jpg", res)
print("校正图保存")
