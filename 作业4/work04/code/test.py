#导入库
import cv2          #用于图像处理、读取图片
import numpy as np  #用于数值计算、数组操作
import matplotlib.pyplot as plt  #用于画图、显示结果
plt.rcParams['axes.unicode_minus'] = False

## 第一部分：生成棋盘格和chirp测试图，下采样+抗混叠+FFT频谱
#1.1、生成棋盘格图像
size = 256 #定义图像的尺寸为256x256像素
chessboard = np.zeros((size, size), dtype=np.uint8) #创建一个全黑色的256x256图像数组
square_size = 16 #定义棋盘格中每个小格子的边长为16像素

#循环画白色格子
#遍历图像行，步长为32（两个格子宽度）
for i in range(0, size, square_size * 2):
    #遍历图像列，步长为32（两个格子宽度）
    for j in range(0, size, square_size * 2):
        #给当前位置的正方形区域赋值为255（白色）
        chessboard[i:i+square_size, j:j+square_size] = 255
        #给对角位置的正方形区域赋值为255（白色），形成棋盘图案
        chessboard[i+square_size:i+square_size*2, j+square_size:j+square_size*2] = 255

#1.2、生成 chirp 频率渐变图
x = np.linspace(-1, 1, size) #生成从-1到1的256个均匀数值，作为x轴坐标
y = np.linspace(-1, 1, size) #生成从-1到1的256个均匀数值，作为y轴坐标
X, Y = np.meshgrid(x, y) #将x、y一维数组转为二维网格矩阵，用于生成二维图像
R = np.sqrt(X**2 + Y**2) #计算每个像素点到图像中心的距离
#生成chirp信号：频率随半径线性增加，用sin函数生成明暗变化
chirp = np.sin(2 * np.pi * 20 * R**2) #公式：sin(2π * (频率系数) * R²)，R²让频率随半径平方增长，实现从中心到边缘频率越来越高
chirp = ((chirp - chirp.min()) / (chirp.max() - chirp.min()) * 255).astype(np.uint8) #把数值从[-1,1]映射到[0,255]，转成uint8格式（图像格式）

#2、直接下采样，观察混叠
down_factor = 4 #定义下采样倍数为4倍
#直接下采用：对棋盘格图像每隔4个像素取一个，实现4倍下采样
chess_direct = chessboard[::down_factor, ::down_factor]
chirp_direct = chirp[::down_factor, ::down_factor]

#3、加高斯滤波再下采样（抗混叠，对比差异）
sigma = 1.0 #高斯滤波的标准差，控制模糊程度
kernel = (5, 5) #高斯滤波核的大小为5x5
chess_blur = cv2.GaussianBlur(chessboard, kernel, sigma) #对棋盘格进行高斯模糊处理
chess_blur_down = chess_blur[::down_factor, ::down_factor] #对模糊后的棋盘格进行4倍下采样
chirp_blur = cv2.GaussianBlur(chirp, kernel, sigma) #对chirp图像进行高斯模糊处理
chirp_blur_down = chirp_blur[::down_factor, ::down_factor] #对模糊后的chirp图像进行4倍下采样

#4、画FFT频谱，频域上确认混叠消失
#定义一个计算并显示FFT频谱的函数
def show_fft(img, title):
    fft = np.fft.fft2(img) #对输入图像进行二维快速傅里叶变换
    fft_shift = np.fft.fftshift(fft) #将频谱的零频分量移到图像中心
    mag = 20 * np.log(np.abs(fft_shift) + 1) #计算频谱幅度，并做对数缩放
    plt.imshow(mag, cmap='gray') #显示频谱图，使用灰度颜色映射
    plt.title(title, fontsize=9) #字体大小为9号
    plt.axis('off')

#显示结果
plt.figure(figsize=(16, 8))
#显示棋盘格原图，使用灰度模式
plt.subplot(2,4,1)
plt.imshow(chessboard, cmap='gray')
plt.title('Chessboard Original', fontsize=9)
plt.axis('off')

#显示直接下采样的棋盘格
plt.subplot(2,4,2)
plt.imshow(chess_direct, cmap='gray')
plt.title('Direct Downsample', fontsize=9)
plt.axis('off')

#显示滤波后下采样的棋盘格
plt.subplot(2,4,3)
plt.imshow(chess_blur_down, cmap='gray')
plt.title('Blur + Downsample', fontsize=9)
plt.axis('off')

#显示chirp原图
plt.subplot(2,4,4)
plt.imshow(chirp, cmap='gray')
plt.title('Chirp Original', fontsize=9)
plt.axis('off')

#FFT频谱
#显示棋盘格原图的FFT频谱
plt.subplot(2,4,5)
show_fft(chessboard, 'Chessboard FFT')

#显示直接下采样的FFT频谱（混叠）
plt.subplot(2,4,6)
show_fft(chess_direct, 'Aliasing FFT')

#显示滤波后下采样的FFT频谱（无混叠）
plt.subplot(2,4,7)
show_fft(chess_blur_down, 'No Aliasing FFT')

#显示chirp图像的FFT频谱
plt.subplot(2,4,8)
show_fft(chirp, 'Chirp FFT')

plt.tight_layout()
plt.savefig("work04/photo/result1.jpg",dpi=300,bbox_inches='tight') #保存结果图像，分辨率300dpi，去除多余边距
plt.show()

## 第二部分：验证σ公式，固定M=4，测试不同σ
M = 4 #固定下采样倍数为4
sigma_theory = 0.45 * M #计算理论最优的sigma值
sigma_list = [0.5, 1.0, 2.0, 4.0] #定义要测试的4个sigma值

plt.figure(figsize=(20, 8))

#原图
plt.subplot(2,5,1)
#显示chirp原图
plt.imshow(chirp, cmap='gray')
plt.title('Original Chirp')
plt.axis('off')

#显示chirp原图的频谱
plt.subplot(2,5,6)
show_fft(chirp, 'Original FFT')

#循环显示不同sigma
#遍历sigma列表，同时获取索引和值
for idx, s in enumerate(sigma_list):
    blur = cv2.GaussianBlur(chirp, (11,11), s) #使用当前sigma对chirp进行高斯模糊
    down = blur[::M, ::M] #对模糊图像进行4倍下采样
    plt.subplot(2,5, idx+2) #选中对应位置的子图
    plt.imshow(down, cmap='gray') #显示下采样结果
    plt.title(f'sigma={s}') #设置标题，显示当前sigma值
    plt.axis('off')
    plt.subplot(2,5, idx+2+5) #选中下一行对应位置的子图
    show_fft(down, f'FFT sigma={s}') #显示当前下采样结果的FFT频谱

plt.tight_layout()
plt.savefig("work04/photo/result2.jpg", dpi=300, bbox_inches='tight')
plt.show()

print(f'Best sigma = 0.45 * {M} = {sigma_theory}') #理论最优sigma值

## 第三部分：自适应下采样
#1、用梯度分析估计局部M值
#计算梯度
gx = cv2.Sobel(chirp, cv2.CV_64F, 1, 0, ksize=3) #计算x方向梯度，检测水平边缘
gy = cv2.Sobel(chirp, cv2.CV_64F, 0, 1, ksize=3) #计算y方向梯度，检测垂直边缘
grad = np.sqrt(gx**2 + gy**2) #计算梯度幅值，代表图像边缘强度
grad = (grad - grad.min()) / (grad.max() - grad.min()) #将梯度值归一化到0~1范围，方便后续计算

#根据梯度计算局部M
min_M = 2 #最小下采样倍数
max_M = 4 #最大下采样倍数
local_M = max_M - (max_M - min_M) * grad #梯度越大，下采样倍数越小；梯度越小，倍数越大
local_M = np.round(local_M).astype(np.uint8)

#2、对不同区域用不同σ滤波
#自适应滤波
adaptive_blur = np.zeros_like(chirp, dtype=np.float32)
h, w = chirp.shape
#遍历图像每一行
for i in range(h):
    #遍历图像每一列
    for j in range(w):
        m = local_M[i, j] #获取当前像素的局部下采样倍数
        sigma = 0.45 * m #根据下采样倍数计算对应的sigma值
        k = max(3, int(2 * np.ceil(3*sigma)+1)) #计算高斯核大小，保证为奇数且不小于3
        blur = cv2.GaussianBlur(chirp, (k,k), sigma) #对图像进行自适应高斯模糊
        adaptive_blur[i,j] = blur[i,j] #将当前像素的模糊结果存入数组

adaptive_blur = adaptive_blur.astype(np.uint8) #将自适应滤波结果转为图像格式

#自适应下采样
#分块下采样
block = 16 #定义分块大小为16x16
adaptive_result = np.zeros_like(chirp) #创建空数组，存储最终自适应下采样结果

#按块遍历图像行
for i in range(0, h, block):
    #按块遍历图像列
    for j in range(0, w, block):
        patch = adaptive_blur[i:i+block, j:j+block] #取出当前16x16的图像块
        m_patch = int(np.mean(local_M[i:i+block, j:j+block])) #计算当前块的平均下采样倍数
        small = patch[::m_patch, ::m_patch] #对图像块进行自适应倍数下采样
        up = cv2.resize(small, (block, block), interpolation=cv2.INTER_NEAREST) #将下采样后的小块放大回16x16
        adaptive_result[i:i+block, j:j+block] = up #将处理后的块放回结果图像对应位置

#统一下采样对比
uni_blur = cv2.GaussianBlur(chirp, (11,11), sigma_theory) #使用理论最优sigma进行高斯模糊
uni_down = uni_blur[::4, ::4] #统一进行4倍下采样
uni_up = cv2.resize(uni_down, (w, h), interpolation=cv2.INTER_NEAREST) #将下采样结果放大回原图大小

#3、误差图
err_adp = cv2.absdiff(chirp, adaptive_result) #计算自适应结果与原图的绝对误差
err_uni = cv2.absdiff(chirp, uni_up) #计算统一下采样结果与原图的绝对误差

#显示结果
plt.figure(figsize=(15,10))

#自适应下采样结果
plt.subplot(2,3,1)
plt.imshow(adaptive_result, cmap='gray')
plt.title('Adaptive Downsample')
plt.axis('off')

#统一下采样结果
plt.subplot(2,3,2)
plt.imshow(uni_up, cmap='gray')
plt.title('Uniform 4x Downsample')
plt.axis('off')

#显示原图
plt.subplot(2,3,3)
plt.imshow(chirp, cmap='gray')
plt.title('Original')
plt.axis('off')

#显示自适应误差图
plt.subplot(2,3,4)
plt.imshow(err_adp, cmap='hot')
plt.title('Adaptive Error')
plt.axis('off')

#显示统一下采样误差图
plt.subplot(2,3,5)
plt.imshow(err_uni, cmap='hot')
plt.title('Uniform Error')
plt.axis('off')

#显示局部下采样倍数分布图
plt.subplot(2,3,6)
plt.imshow(local_M)
plt.title('Local M Map')
plt.axis('off')

plt.tight_layout()
plt.savefig("work04/photo/result3.png", dpi=300, bbox_inches='tight')
plt.show()

#自适应下采样的平均误差
print(f'Avg adaptive error: {np.mean(err_adp):.2f}')
#统一下采样的平均误差
print(f'Avg uniform error: {np.mean(err_uni):.2f}')