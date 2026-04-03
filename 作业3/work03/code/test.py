import cv2 #处理图像
import numpy as np #做数学计算
import matplotlib.pyplot as plt #画图显示
import os #存储图像

##计算两张图的差距，均方差MSE越大，则差别越大。峰值信噪比PSNR越大，说明图片质量越好。
def compute_mse_psnr(img1, img2):
    #计算MSE
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:#如果两张图一样，则MSE为0，PSNR无穷大
        return 0, float('inf')
    psnr = 10 * np.log10((255 ** 2) / mse)#计算PSNR
    return mse, psnr

##进行傅里叶变换，得到频谱图。分辨低频（大块颜色）和高频（边缘、细节）
def fft2_shift_log(img):
    f = np.fft.fft2(img)#二维傅里叶变换
    f_shift = np.fft.fftshift(f)#把低频转移到图像中心
    magnitude = 20 * np.log(np.abs(f_shift) + 1)#计算幅度，并用对数压缩，让图片更清晰
    return magnitude

##做DCT变换，得到频域表示。计算低频能量占比，看看图像的主要信息集中在哪些频率上。
def dct2_energy(img):
    h, w = img.shape#获取图像尺寸
    dct = cv2.dct(img.astype(np.float32))#DCT变换
    total_energy = np.sum(dct ** 2)#总能量
    #取左上角1/4区域作为低频部分
    low_freq_h, low_freq_w = h // 4, w // 4
    low_energy = np.sum(dct[:low_freq_h, :low_freq_w] ** 2)
    
    energy_ratio = low_energy / total_energy#低频能量占比
    dct_log = 20 * np.log(np.abs(dct) + 1)
    return dct_log, energy_ratio

#1、图像读入和预处理
##读取图片
script_dir = os.path.dirname(os.path.abspath(__file__))#获取code目录
photo_dir = os.path.join(script_dir, "../photo")#定位到photo目录
img_path = os.path.join(photo_dir, "test.jpg")#图片路径

#读取灰度图
img_original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img_original is None:
    raise FileNotFoundError(f"图片不存在：{img_path}")

#原图尺寸
h, w = img_original.shape
print(f"原始图像尺寸：{w}x{h}")

#2、下采样
scale = 0.5#缩小为原来的1/2
new_h, new_w = int(h * scale), int(w * scale)#图片缩小后的尺寸
#直接缩小
img_down_direct = cv2.resize(img_original, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
#高斯滤波后缩小
img_blur = cv2.GaussianBlur(img_original, (5,5), 1.0)#5×5高斯核，sigma=1.0
img_down_gauss = cv2.resize(img_blur, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

#3、图片恢复
#将缩小后的图像恢复到原始尺寸，使用三种插值方法
img_up_nearest = cv2.resize(img_down_direct, (w, h), interpolation=cv2.INTER_NEAREST)#最近邻内插
img_up_bilinear = cv2.resize(img_down_direct, (w, h), interpolation=cv2.INTER_LINEAR)#双线性内插
img_up_bicubic = cv2.resize(img_down_direct, (w, h), interpolation=cv2.INTER_CUBIC)#双三次内插

#4、空间域比较
#计算三种恢复方法与原图的MSE、PSNR
mse_n, psnr_n = compute_mse_psnr(img_original, img_up_nearest)
mse_bi, psnr_bi = compute_mse_psnr(img_original, img_up_bilinear)
mse_bic, psnr_bic = compute_mse_psnr(img_original, img_up_bicubic)

#空间域对比结果
print("\n4、空间域比较结果：\n")
print("直接缩小后恢复的MSE和PSNR：")
print(f"最近邻插值: MSE={mse_n:.2f}, PSNR={psnr_n:.2f} dB")
print(f"双线性插值: MSE={mse_bi:.2f}, PSNR={psnr_bi:.2f} dB")
print(f"双三次插值: MSE={mse_bic:.2f}, PSNR={psnr_bic:.2f} dB")

#空间域对比图：原图、缩小图、恢复图
save1 = os.path.join(photo_dir, "result_spatial.png")#空间域结果保存
plt.figure(figsize=(15, 10))
#第一行：原图、直接缩小图、高斯平滑后缩小图
plt.subplot(2, 4, 1);
plt.imshow(img_original, cmap='gray'); plt.title('Original'); plt.axis('off')
plt.subplot(2, 4, 2);
plt.imshow(img_down_direct, cmap='gray'); plt.title('Down Direct'); plt.axis('off')
plt.subplot(2, 4, 3);
plt.imshow(img_down_gauss, cmap='gray'); plt.title('Down Gauss'); plt.axis('off')
#第二行：三种插值恢复图（标注PSNR）
plt.subplot(2, 4, 5);
plt.imshow(img_up_nearest, cmap='gray'); plt.title(f'Nearest\nPSNR={psnr_n:.1f}'); plt.axis('off')
plt.subplot(2, 4, 6);
plt.imshow(img_up_bilinear, cmap='gray'); plt.title(f'Bilinear\nPSNR={psnr_bi:.1f}'); plt.axis('off')
plt.subplot(2, 4, 7);
plt.imshow(img_up_bicubic, cmap='gray'); plt.title(f'Bicubic\nPSNR={psnr_bic:.1f}'); plt.axis('off')
plt.tight_layout()
plt.savefig(save1, dpi=300, bbox_inches='tight')#空间域对比图

#5、傅里叶变换分析
#分别对原图、缩小后图像、双线性恢复后图像做二维傅里叶变换
fft_original = fft2_shift_log(img_original)
fft_down = fft2_shift_log(img_down_direct)
fft_restore = fft2_shift_log(img_up_bilinear)

#频谱图
save2 = os.path.join(photo_dir, "result_fft.png")#傅里叶频谱图保存
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1); plt.imshow(fft_original, cmap='gray'); plt.title('Original FFT'); plt.axis('off')
plt.subplot(1, 3, 2); plt.imshow(fft_down, cmap='gray'); plt.title('Down FFT'); plt.axis('off')
plt.subplot(1, 3, 3); plt.imshow(fft_restore, cmap='gray'); plt.title('Restore FFT'); plt.axis('off')
plt.tight_layout()
plt.savefig(save2, dpi=300, bbox_inches='tight')#保存傅里叶频谱图

#6、DCT分析
#分别对原图和三种恢复图做二维DCT，计算低频能量占比
dct_original, ratio_o = dct2_energy(img_original)
dct_n, ratio_n = dct2_energy(img_up_nearest)
dct_bi, ratio_bi = dct2_energy(img_up_bilinear)
dct_bic, ratio_bic = dct2_energy(img_up_bicubic)

#打印DCT低频能量占比结果
print("\n6、DCT分析结果\n")
print("左上角1/4低频区域能量占总能量的比例：")
print(f"原图:    {ratio_o:.4f}")
print(f"最近邻:  {ratio_n:.4f}")
print(f"双线性:  {ratio_bi:.4f}")
print(f"双三次:  {ratio_bic:.4f}")

#绘制DCT系数图
save3 = os.path.join(photo_dir, "result_dct.png")#DCT系数图保存
plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1); plt.imshow(dct_original, cmap='gray'); plt.title(f'Original DCT\n{ratio_o:.4f}'); plt.axis('off')
plt.subplot(2, 2, 2); plt.imshow(dct_n, cmap='gray'); plt.title(f'Nearest DCT\n{ratio_n:.4f}'); plt.axis('off')
plt.subplot(2, 2, 3); plt.imshow(dct_bi, cmap='gray'); plt.title(f'Bilinear DCT\n{ratio_bi:.4f}'); plt.axis('off')
plt.subplot(2, 2, 4); plt.imshow(dct_bic, cmap='gray'); plt.title(f'Bicubic DCT\n{ratio_bic:.4f}'); plt.axis('off')
plt.tight_layout()
plt.savefig(save3, dpi=300, bbox_inches='tight')#保存DCT系数图

#显示所有图像
plt.show()