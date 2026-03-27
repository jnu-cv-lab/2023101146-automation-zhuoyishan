####任务一：使用OpenCV读取一张测试图片
#导入OpenCV库
import cv2
import matplotlib.pyplot as plt

#定义图片路径
image_name="test.jpg"
image_path=f"work01/images/{image_name}"

#通过OpenCV的imread的功能，根据定义的图片路径读取图片数据，并将读取的图片数据存到image里面
image=cv2.imread(image_path)

#判断图片是否读取成功
if image is None:
    print("无图片，图片读取失败")
else:
    print("图片读取成功")
    ####任务二：输出图片基本信息
    #图片的尺寸：高度，宽度
    height,width=image.shape[:2]
    #图片通道数：三个通道说明为彩色图,一个通道说明是灰度图
    if len(image.shape)==3:
        channels=image.shape[2]
    else:
        channels=1
    #像素数据类型
    pixel_type=image.dtype

    #在终端打印
    print(f"图片尺寸(宽度×高度):{height} × {width}")
    print(f"图像通道数：{channels}")
    print(f"像素数据类型：{pixel_type}")

    ####任务三：显示原图：用Matplotlib显示图片
    #排列图片，子图一
    plt.subplot(1,2,1)
    #将OpenCV读取的BGR格式转化为Matplotlib的RGB格式
    image_rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    #图片显示
    plt.imshow(image_rgb)
    #隐藏坐标轴
    plt.axis("off")
    #图片标题
    plt.title("picture")

    ####任务四：转换为灰度图并显示
    #排列图片，子图二
    plt.subplot(1,2,2)
    #将彩色图转换为灰度图
    image_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #用Matplotlib显示灰度图
    plt.imshow(image_gray,cmap='gray')
    plt.axis("off")
    plt.title("grayscale image")

    #图片显示
    plt.show()

    ####任务五：保存灰度图
    if image_gray is not None:
        #保存灰度图的文件名
        gray_save_name="test_gray.jpg"
        #保存灰度图的路径
        gray_save_path=f"work01/images/{gray_save_name}"

        #通过OpenCV的imwrite函数保存图片
        cv2.imwrite(gray_save_path,image_gray)
        print(f"灰度图保存成功，保存路径为：{gray_save_path}")
    else:
        print("灰度图生成失败，无法保存")

    ####任务六：用NumPY做操作
    #输出(50,50)位置的像素值
    pixel=image[50,50]
    print(f"位置(50,50)的像素值:{pixel}")

    #修改某一块像素值
    #复制原图（不修改原图）
    image_copy=image.copy()
    #把（15:25, 43:53）位置的颜色改成鲜绿色
    image_copy[15:25, 43:53]=[0,255,0]
    #定义保存路径
    pixel_change_save_name="pixel_change.jpg"
    pixel_change_save_path=f"work01/images/{pixel_change_save_name}"
    cv2.imwrite(pixel_change_save_path,image_copy)
    print(f"像素修改完成，保存路径为：{pixel_change_save_path}")
