#include <iostream>
//导入OpenCV库
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main()
{
    ////任务一：使用OpenCV读取一张测试图片
    //定义图片路径
    string image_name = "test.jpg";
    string image_path = "/home/administrator/myproj/work02/photo/" + image_name;
    Mat image = imread(image_path);

    //判断图片是否读取成功
    if (image.empty())
    {
        cout << "无图片，图片读取失败" << endl;
        return -1;
    }
    cout << "图片读取成功" << endl;

    ////任务二：输出图片基本信息
    int height = image.rows;
    int width = image.cols;
    int channels = image.channels();
    string pixel_type = "uint8 (CV_8U)";

    cout << "图片尺寸(宽度×高度): " << width << " × " << height << endl;
    cout << "图像通道数：" << channels << endl;
    cout << "像素数据类型：" << pixel_type << endl;

    //任务三：显示原图
    namedWindow("原图", WINDOW_NORMAL);//建立一个图片窗口，名字为“原图”
    resizeWindow("原图", 800, 600);//强制窗口大小
    imshow("原图", image);//显示原图
    cout << "请查看窗口，按任意键继续..." << endl;
    waitKey(0);
    destroyWindow("原图");//关闭窗口

    //任务四：灰度图并显示
    Mat image_gray;
    cvtColor(image, image_gray, COLOR_BGR2GRAY); 
    namedWindow("灰度图", WINDOW_NORMAL);//建立一个图片窗口，名字为“灰度图”
    resizeWindow("灰度图", 800, 600);//强制窗口大小
    imshow("灰度图", image_gray);//显示灰度图
    cout << "请查看窗口，按任意键继续..." << endl;
    waitKey(0);
    destroyWindow("灰度图");//关闭窗口

    //任务五：保存灰度图
    string gray_save_name = "test_gray.jpg";
    string gray_save_path = "/home/administrator/myproj/work02/photo/" + gray_save_name;
    imwrite(gray_save_path, image_gray);
    cout << "灰度图保存成功，路径：" << gray_save_path << endl;

    //任务六：像素操作
    Vec3b pixel = image.at<Vec3b>(50, 50);
    cout << "位置(50,50)的像素值: " 
         << (int)pixel[0] << " " 
         << (int)pixel[1] << " " 
         << (int)pixel[2] << endl;

    //修改一块像素为绿色
    Mat image_copy = image.clone();
    for (int y = 15; y < 25; y++)
    {
        for (int x = 43; x < 53; x++)
        {
            image_copy.at<Vec3b>(y, x) = Vec3b(0, 255, 0);
        }
    }

    string pixel_change_save_name = "pixel_change.jpg";
    string pixel_change_save_path = "/home/administrator/myproj/work02/photo/" + pixel_change_save_name;
    imwrite(pixel_change_save_path, image_copy);
    cout << "像素修改完成，保存路径：" << pixel_change_save_path << endl;

    return 0;
}