# 计算机视觉课程-作业2
- 姓名：卓宜姗
- 学号：2023101146
- 专业：自动化

# 作业目标
基础图像操作：
使用 C++ 与 OpenCV 完成基础图像处理：
1. 读取一张测试图片
2. 输出图片基本信息（尺寸、通道数、像素类型）
3. 显示原图
4. 转换为灰度图并显示
5. 保存灰度图
6. 像素操作：读取指定位置像素值，修改区域像素并保存结果

# 项目结构
MYPROJ
 ├── .vscode
    ├── c_cpp_properties.json
    ├── launch.json
    └── tasks.json
 └── work02
    ├── build
        └── main
    ├── code
        ├── task.cpp
        └── task
    └── photo
        ├── test.jpg 输入原始图片
        ├── test_gray.jpg 灰度化后的图片
        └── pixel_change.jpg #像素修改后的图片

# 运行说明
- 操作系统：WSL Ubuntu 24.04
- 编译器：g++
- 核心库：OpenCV 4
- 开发工具：VS Code + C/C++ 扩展

# 
默认情况下，`tasks.json` 只编译单个源文件 
若要编译多个 `.cpp` 文件（如 `main.cpp`、`utils.cpp` 等），需修改 `args` 参数，将 `"${file}"` 替换为需要编译的所有源文件，并指定输出的可执行文件名。