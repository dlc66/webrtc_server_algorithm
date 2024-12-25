#include <opencv2/opencv.hpp>
#include <algorithm>
//  算术均值滤波器 (参数支持滚动条调整)
cv::Mat mean_filter(const cv::Mat& img,int width,int height){
    // """使用滚动滑块调整均值滤波器参数.
    // Args:
    //     image_path (str): 传入图像文件的路径.
    //     width (int): . 滤波核宽度
    //     height (int):  滤波核高度
    // """
    //调整后的结果图，width与height分别表示滤波器核的宽度和高度
    cv::Mat result;
    cv::blur(img, result,cv::Size(std::max(width,1), std::max(height,1)));
    return result;
}
//  Box滤波器 (参数支持滚动条调整)
cv::Mat box_filter(const cv::Mat& img,int width,int height){
    //调整后的结果图，width与height分别表示滤波器核的宽度和高度
    cv::Mat result;
    cv::boxFilter(img, result,-1,cv::Size(std::max(width,1), std::max(height,1)));
    return result;
}
//  高斯滤波器 (参数支持滚动条调整)
cv::Mat gaussian_filter(const cv::Mat& img,int width,int height){
    //调整后的结果图，width与height分别表示滤波器核的宽度和高度
    cv::Mat result;
    cv::GaussianBlur(img, result,cv::Size(2*int(width/2)+1, 2*int(height/2)+1),0);
    return result;
}
//  双边滤波器 (参数支持滚动条调整)
cv::Mat bilateral_filter(const cv::Mat& img,int d,double color, double space){
    //调整后的结果图，width与height分别表示滤波器核的宽度和高度
    cv::Mat result;
    cv::bilateralFilter(img, result,d, color,space);
    return result;
}
//  定义修正阿尔法均值滤波函数
//  https://zhuanlan.zhihu.com/p/497902797另一个方法
cv::Mat alphaFilter(const cv::Mat& img, int d, int width, int height) {
    int m = (width - 1) / 2;
    int n = (height - 1) / 2;
    d = std::min({m, n, d});

    // 边缘填充
    cv::Mat border_filling;
    cv::copyMakeBorder(img, border_filling, m, m, n, n, cv::BORDER_REPLICATE);

    cv::Mat imgout = img.clone();

    // 修正阿尔法均值滤波
    for (int i = m; i < border_filling.rows - m; ++i) {
        for (int j = n; j < border_filling.cols - n; ++j) {
            // 读取模板下像素的灰度值
            std::vector<uchar> list;
            for (int s = i - m; s <= i + m; ++s) {
                for (int t = j - n; t <= j + n; ++t) {
                    list.push_back(border_filling.at<uchar>(s, t));
                }
            }
            // 选修正阿尔法均值作为输出图像模板中心像素的灰度值
            std::sort(list.begin(), list.end());
            int start_idx = d / 2;
            int end_idx = list.size() - d / 2;
            float sum = 0.0f;
            for (int k = start_idx; k < end_idx; ++k) {
                sum += list[k];
            }
            imgout.at<uchar>(i - m, j - n) = sum / (end_idx - start_idx);
        }
    }

    // 返回滤过图像
    return imgout;
}