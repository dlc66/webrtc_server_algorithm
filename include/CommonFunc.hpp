#include <vector>
#include <string>
#include <opencv2/opencv.hpp>  
#include "global.h"
class CommonFunction {
public:
//判断点是否在矩形内部
static bool PointInRect(const std::vector<rect_name>& rect_vec, const cv::Point& point, std::string& re_name) 
{
    if (rect_vec.empty()) return true;
    bool inRect_flag = false;
    for (const auto& r : rect_vec) {
        const cv::Rect& rect = r.rect;
        if (point.x <= rect.x + rect.width && point.x >= rect.x &&
            point.y <= rect.y + rect.height && point.y >= rect.y) {
            inRect_flag = true;
            re_name = r.name; 
        }//中心点在矩形范围内
    }
    return inRect_flag;
}
//多标定框排序
static bool rectsort(std::vector<cv::Rect> &rect_from, std::vector<cv::Rect> &rect_to) {
    // 使用 std::sort 对 rect_from 按 x 坐标排序
    std::sort(rect_from.begin(), rect_from.end(), [](const cv::Rect &a, const cv::Rect &b) {
        return a.x < b.x;
    });
    // 清空 rect_to 以确保其大小为零
    rect_to.clear();
    rect_to.reserve(rect_from.size()); // 为了提高性能预先分配空间
    // 直接将排序后的 rect_from 添加到 rect_to
    for (const auto &rect : rect_from) {
        rect_to.push_back(rect);
    }
    return true;
}
//保证矩形在图像内
 static cv::Rect CorrectRect(cv::Rect roi_rect, int img_w, int img_h) 
 {
        // 校正 x 和 y 坐标
        if (roi_rect.x < 0) {
            roi_rect.x = 0;
        }
        if (roi_rect.y < 0) {
            roi_rect.y = 0;
        }

        // 校正宽度和高度
        if (roi_rect.x + roi_rect.width > img_w) {
            roi_rect.width = img_w - roi_rect.x;
        }
        if (roi_rect.y + roi_rect.height > img_h) {
            roi_rect.height = img_h - roi_rect.y;
        }

        // 处理负宽度和高度
        if (roi_rect.width < 0) {
            roi_rect.x = img_w;
            roi_rect.width = 0;
        }
        if (roi_rect.height < 0) {
            roi_rect.y = img_h;
            roi_rect.height = 0;
        }

        return roi_rect;
}

// 计算包含所有矩形的最小边界框-多框数字识别分类适配
static cv::Rect  ComputeBoundingBox(const std::vector<cv::Rect>& rects, int imageWidth, int imageHeight) {
    std::vector<cv::Rect> rect_cv = rects; // 复制传入的矩形
    // // 确保 rect_cv 不为空
    // if (rect_cv.empty()) {
    //     return cv::Rect(0, 0, 0, 0); // 返回一个空的矩形
    // }
    // 初始化边界坐标
    int left_x = rect_cv[0].x;
    int right_x = rect_cv[0].x + rect_cv[0].width;
    int top_y = rect_cv[0].y;
    int bottom_y = rect_cv[0].y + rect_cv[0].height;

    for (const auto &rect : rects) {
        cv::Rect tmp_roi_rect = CommonFunction::CorrectRect(rect, imageWidth, imageHeight);

        // 更新边界坐标
        if (left_x > tmp_roi_rect.x) {
            left_x = tmp_roi_rect.x;
        }
        if (right_x < tmp_roi_rect.x + tmp_roi_rect.width) {
            right_x = tmp_roi_rect.x + tmp_roi_rect.width;
        }
        if (top_y > tmp_roi_rect.y) {
            top_y = tmp_roi_rect.y;
        }
        if (bottom_y < tmp_roi_rect.y + tmp_roi_rect.height) {
            bottom_y = tmp_roi_rect.y + tmp_roi_rect.height;
        }
    }

    // 返回最小边界框
    return cv::Rect(left_x, top_y, right_x - left_x, bottom_y - top_y);
}


};