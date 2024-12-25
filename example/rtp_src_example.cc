extern "C" {
#include <libavcodec/bsf.h>
#include <libavutil/pixfmt.h>
#include <libavformat/avformat.h>
#include <libavutil/opt.h>
#include <libavutil/timestamp.h>
#include <libswscale/swscale.h>
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libavcodec/avcodec.h>
}
#include <jpeglib.h>
#include <cstdlib>
//imread 和 imwrite在c++调用python慎用
#include <fstream>
#include <json.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <map>
#include <mutex>
#include <thread>
#include <bitset>
#include "common/utils.h"
#include "muduo/base/Logging.h"
#include "muduo/net/EventLoop.h"
#include "muduo/net/http/HttpRequest.h"
#include "muduo/net/http/HttpResponse.h"
#include "muduo/net/http/HttpServer.h"
#include "net/udp_connection.h"
#include "net/udp_server.h"
#include "rtc/dtls_transport.h"
#include "rtc/srtp_session.h"
#include "rtc/stun_packet.h"
#include "rtc/transport_interface.h"
#include "rtc/webrtc_transport.h"
#include "session/webrtc_session.h"
#include <iostream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <semaphore.h>
#include <python3.8/Python.h>
#include <numpy/ndarrayobject.h>
#include <thread>
#include <functional>
//加入检测模型
#include "chrono"
#include <experimental/filesystem>
// #include "yolov8.hpp"
using namespace muduo;
using namespace muduo::net;
std::mutex mtx;
//共享内存
// #include "shm_buf.h"
// URL 解码函数
std::string urlDecode(const std::string &value) {
    std::ostringstream decoded;
    for (size_t i = 0; i < value.length(); ++i) {
        if (value[i] == '%' && i + 2 < value.length()) {
            std::istringstream iss(value.substr(i + 1, 2));
            int hexValue;
            iss >> std::hex >> hexValue;
            decoded << static_cast<char>(hexValue);
            i += 2;
        } else if (value[i] == '+') {
            decoded << ' ';
        } else {
            decoded << value[i];
        }
    }
    return decoded.str();
}
// bool YUV2RGB(uchar* pYuvBuf, int nWidth, int nHeight, int channels, uchar* pRgbBuf)
// {
// #define PIXELSIZE nWidth * nHeight  
//     const int Table_fv1[256] = { -180, -179, -177, -176, -174, -173, -172, -170, -169, -167, -166, -165, -163, -162, -160, -159, -158, -156, -155, -153, -152, -151, -149, -148, -146, -145, -144, -142, -141, -139, -138, -137, -135, -134, -132, -131, -130, -128, -127, -125, -124, -123, -121, -120, -118, -117, -115, -114, -113, -111, -110, -108, -107, -106, -104, -103, -101, -100, -99, -97, -96, -94, -93, -92, -90, -89, -87, -86, -85, -83, -82, -80, -79, -78, -76, -75, -73, -72, -71, -69, -68, -66, -65, -64, -62, -61, -59, -58, -57, -55, -54, -52, -51, -50, -48, -47, -45, -44, -43, -41, -40, -38, -37, -36, -34, -33, -31, -30, -29, -27, -26, -24, -23, -22, -20, -19, -17, -16, -15, -13, -12, -10, -9, -8, -6, -5, -3, -2, 0, 1, 2, 4, 5, 7, 8, 9, 11, 12, 14, 15, 16, 18, 19, 21, 22, 23, 25, 26, 28, 29, 30, 32, 33, 35, 36, 37, 39, 40, 42, 43, 44, 46, 47, 49, 50, 51, 53, 54, 56, 57, 58, 60, 61, 63, 64, 65, 67, 68, 70, 71, 72, 74, 75, 77, 78, 79, 81, 82, 84, 85, 86, 88, 89, 91, 92, 93, 95, 96, 98, 99, 100, 102, 103, 105, 106, 107, 109, 110, 112, 113, 114, 116, 117, 119, 120, 122, 123, 124, 126, 127, 129, 130, 131, 133, 134, 136, 137, 138, 140, 141, 143, 144, 145, 147, 148, 150, 151, 152, 154, 155, 157, 158, 159, 161, 162, 164, 165, 166, 168, 169, 171, 172, 173, 175, 176, 178 };
//     const int Table_fv2[256] = { -92, -91, -91, -90, -89, -88, -88, -87, -86, -86, -85, -84, -83, -83, -82, -81, -81, -80, -79, -78, -78, -77, -76, -76, -75, -74, -73, -73, -72, -71, -71, -70, -69, -68, -68, -67, -66, -66, -65, -64, -63, -63, -62, -61, -61, -60, -59, -58, -58, -57, -56, -56, -55, -54, -53, -53, -52, -51, -51, -50, -49, -48, -48, -47, -46, -46, -45, -44, -43, -43, -42, -41, -41, -40, -39, -38, -38, -37, -36, -36, -35, -34, -33, -33, -32, -31, -31, -30, -29, -28, -28, -27, -26, -26, -25, -24, -23, -23, -22, -21, -21, -20, -19, -18, -18, -17, -16, -16, -15, -14, -13, -13, -12, -11, -11, -10, -9, -8, -8, -7, -6, -6, -5, -4, -3, -3, -2, -1, 0, 0, 1, 2, 2, 3, 4, 5, 5, 6, 7, 7, 8, 9, 10, 10, 11, 12, 12, 13, 14, 15, 15, 16, 17, 17, 18, 19, 20, 20, 21, 22, 22, 23, 24, 25, 25, 26, 27, 27, 28, 29, 30, 30, 31, 32, 32, 33, 34, 35, 35, 36, 37, 37, 38, 39, 40, 40, 41, 42, 42, 43, 44, 45, 45, 46, 47, 47, 48, 49, 50, 50, 51, 52, 52, 53, 54, 55, 55, 56, 57, 57, 58, 59, 60, 60, 61, 62, 62, 63, 64, 65, 65, 66, 67, 67, 68, 69, 70, 70, 71, 72, 72, 73, 74, 75, 75, 76, 77, 77, 78, 79, 80, 80, 81, 82, 82, 83, 84, 85, 85, 86, 87, 87, 88, 89, 90, 90 };
//     const int Table_fu1[256] = { -44, -44, -44, -43, -43, -43, -42, -42, -42, -41, -41, -41, -40, -40, -40, -39, -39, -39, -38, -38, -38, -37, -37, -37, -36, -36, -36, -35, -35, -35, -34, -34, -33, -33, -33, -32, -32, -32, -31, -31, -31, -30, -30, -30, -29, -29, -29, -28, -28, -28, -27, -27, -27, -26, -26, -26, -25, -25, -25, -24, -24, -24, -23, -23, -22, -22, -22, -21, -21, -21, -20, -20, -20, -19, -19, -19, -18, -18, -18, -17, -17, -17, -16, -16, -16, -15, -15, -15, -14, -14, -14, -13, -13, -13, -12, -12, -11, -11, -11, -10, -10, -10, -9, -9, -9, -8, -8, -8, -7, -7, -7, -6, -6, -6, -5, -5, -5, -4, -4, -4, -3, -3, -3, -2, -2, -2, -1, -1, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 11, 11, 11, 12, 12, 12, 13, 13, 13, 14, 14, 14, 15, 15, 15, 16, 16, 16, 17, 17, 17, 18, 18, 18, 19, 19, 19, 20, 20, 20, 21, 21, 22, 22, 22, 23, 23, 23, 24, 24, 24, 25, 25, 25, 26, 26, 26, 27, 27, 27, 28, 28, 28, 29, 29, 29, 30, 30, 30, 31, 31, 31, 32, 32, 33, 33, 33, 34, 34, 34, 35, 35, 35, 36, 36, 36, 37, 37, 37, 38, 38, 38, 39, 39, 39, 40, 40, 40, 41, 41, 41, 42, 42, 42, 43, 43 };
//     const int Table_fu2[256] = { -227, -226, -224, -222, -220, -219, -217, -215, -213, -212, -210, -208, -206, -204, -203, -201, -199, -197, -196, -194, -192, -190, -188, -187, -185, -183, -181, -180, -178, -176, -174, -173, -171, -169, -167, -165, -164, -162, -160, -158, -157, -155, -153, -151, -149, -148, -146, -144, -142, -141, -139, -137, -135, -134, -132, -130, -128, -126, -125, -123, -121, -119, -118, -116, -114, -112, -110, -109, -107, -105, -103, -102, -100, -98, -96, -94, -93, -91, -89, -87, -86, -84, -82, -80, -79, -77, -75, -73, -71, -70, -68, -66, -64, -63, -61, -59, -57, -55, -54, -52, -50, -48, -47, -45, -43, -41, -40, -38, -36, -34, -32, -31, -29, -27, -25, -24, -22, -20, -18, -16, -15, -13, -11, -9, -8, -6, -4, -2, 0, 1, 3, 5, 7, 8, 10, 12, 14, 15, 17, 19, 21, 23, 24, 26, 28, 30, 31, 33, 35, 37, 39, 40, 42, 44, 46, 47, 49, 51, 53, 54, 56, 58, 60, 62, 63, 65, 67, 69, 70, 72, 74, 76, 78, 79, 81, 83, 85, 86, 88, 90, 92, 93, 95, 97, 99, 101, 102, 104, 106, 108, 109, 111, 113, 115, 117, 118, 120, 122, 124, 125, 127, 129, 131, 133, 134, 136, 138, 140, 141, 143, 145, 147, 148, 150, 152, 154, 156, 157, 159, 161, 163, 164, 166, 168, 170, 172, 173, 175, 177, 179, 180, 182, 184, 186, 187, 189, 191, 193, 195, 196, 198, 200, 202, 203, 205, 207, 209, 211, 212, 214, 216, 218, 219, 221, 223, 225 };
//     int len = channels * nWidth * nHeight;
//     if (!pYuvBuf || !pRgbBuf)
//         return false;
//     const long nYLen = long(PIXELSIZE);
//     const int nHfWidth = (nWidth >> 1);
//     if (nYLen<1 || nHfWidth<1)
//         return false;
//     // Y data  
//     unsigned char* yData = pYuvBuf;
//     // v data  
//     unsigned char* vData = &yData[nYLen];
//     // u data  
//     unsigned char* uData = &vData[nYLen >> 2];
//     if (!uData || !vData)
//         return false;
//     int rgb[3];
//     int i, j, m, n, x, y, pu, pv, py, rdif, invgdif, bdif;
//     m = -nWidth;
//     n = -nHfWidth;
//     bool addhalf = true;
//     for (y = 0; y<nHeight; y++) {
//         m += nWidth;
//         if (addhalf) {
//             n += nHfWidth;
//             addhalf = false;
//         }
//         else {
//             addhalf = true;
//         }
//         for (x = 0; x<nWidth; x++) {
//             i = m + x;
//             j = n + (x >> 1);
//             py = yData[i];
//             // search tables to get rdif invgdif and bidif  
//             rdif = Table_fv1[vData[j]];    // fv1  
//             invgdif = Table_fu1[uData[j]] + Table_fv2[vData[j]]; // fu1+fv2  
//             bdif = Table_fu2[uData[j]]; // fu2       
//             rgb[0] = py + rdif;    // R  
//             rgb[1] = py - invgdif; // G  
//             rgb[2] = py + bdif;    // B  
//             j = nYLen - nWidth - m + x;
//             i = (j << 1) + j;
//             // copy this pixel to rgb data  
//             for (j = 0; j<3; j++)
//             {
//                 if (rgb[j] >= 0 && rgb[j] <= 255) {
//                     pRgbBuf[i + j] = rgb[j];
//                 }
//                 else {
//                     pRgbBuf[i + j] = (rgb[j] < 0) ? 0 : 255;
//                 }
//             }
//         }
//     }
//     return true;
// }
// void AVFrame2Img(AVFrame *pFrame, cv::Mat& img)
// {
//     int frameHeight = pFrame->height;
//     int frameWidth = pFrame->width;
//     int channels = 3;
//     //输出图像分配内存
//     img = cv::Mat::zeros(frameHeight, frameWidth, CV_8UC3);
//     //反转图像
//     pFrame->data[0] += pFrame->linesize[0] * (frameHeight - 1);
//     pFrame->linesize[0] *= -1;
//     pFrame->data[1] += pFrame->linesize[1] * (frameHeight / 2 - 1);
//     pFrame->linesize[1] *= -1;
//     pFrame->data[2] += pFrame->linesize[2] * (frameHeight / 2 - 1);
//     pFrame->linesize[2] *= -1;

//     //创建保存yuv数据的buffer
//     uchar* pDecodedBuffer = (uchar*)malloc(frameHeight*frameWidth * sizeof(uchar)*channels);

//     //从AVFrame中获取yuv420p数据，并保存到buffer
//     int i, j, k;
//     //拷贝y分量
//     for (i = 0; i < frameHeight; i++)
//     {
//         memcpy(pDecodedBuffer + frameWidth*i,
//             pFrame->data[0] + pFrame->linesize[0] * i,
//             frameWidth);
//     }
//     //拷贝u分量
//     for (j = 0; j < frameHeight / 2; j++)
//     {
//         memcpy(pDecodedBuffer + frameWidth*i + frameWidth / 2 * j,
//             pFrame->data[1] + pFrame->linesize[1] * j,
//             frameWidth / 2);
//     }
//     //拷贝v分量
//     for (k = 0; k < frameHeight / 2; k++)
//     {
//         memcpy(pDecodedBuffer + frameWidth*i + frameWidth / 2 * j + frameWidth / 2 * k,
//             pFrame->data[2] + pFrame->linesize[2] * k,
//             frameWidth / 2);
//     }

//     //将buffer中的yuv420p数据转换为RGB;
//     YUV2RGB(pDecodedBuffer, frameWidth, frameHeight, channels, img.data);

//     //释放buffer
//     free(pDecodedBuffer);
// }
//shit方法，慢的一批
// 定义一个自定义结构体
#include <vector>
//记录超过40ms的算法的帧数
int frame_number=0;
struct Algorithm {
    std::string name;
    int height;
    int width;
    int d;
    int color;
    int space;
    int lights;
    float contrasts;
    int r_1;
    int r_2;
    int g_1;
    int g_2;
    int b_1;
    int b_2;
    float limits;
    int grids;
    int ratios;
    int radii;
    std::string tracker_name;
    // 构造函数
    Algorithm(std::string name,int height=0, int width =0,int d =0,int color =0, int space =0 ,int lights=0, float contrasts=0,int r_1=0,int r_2=0,int g_1=0,int g_2=0,int b_1=0,int b_2=0,float limits=0,int grids=0,int ratios=0,int radii=0,std::string tracker_name="") 
    : name(name),height(height), width(width),d(d),color(color),space(space),lights(lights),contrasts(contrasts),r_1(r_1),r_2(r_2),g_1(g_1),g_2(g_2),b_1(b_1),b_2(b_2),limits(limits),grids(grids),ratios(ratios),radii(radii),tracker_name(tracker_name) {}
};
const std::vector<std::string> CLASS_NAMES = //{"qhkg_s"};
{"people","sly_dmyw","pzqcd","jyhbx","drqgd","jyz_pl","yxdgsg","jdyxxsd","ws_ywyc",
                            "ws_ywzc","yljdq_flow","yljdq_stop","fhz_f","fhz_h","fhz_ztyc","bj_wkps","bj_bpps","bj_bpmh","bjdsyc_zz",
                            "bjdsyc_ywj","bjdsyc_ywc","bjzc","pzq","jyh","drq","cysb_cyg","cysb_sgz","ld","cysb_lqq","cysb_qyb",
                            "cysb_qtjdq","ylsff","yx","jdyxx","ywj","SF6ylb","xldlb","ylb","ywb","ywc"};
//{"hxq_gjbs","hxq_gjtps","hxq_yfps","pzqcd","jdyxxsd","pzq"};

const std::vector<std::vector<unsigned int>> COLORS = {
    {0, 114, 189},   {217, 83, 25},   {237, 177, 32},  {126, 47, 142},  {119, 172, 48},  {77, 190, 238},
    {162, 20, 47},   {76, 76, 76},    {153, 153, 153}, {255, 0, 0},     {255, 128, 0},   {191, 191, 0},
    {0, 255, 0},     {0, 0, 255},     {170, 0, 255},   {85, 85, 0},     {85, 170, 0},    {85, 255, 0},
    {170, 85, 0},    {170, 170, 0},   {170, 255, 0},   {255, 85, 0},    {255, 170, 0},   {255, 255, 0},
    {0, 85, 128},    {0, 170, 128},   {0, 255, 128},   {85, 0, 128},    {85, 85, 128},   {85, 170, 128},
    {85, 255, 128},  {170, 0, 128},   {170, 85, 128},  {170, 170, 128}, {170, 255, 128}, {255, 0, 128},
    {255, 85, 128},  {255, 170, 128}, {255, 255, 128}, {0, 85, 255},    {0, 170, 255},   {0, 255, 255},
    {85, 0, 255},    {85, 85, 255},   {85, 170, 255},  {85, 255, 255},  {170, 0, 255},   {170, 85, 255},
    {170, 170, 255}, {170, 255, 255}, {255, 0, 255},   {255, 85, 255},  {255, 170, 255}, {85, 0, 0},
    {128, 0, 0},     {170, 0, 0},     {212, 0, 0},     {255, 0, 0},     {0, 43, 0},      {0, 85, 0},
    {0, 128, 0},     {0, 170, 0},     {0, 212, 0},     {0, 255, 0},     {0, 0, 43},      {0, 0, 85},
    {0, 0, 128},     {0, 0, 170},     {0, 0, 212},     {0, 0, 255},     {0, 0, 0},       {36, 36, 36},
    {73, 73, 73},    {109, 109, 109}, {146, 146, 146}, {182, 182, 182}, {219, 219, 219}, {0, 114, 189},
    {80, 183, 189},  {128, 128, 0}};
// 创建一个 vector 来存储 Algorithm 对象
std::vector<Algorithm> algorithm;
//天气算法
int weather_number;
// cv::Mat pyopencv_to_cvmat(PyObject* input) {
//     // 确保 NumPy 数组初始化
//     if (!PyArray_Check(input)) {
//         throw std::runtime_error("Input is not a valid NumPy array");
//     }

//     PyArrayObject* array = reinterpret_cast<PyArrayObject*>(input);

//     // 获取数组维度信息
//     int ndims = PyArray_NDIM(array);

//     if (ndims == 2) {
//         // 灰度图像
//         int rows = PyArray_DIM(array, 0);
//         int cols = PyArray_DIM(array, 1);

//         // 获取指向数组数据的指针
//         unsigned char* data = static_cast<unsigned char*>(PyArray_DATA(array));

//         // 创建 OpenCV Mat 对象
//         cv::Mat mat(rows, cols, CV_8UC1, data);

//         if (rows != 720 || cols != 1280) {
//             cv::Mat newmat;
//             cv::resize(mat, newmat, cv::Size(1280, 720));
//             return newmat;
//         }

//         return mat;

//     } else if (ndims == 3) {
//         // 彩色图像
//         int rows = PyArray_DIM(array, 0);
//         int cols = PyArray_DIM(array, 1);
//         int channels = PyArray_DIM(array, 2);

//         // 获取指向数组数据的指针
//         unsigned char* data = static_cast<unsigned char*>(PyArray_DATA(array));

//         // 创建 OpenCV Mat 对象
//         cv::Mat mat(rows, cols, CV_8UC3, data);

//         if (rows != 720 || cols != 1280) {
//             cv::Mat newmat;
//             cv::resize(mat, newmat, cv::Size(1280, 720));
//             return newmat;
//         }

//         return mat;

//     } else {
//         throw std::runtime_error("Unsupported array dimension");
//     }
// }

// py::array_t<unsigned char> cvmat_to_pyopencv(const cv::Mat& mat) {
//     std::vector<size_t> shape = { (size_t)mat.rows, (size_t)mat.cols, (size_t)mat.channels() };
//     std::vector<size_t> strides = { (size_t)mat.step[0], (size_t)mat.step[1], (size_t)mat.elemSize() };
    
//     return py::array_t<unsigned char>(shape, strides, mat.data);
// }
// PyObject* cvmat_to_pyopencv(const cv::Mat& mat) {
//     // 确保 NumPy 数组初始化
//     if (!PyArray_API) {
//         import_array();
//     }

//     // 获取图像的数据指针、形状和步幅
//     void* ptr = mat.data;
//     npy_intp dims[3] = { mat.rows, mat.cols, mat.channels() };

//     // 使用 NumPy C API 构造 NumPy 数组对象并返回
//     PyObject* array = PyArray_SimpleNewFromData(3, dims, NPY_UINT8, ptr);
//     if (!array) {
//         throw std::runtime_error("Could not create NumPy array");
//     }

//     // 设置数组的所有权信息
//     PyArrayObject* np_array = reinterpret_cast<PyArrayObject*>(array);
//     PyArray_ENABLEFLAGS(np_array, NPY_ARRAY_OWNDATA);

//     return array;
// }
// PyObject* cvmat_to_pyopencv(const cv::Mat& mat) {
//     // 确保 NumPy 数组初始化
//     if (!PyArray_API) {
//         import_array();
//     }

//     // 获取图像的数据指针
//     void* ptr = mat.data;

//     // 根据 mat 类型设置 NumPy 数组的数据类型
//     int type = NPY_UINT8;  // 默认类型是 NPY_UINT8，适用于 CV_8U 类型的 Mat
//     if (mat.type() == CV_8UC3) {
//         type = NPY_UINT8;
//     } else if (mat.type() == CV_8UC1) {
//         type = NPY_UINT8;
//     }
//     // 根据需要，处理其他类型，如 CV_32F 等
//     // else if (mat.type() == CV_32FC1) {
//     //     type = NPY_FLOAT32;
//     // }

//     // 设置形状维度
//     npy_intp dims[3];
//     int ndims = 0;

//     if (mat.channels() == 1) {
//         // 灰度图像
//         dims[0] = mat.rows;
//         dims[1] = mat.cols;
//         ndims = 2;  // 二维数组
//     } else {
//         // 彩色图像或其他多通道图像
//         dims[0] = mat.rows;
//         dims[1] = mat.cols;
//         dims[2] = mat.channels();
//         ndims = 3;  // 三维数组
//     }

//     // 使用 NumPy C API 构造 NumPy 数组对象并返回
//     PyObject* array = PyArray_SimpleNewFromData(ndims, dims, type, ptr);
//     if (!array) {
//         throw std::runtime_error("Could not create NumPy array");
//     }
//     PyArrayObject* np_array = reinterpret_cast<PyArrayObject*>(array);

//     // 数据由 OpenCV 管理，因此确保 NumPy 不会尝试释放数据
//     PyArray_CLEARFLAGS(np_array, NPY_ARRAY_OWNDATA);

//     return array;
// }
#include <cstdio>
#include <cstdlib>
#include <Python.h>
#include <numpy/arrayobject.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
}

void ffmpeg_to_pyobject(const char* input_filename, PyObject** output_array) {
    // 1. 初始化 FFmpeg 和 Python NumPy
    av_register_all();
    if (PyArray_API == nullptr) {
        if (_import_array() < 0) {
            PyErr_Print();
            PyErr_SetString(PyExc_ImportError, "NumPy core.multiarray failed to import");
            throw std::runtime_error("Failed to initialize NumPy array API");
        }
    }

    // 2. 打开输入文件并初始化格式上下文
    AVFormatContext* pFormatCtx = avformat_alloc_context();
    if (avformat_open_input(&pFormatCtx, input_filename, nullptr, nullptr) != 0) {
        throw std::runtime_error("无法打开输入文件");
    }

    // 3. 查找流信息
    if (avformat_find_stream_info(pFormatCtx, nullptr) < 0) {
        avformat_close_input(&pFormatCtx);
        throw std::runtime_error("无法找到流信息");
    }

    // 4. 查找视频流
    AVCodec* pCodec = nullptr;
    AVCodecContext* pCodecCtx = nullptr;
    int videoStream = -1;
    for (unsigned i = 0; i < pFormatCtx->nb_streams; i++) {
        if (pFormatCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            videoStream = i;
            pCodec = avcodec_find_decoder(pFormatCtx->streams[i]->codecpar->codec_id);
            if (!pCodec) {
                avformat_close_input(&pFormatCtx);
                throw std::runtime_error("无法找到视频解码器");
            }
            pCodecCtx = avcodec_alloc_context3(pCodec);
            if (avcodec_parameters_to_context(pCodecCtx, pFormatCtx->streams[i]->codecpar) < 0) {
                avcodec_free_context(&pCodecCtx);
                avformat_close_input(&pFormatCtx);
                throw std::runtime_error("无法初始化编解码器上下文");
            }
            if (avcodec_open2(pCodecCtx, pCodec, nullptr) < 0) {
                avcodec_free_context(&pCodecCtx);
                avformat_close_input(&pFormatCtx);
                throw std::runtime_error("无法打开编解码器");
            }
            break;
        }
    }

    if (videoStream == -1) {
        avformat_close_input(&pFormatCtx);
        throw std::runtime_error("无法找到视频流");
    }

    // 5. 初始化解码过程
    AVFrame* pFrame = av_frame_alloc();
    AVPacket packet;
    while (av_read_frame(pFormatCtx, &packet) >= 0) {
        if (packet.stream_index == videoStream) {
            if (avcodec_send_packet(pCodecCtx, &packet) == 0) {
                if (avcodec_receive_frame(pCodecCtx, pFrame) == 0) {
                    // 我们有一个完整的解码帧
                    break;
                }
            }
        }
        av_packet_unref(&packet);
    }

    // 6. 将解码帧转换为灰度（如果不是灰度）
    SwsContext* sws_ctx = sws_getContext(
        pCodecCtx->width, pCodecCtx->height, pCodecCtx->pix_fmt,
        pCodecCtx->width, pCodecCtx->height, AV_PIX_FMT_GRAY8,
        SWS_BILINEAR, nullptr, nullptr, nullptr
    );
    AVFrame* pFrameGray = av_frame_alloc();
    int numBytes = av_image_get_buffer_size(AV_PIX_FMT_GRAY8, pCodecCtx->width, pCodecCtx->height, 1);
    uint8_t* buffer = (uint8_t*)av_malloc(numBytes * sizeof(uint8_t));
    av_image_fill_arrays(pFrameGray->data, pFrameGray->linesize, buffer, AV_PIX_FMT_GRAY8, pCodecCtx->width, pCodecCtx->height, 1);
    sws_scale(sws_ctx, pFrame->data, pFrame->linesize, 0, pCodecCtx->height, pFrameGray->data, pFrameGray->linesize);

    // 7. 将灰度数据转换为 NumPy 数组
    npy_intp dims[2] = { pCodecCtx->height, pCodecCtx->width };  // 灰度图像是二维数组
    *output_array = PyArray_SimpleNewFromData(2, dims, NPY_UINT8, pFrameGray->data[0]);
    if (!*output_array) {
        throw std::runtime_error("无法创建 NumPy 数组");
    }

    // 8. 清理内存
    av_free(buffer);
    av_frame_free(&pFrameGray);
    av_frame_free(&pFrame);
    sws_freeContext(sws_ctx);
    avcodec_free_context(&pCodecCtx);
    avformat_close_input(&pFormatCtx);
}
// PyObject* avframe_to_pyobject(AVFrame* pFrameYUV, int width, int height) {
//     // 初始化 NumPy 数组
//     if (!PyArray_API) {
//         import_array();
//     }

//     // 确保 AVFrame 有数据
//     if (!pFrameYUV) {
//         throw std::runtime_error("空的 AVFrame");
//     }

//     // 创建一个 SwsContext，用于 YUV 到 BGR 的转换，指定颜色空间
//     SwsContext* sws_ctx = sws_getContext(
//         width, height, AV_PIX_FMT_YUV420P,  // 输入格式
//         width, height, AV_PIX_FMT_BGR24,     // 输出格式（注意这里是 BGR）
//         SWS_BILINEAR,                       // 缩放算法
//         nullptr, nullptr, nullptr);

//     if (!sws_ctx) {
//         throw std::runtime_error("无法创建 SwsContext");
//     }

//     // 分配用于存储 BGR 数据的 AVFrame
//     AVFrame* pFrameBGR = av_frame_alloc();
//     if (!pFrameBGR) {
//         sws_freeContext(sws_ctx);
//         throw std::runtime_error("无法分配 AVFrame");
//     }

//     // 分配 BGR 数据的缓冲区
//     int numBytes = av_image_get_buffer_size(AV_PIX_FMT_BGR24, width, height, 1);
//     uint8_t* buffer = (uint8_t*)av_malloc(numBytes * sizeof(uint8_t));

//     // 将缓冲区与 AVFrame 进行关联
//     av_image_fill_arrays(pFrameBGR->data, pFrameBGR->linesize, buffer, AV_PIX_FMT_BGR24, width, height, 1);

//     // 使用 sws_scale 进行转换
//     sws_scale(
//         sws_ctx, 
//         pFrameYUV->data, pFrameYUV->linesize, 0, height,  // 输入数据 (YUV420P)
//         pFrameBGR->data, pFrameBGR->linesize);            // 输出数据 (BGR)

//     // 释放 SwsContext
//     sws_freeContext(sws_ctx);

//     // 准备 NumPy 数组的维度
//     npy_intp dims[3] = { height, width, 3 };  // BGR 有 3 个通道

//     // 创建 NumPy 数组，将 BGR 数据转换为 NumPy 格式
//     PyObject* array = PyArray_SimpleNewFromData(3, dims, NPY_UINT8, pFrameBGR->data[0]);
//     if (!array) {
//         av_free(buffer);
//         av_frame_free(&pFrameBGR);
//         throw std::runtime_error("无法创建 NumPy 数组");
//     }

//     // 确保 NumPy 不会尝试释放 FFmpeg 管理的数据
//     PyArrayObject* np_array = reinterpret_cast<PyArrayObject*>(array);
//     PyArray_CLEARFLAGS(np_array, NPY_ARRAY_OWNDATA);  // 数据由 FFmpeg 管理，不要释放

//     // 清理
//     av_frame_free(&pFrameBGR);  // 释放 AVFrame，但不释放数据缓冲区 buffer
//     av_free(buffer);  // 释放 BGR 缓冲区

//     return array;
// }
PyObject* avframe_to_pyobject(AVFrame* pFrameYUV, int width, int height) {
    // 初始化 NumPy 数组
    if (!PyArray_API) {
        import_array();
    }

    // 确保 AVFrame 有数据
    if (!pFrameYUV) {
        throw std::runtime_error("空的 AVFrame");
    }

    // 创建 SwsContext，用于 YUV 到 BGR 的转换
    SwsContext* sws_ctx = sws_getContext(
        width, height, AV_PIX_FMT_YUV420P,
        width, height, AV_PIX_FMT_BGR24,
        SWS_BILINEAR, nullptr, nullptr, nullptr);

    if (!sws_ctx) {
        throw std::runtime_error("无法创建 SwsContext");
    }

    // 分配 BGR 数据的缓冲区
    int numBytes = av_image_get_buffer_size(AV_PIX_FMT_BGR24, width, height, 1);
    uint8_t* buffer = (uint8_t*)av_malloc(numBytes * sizeof(uint8_t));
    if (!buffer) {
        sws_freeContext(sws_ctx);
        throw std::runtime_error("无法分配缓冲区");
    }

    // 创建 AVFrame 来存储 BGR 数据
    AVFrame* pFrameBGR = av_frame_alloc();
    if (!pFrameBGR) {
        sws_freeContext(sws_ctx);
        av_free(buffer);
        throw std::runtime_error("无法分配 AVFrame");
    }

    av_image_fill_arrays(pFrameBGR->data, pFrameBGR->linesize, buffer, AV_PIX_FMT_BGR24, width, height, 1);

    // 执行 YUV 到 BGR 的转换
    sws_scale(sws_ctx, pFrameYUV->data, pFrameYUV->linesize, 0, height, pFrameBGR->data, pFrameBGR->linesize);

    // 释放 SwsContext
    sws_freeContext(sws_ctx);

    // 准备 NumPy 数组的维度
    npy_intp dims[3] = { height, width, 3 };  // BGR 有 3 个通道

    // 创建 NumPy 数组，将 BGR 数据转换为 NumPy 格式
    PyObject* array = PyArray_SimpleNewFromData(3, dims, NPY_UINT8, pFrameBGR->data[0]);
    if (!array) {
        av_frame_free(&pFrameBGR);
        av_free(buffer);
        throw std::runtime_error("无法创建 NumPy 数组");
    }

    // 创建 PyCapsule 来管理缓冲区的释放
    PyObject* capsule = PyCapsule_New(buffer, nullptr, [](PyObject* cap) {
        av_free(PyCapsule_GetPointer(cap, nullptr));
    });

    // 设置 NumPy 数组的 base 为 capsule，确保缓冲区在 Python 端的生命周期正确管理
    PyArrayObject* np_array = reinterpret_cast<PyArrayObject*>(array);
    PyArray_SetBaseObject(np_array, capsule);  // 设置 capsule 为 base，确保正确释放

    // 清理
    av_frame_free(&pFrameBGR);  // 释放 AVFrame，但不释放数据缓冲区，因为它现在由 Python 管理

    return array;
}
void save_frame_as_jpeg(AVFrame *frame, const char *filename) {
    // 使用 libjpeg 库来保存为 JPEG
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;

    // 初始化 JPEG 压缩对象
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);

    // 打开输出文件
    FILE *outfile = fopen(filename, "wb");
    if (!outfile) {
        fprintf(stderr, "无法打开文件 %s\n", filename);
        return;
    }
    jpeg_stdio_dest(&cinfo, outfile);

    // 设置 JPEG 压缩参数
    cinfo.image_width = frame->width;
    cinfo.image_height = frame->height;
    cinfo.input_components = 3; // RGB
    cinfo.in_color_space = JCS_RGB;

    jpeg_set_defaults(&cinfo);
    jpeg_start_compress(&cinfo, TRUE);

    // 写入每一行的像素数据
    while (cinfo.next_scanline < cinfo.image_height) {
        uint8_t *row_pointer[1];
        row_pointer[0] = frame->data[0] + cinfo.next_scanline * frame->linesize[0];
        jpeg_write_scanlines(&cinfo, row_pointer, 1);
    }

    // 结束压缩
    jpeg_finish_compress(&cinfo);
    fclose(outfile);
    jpeg_destroy_compress(&cinfo);
}

void PyObjectToAVFrame(PyObject* pyArray, AVFrame* out_avframe, int width, int height) {
    // 确保 NumPy 数组初始化
    if (PyArray_API == nullptr) {
        if (_import_array() < 0) {
            PyErr_Print();
            throw std::runtime_error("Failed to initialize NumPy API.");
        }
    }

    // 检查输入是否为 NumPy 数组
    if (!PyArray_Check(pyArray)) {
        throw std::runtime_error("输入不是有效的 NumPy 数组");
    }

    // 获取 NumPy 数组的类型和数据
    PyArrayObject* np_array = reinterpret_cast<PyArrayObject*>(pyArray);
    uint8_t* data = static_cast<uint8_t*>(PyArray_DATA(np_array));

    // 检查 NumPy 数据是否有效
    if (!data) {
        throw std::runtime_error("无法获取 NumPy 数组的数据");
    }

    // 获取 NumPy 输入图像的原始宽高
    int numpy_width = static_cast<int>(PyArray_DIMS(np_array)[1]);  // 图像的宽度
    int numpy_height = static_cast<int>(PyArray_DIMS(np_array)[0]); // 图像的高度
    int numpy_linesize[1] = { static_cast<int>(PyArray_STRIDE(np_array, 0)) };

    // 如果输入图像的尺寸不是 1280x720，则调整图像大小
    SwsContext* resizeContext = nullptr;
    uint8_t* resized_data[1];
    int resized_linesize[1];
    
    if (numpy_width != 1280 || numpy_height != 720) {
        // 分配用于缩放后图像的内存
        AVFrame* tempFrame = av_frame_alloc();
        av_image_alloc(tempFrame->data, tempFrame->linesize, 1280, 720, AV_PIX_FMT_BGR24, 1);

        // 创建一个新的 SWS 上下文来调整大小
        resizeContext = sws_getContext(
            numpy_width,
            numpy_height,
            AV_PIX_FMT_BGR24,         // 输入图像格式
            1280,
            720,
            AV_PIX_FMT_BGR24,         // 输出图像格式
            SWS_BILINEAR,             // 缩放算法
            nullptr, nullptr, nullptr
        );

        if (!resizeContext) {
            av_freep(&tempFrame->data[0]);
            av_frame_free(&tempFrame);
            throw std::runtime_error("无法创建图像缩放 SwsContext");
        }

        // 调整图像大小
        sws_scale(
            resizeContext,
            &data,                   // 输入数据
            numpy_linesize,           // 输入步长
            0,                       // 开始处理的行
            numpy_height,            // 原始图像的高度
            tempFrame->data,         // 缩放后的数据
            tempFrame->linesize      // 缩放后的步长
        );

        // 使用缩放后的数据
        resized_data[0] = tempFrame->data[0];
        resized_linesize[0] = tempFrame->linesize[0];

        // 释放调整大小的 SWS 上下文
        if (resizeContext) {
            sws_freeContext(resizeContext);
            resizeContext = nullptr;
        }
    } else {
        // 如果图像已经是 1280x720，直接使用原始数据
        resized_data[0] = data;
        resized_linesize[0] = numpy_linesize[0];
    }

    // 创建 SWS 上下文，用于从 BGR24 转换到 YUV420P
    SwsContext* colorConvertContext = sws_getContext(
        1280,
        720,
        AV_PIX_FMT_BGR24,         // 输入图像格式 (BGR)
        1280,
        720,
        AV_PIX_FMT_YUV420P,       // 输出图像格式 (YUV420P)
        SWS_FAST_BILINEAR,        // 缩放算法
        nullptr, nullptr, nullptr
    );

    if (!colorConvertContext) {
        throw std::runtime_error("无法创建颜色转换 SwsContext");
    }

    // 使用 sws_scale 将图像从 BGR24 转换为 YUV420P
    sws_scale(
        colorConvertContext,
        resized_data,             // 输入数据 (可能是缩放后的数据)
        resized_linesize,         // 输入步长
        0,                        // 开始处理的行
        720,                      // 图像的高度
        out_avframe->data,        // 输出 AVFrame 的数据
        out_avframe->linesize     // 输出 AVFrame 的步长
    );

    // 释放颜色转换的 SWS 上下文
    if (colorConvertContext) {
        sws_freeContext(colorConvertContext);
        colorConvertContext = nullptr;
    }

    // 如果有临时缩放的帧，需要释放内存
    if (resized_data[0] != data) {
        av_freep(&resized_data[0]);
    }
}


// void CvMatToAVFrame(const cv::Mat& input_mat, AVFrame* out_avframe)
// {
//     int image_width = input_mat.cols;
//     int image_height = input_mat.rows;
//     int cvLinesizes[1];
//     cvLinesizes[0] = input_mat.step1();

//     SwsContext* openCVBGRToAVFrameSwsContext = sws_getContext(
//         image_width,
//         image_height,
//         AVPixelFormat::AV_PIX_FMT_BGR24,
//         image_width,
//         image_height,
//         AVPixelFormat::AV_PIX_FMT_YUV420P,
//         SWS_FAST_BILINEAR,
//         nullptr, nullptr, nullptr
//     );

//     sws_scale(openCVBGRToAVFrameSwsContext,
//         &input_mat.data,
//         cvLinesizes,
//         0,
//         image_height,
//         out_avframe->data,
//         out_avframe->linesize);

//     if (openCVBGRToAVFrameSwsContext != nullptr)
//     {
//         sws_freeContext(openCVBGRToAVFrameSwsContext);
//         openCVBGRToAVFrameSwsContext = nullptr;
//     }
// }
// void CvGrayMatToAVFrame(const cv::Mat& input_mat, AVFrame* out_avframe)
// {
//     int image_width = input_mat.cols;
//     int image_height = input_mat.rows;
//     int cvLinesizes[1];
//     cvLinesizes[0] = input_mat.step1();
//     // 创建 SwsContext 用于将灰度图像转换为 YUV420P
//     SwsContext* openCVGrayToAVFrameSwsContext = sws_getContext(
//         image_width,
//         image_height,
//         AVPixelFormat::AV_PIX_FMT_GRAY8,
//         image_width,
//         image_height,
//         AVPixelFormat::AV_PIX_FMT_YUV420P,
//         SWS_FAST_BILINEAR,
//         nullptr, nullptr, nullptr
//     );
//     sws_scale(openCVGrayToAVFrameSwsContext,
//         &input_mat.data,
//         cvLinesizes,
//         0,
//         image_height,
//         out_avframe->data,
//         out_avframe->linesize);
//     // 释放 SwsContext
//     if (openCVGrayToAVFrameSwsContext != nullptr)
//     {
//         sws_freeContext(openCVGrayToAVFrameSwsContext);
//         openCVGrayToAVFrameSwsContext = nullptr;
//     }
// }
// cv::Mat AVFrameToCvMat(AVFrame* input_avframe)
// {
//     int image_width = input_avframe->width;
//     int image_height = input_avframe->height;

//     cv::Mat resMat(image_height, image_width, CV_8UC3);
//     int cvLinesizes[1];
//     cvLinesizes[0] = resMat.step1();

//     SwsContext* avFrameToOpenCVBGRSwsContext = sws_getContext(
//         image_width,
//         image_height,
//         AVPixelFormat::AV_PIX_FMT_YUV420P,
//         image_width,
//         image_height,
//         AVPixelFormat::AV_PIX_FMT_BGR24,
//         SWS_FAST_BILINEAR,
//         nullptr, nullptr, nullptr
//     );

//     sws_scale(avFrameToOpenCVBGRSwsContext,
//         input_avframe->data,
//         input_avframe->linesize,
//         0,
//         image_height,
//         &resMat.data,
//         cvLinesizes);

//     if (avFrameToOpenCVBGRSwsContext != nullptr)
//     {
//         sws_freeContext(avFrameToOpenCVBGRSwsContext);
//         avFrameToOpenCVBGRSwsContext = nullptr;
//     }

//     return resMat;
// }

int ScaleImg(int nSrcH,int nSrcW,AVFrame *src_picture,AVFrame *dst_picture,int nDstH ,int nDstW )  
    {  
    int i ;  
    int nSrcStride[3];  
    int nDstStride[3];    
    struct SwsContext* m_pSwsContext;  
      
      
    uint8_t *pSrcBuff[3] = {src_picture->data[0],src_picture->data[1], src_picture->data[2]};  
      
      
    nSrcStride[0] = nSrcW ;  
    nSrcStride[1] = nSrcW/2 ;  
    nSrcStride[2] = nSrcW/2 ;  
      
      
      
      
    dst_picture->linesize[0] = nDstW;  
    dst_picture->linesize[1] = nDstW / 2;  
    dst_picture->linesize[2] = nDstW / 2;  
      
      
    printf("nSrcW%d\n",nSrcW);  
      
      
    m_pSwsContext = sws_getContext(nSrcW, nSrcH, AV_PIX_FMT_YUV420P,  
    nDstW, nDstH, AV_PIX_FMT_YUV420P,  
    SWS_BICUBIC,  
    NULL, NULL, NULL);  
      
      
    if (NULL == m_pSwsContext)  
    {  
    printf("ffmpeg get context error!\n");  
    exit (-1);  
    }  
      
      
    sws_scale(m_pSwsContext, src_picture->data,src_picture->linesize, 0, nSrcH,dst_picture->data,dst_picture->linesize);  
      
      
    printf("line0:%d line1:%d line2:%d\n",dst_picture->linesize[0] ,dst_picture->linesize[1] ,dst_picture->linesize[2]);  
    sws_freeContext(m_pSwsContext);  
      
      
    return 1 ;  
    }  
static int WriteRtpCallback(void* opaque, uint8_t* buf, int buf_size) {
  WebRTCSessionFactory* webrtc_session_factory = (WebRTCSessionFactory*)opaque;
  std::vector<std::shared_ptr<WebRTCSession>> all_sessions;
  std::shared_ptr<uint8_t> shared_buf(new uint8_t[buf_size]);
  memcpy(shared_buf.get(), buf, buf_size);
  webrtc_session_factory->GetAllReadyWebRTCSession(&all_sessions);
  for (const auto& session : all_sessions) {
    
    session->loop()->runInLoop([session, shared_buf, buf_size]() {

      session->webrtc_transport()->EncryptAndSendRtpPacket(shared_buf.get(), buf_size);
    });
  }
  return buf_size;
}

static void encode(AVCodecContext* enc_ctx, AVFrame* frame, AVPacket* pkt) {
    int ret;

    /* send the frame to the encoder */
    ret = avcodec_send_frame(enc_ctx, frame);
    if (ret < 0) {
        exit(1);
    }
    ret = avcodec_receive_packet(enc_ctx, pkt);
    // while (ret >= 0) {
    //     // receive packet
    //     ret = avcodec_receive_packet(enc_ctx, pkt);
    //     if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
    //         return;
    //     else if (ret < 0) {
    //         exit(1);
    //     }
    //     //会清空pkt
    //     // av_packet_unref(pkt);
    // }
}
static void decode(AVCodecContext* dec_ctx, AVFrame* frame, AVPacket* pkt) {

    int ret;

    // send packet to decoder
    ret = avcodec_send_packet(dec_ctx, pkt);
    if (ret < 0) {
        fprintf(stderr, "Error sending a packet for decoding %d\n", ret);
        exit(1);
    }

    if (ret >= 0) {
        // receive frame
        ret = avcodec_receive_frame(dec_ctx, frame);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            return;
        }
        else if (ret < 0) {
            fprintf(stderr, "Error during decoding\n");
            exit(1);
        }

        // printf("dec_ctx->frame %3d, key %d, type %d\n", 
        //                 dec_ctx->frame_number, 
        //                 frame->key_frame,
        //                 frame->pict_type);
        fflush(stdout);
        // frame --> mat
        // AVFrame* frame_bgr24 = av_frame_alloc();
        // if (frame == nullptr) {
        //     cerr << "could not allocate frame_bgr24" << endl;
        //     exit(1);
        // }
        // frame_bgr24->format = AV_PIX_FMT_BGR24;
        // frame_bgr24->width = frame->width;
        // frame_bgr24->height = frame->height;
        // if (av_frame_get_buffer(frame_bgr24, 0) < 0) {
        //     cerr << "could not allocate video frame_bgr24 data" << endl;
        //     exit(1);
        // }
        // struct SwsContext* frame_convert_ctx;
        // frame_convert_ctx = sws_getContext(
        //     frame->width,
        //     frame->height,
        //     AV_PIX_FMT_YUV420P,
        //     frame_bgr24->width,
        //     frame_bgr24->height,
        //     AV_PIX_FMT_BGR24,
        //     SWS_BICUBIC,
        //     NULL, NULL, NULL);
        // sws_scale(
        //     frame_convert_ctx,
        //     frame->data,
        //     frame->linesize,
        //     0,
        //     frame_bgr24->height,
        //     frame_bgr24->data,
        //     frame_bgr24->linesize);

        // cv::Mat mat(cv::Size(frame_bgr24->width, frame_bgr24->height), CV_8UC3);
        // mat.data = frame_bgr24->data[0];
        // cv::imshow("show", mat);
        // if (cv::waitKey(50) >= 0) {
        //     av_free(frame_bgr24);
        //     sws_freeContext(frame_convert_ctx);
        //     break;
        // }
        // av_free(frame_bgr24);
        // sws_freeContext(frame_convert_ctx);
    }
}
static int H2642Rtp(const char* in_filename, void* opaque,std::vector<Algorithm> *algorithm,int * weather_number) {
    // cv::Mat mat1=cv::imread("1.jpg");
    Py_Initialize();
    // 添加 Python 脚本目录到 Python 路径
    PyRun_SimpleString("import sys");
    // const wchar_t*  python_path = L"../python";
    // PySys_SetPath(python_path);
    PyRun_SimpleString("sys.path.append('/home/easy_webrtc_server/python')");
    PyObject* pModule = PyImport_ImportModule("algorithm");
    PyRun_SimpleString("sys.path.append('/home/easy_webrtc_server/python/HJJX/model')");
    PyObject* pModuleSOT = PyImport_ImportModule("HJJX_TEST");
    // PyObject* pModuleSOT = nullptr;
    // 导入 Python 模块
    // PyObject* pModule = PyImport_ImportModule("algorithm");
    // if (!pModule) {
    //     PyErr_Print();
    //     std::cerr << "Failed to import Python module" << std::endl;
    //     return 1;
    // }
    // PyObject* mean_filter = PyObject_GetAttrString(pModule, "mean_filter");
    // if (!mean_filter || !PyCallable_Check(mean_filter)) {
    //     PyErr_Print();
    //     std::cerr << "Cannot find function 'process_image'" << std::endl;
    //     Py_XDECREF(mean_filter);
    //     return 1;
    // }
    // PyObject* adjust_contrast_and_lightness = PyObject_GetAttrString(pModule, "adjust_contrast_and_lightness");
    // if (!adjust_contrast_and_lightness || !PyCallable_Check(adjust_contrast_and_lightness)) {
    //     PyErr_Print();
    //     std::cerr << "Cannot find function 'process_image'" << std::endl;
    //     Py_XDECREF(adjust_contrast_and_lightness);
    //     return 1;
    // }
    // PyRun_SimpleString("sys.path.append('../python/YoloStitch')");
    // PyObject* pModulestitch = PyImport_ImportModule("mystitch");
    // PyObject* stitch = PyObject_GetAttrString(pModulestitch, "stitch");
    // printf("8\n");
    // if (!stitch || !PyCallable_Check(stitch)) {
    //     PyErr_Print();
    //     std::cerr << "Cannot find function 'process_image'" << std::endl;
    //     return 1;
    // }
    // cv::Mat mat1=cv::imread("1.jpg");
    // cv::Mat mat2=cv::imread("2.jpg");
    // PyObject* input1 = cvmat_to_pyopencv(mat1);
    // PyObject* input2 = cvmat_to_pyopencv(mat2);
    // PyObject* pArgs = PyTuple_Pack(2, input1,input2); // 传递参数
    // PyObject* presult=PyObject_CallObject(stitch, pArgs);
    // cv::Mat mat = pyopencv_to_cvmat(presult);
    // cv::imwrite("3.jpg",mat);
    // return 0;
    PyRun_SimpleString("sys.path.append('/home/opencvdemo/4_area_search')");
    PyObject* pModuleDetect = PyImport_ImportModule("sdk_python");
    // PyObject* pModuleDetect = nullptr;
    PyObject* pModulemmtracker = PyImport_ImportModule("mmtest");
    // PyRun_SimpleString("sys.path.append('/home/opencvdemo/SiamDW/siamese_tracking')");
    PyRun_SimpleString("sys.path.append('/home/opencvdemo/SiamDW')");
    PyObject* pModuleSOTR = PyImport_ImportModule("siamese_tracking.run_given_videoflow_ONNX_RT");
    // PyObject* pModuleSOTR = nullptr;
    PyRun_SimpleString("sys.path.append('/home/opencvdemo/skyAR/model/task6')");
    PyObject* pModuleweather = PyImport_ImportModule("task6");
    // PyObject* pModuleweather = nullptr;
    PyRun_SimpleString("sys.path.append('/home/opencvdemo/stitch')");
    PyObject* pModuleStitch = PyImport_ImportModule("sdk_python_stitch");
    // PyObject* pModuleStitch = nullptr;
    // PyObject* pModuleweather = nullptr;
    // PyObject* detect = PyObject_GetAttrString(pModuleDetect, "detect");
    // printf("8\n");
    // if (!detect || !PyCallable_Check(detect)) {
    //     PyErr_Print();
    //     std::cerr << "Cannot find function 'process_image'" << std::endl;
    //     return 1;
    // }
  
  const AVOutputFormat* ofmt = NULL;
  AVIOContext* avio_ctx = NULL;
  //用于打开、读取、写入音视频文件，并维护了音视频格式的全局信息
  AVFormatContext *ifmt_ctx = NULL, *ofmt_ctx = NULL;
  AVPacket* pkt = NULL;
  AVCodecContext* dec_context = nullptr;
  AVCodecContext* enc_context = nullptr;
  // cv::Mat grayMat= cv::imread("gray1280.jpg",cv::IMREAD_GRAYSCALE);
  // PyObject* gray = cvmat_to_pyopencv(grayMat);
  PyObject* gray ;
  PyObject* input;
  PyObject* previmg;
  PyObject* originalinput;
  bool prevlabel = false;
  const char* filename = "gray1280.jpg";
  ffmpeg_to_pyobject(filename, &gray);
  const AVCodec* codec = nullptr;
  avcodec_register_all;
  const char* input_filename = "uav/1.mp4";
  const char* input_filename1 = "uav/2.mp4";
  //内置视频
  // 初始化 FFmpeg 库
    av_register_all();
    
    // 打开输入文件
    AVFormatContext* format_ctx1 = avformat_alloc_context();
    if (avformat_open_input(&format_ctx1, input_filename, nullptr, nullptr) != 0) {
        std::cerr << "Failed to open input file!" << std::endl;
        return -1;
    }
    AVFormatContext* format_ctx2 = avformat_alloc_context();
    if (avformat_open_input(&format_ctx2, input_filename1, nullptr, nullptr) != 0) {
        std::cerr << "Failed to open input file!" << std::endl;
        return -1;
    }
    // 查找流信息
    if (avformat_find_stream_info(format_ctx1, nullptr) < 0) {
        std::cerr << "Failed to find stream info!" << std::endl;
        return -1;
    }
    if (avformat_find_stream_info(format_ctx2, nullptr) < 0) {
        std::cerr << "Failed to find stream info!" << std::endl;
        return -1;
    }
    // 找到视频流
    int video_stream_idx1 = -1;
    for (unsigned int i = 0; i < format_ctx1->nb_streams; ++i) {
        if (format_ctx1->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream_idx1 = i;
            break;
        }
    }
    int video_stream_idx2 = -1;
    for (unsigned int i = 0; i < format_ctx2->nb_streams; ++i) {
        if (format_ctx2->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream_idx2 = i;
            break;
        }
    }
    if (video_stream_idx1 == -1) {
        std::cerr << "Failed to find video stream!" << std::endl;
        return -1;
    }
    if (video_stream_idx2 == -1) {
        std::cerr << "Failed to find video stream!" << std::endl;
        return -1;
    }
    // 获取视频流的编解码器参数
    AVCodecParameters* codec_params1 = format_ctx1->streams[video_stream_idx1]->codecpar;
    AVCodecParameters* codec_params2 = format_ctx2->streams[video_stream_idx1]->codecpar;
    // 查找编解码器
    AVCodec* codec1 = avcodec_find_decoder(codec_params1->codec_id);
    if (!codec1) {
        std::cerr << "Failed to find codec!" << std::endl;
        return -1;
    }
    AVCodec* codec2 = avcodec_find_decoder(codec_params2->codec_id);
    if (!codec1) {
        std::cerr << "Failed to find codec!" << std::endl;
        return -1;
    }
    // 创建解码器上下文
    AVCodecContext* codec_ctx1 = avcodec_alloc_context3(codec1);
    if (!codec_ctx1) {
        std::cerr << "Failed to allocate codec context!" << std::endl;
        return -1;
    }
    AVCodecContext* codec_ctx2 = avcodec_alloc_context3(codec2);
    if (!codec_ctx2) {
        std::cerr << "Failed to allocate codec context!" << std::endl;
        return -1;
    }
    //将流中的编解码器参数复制到解码器上下文中
    if (avcodec_parameters_to_context(codec_ctx1, codec_params1) < 0) {
        std::cerr << "Failed to copy codec parameters to context!" << std::endl;
        return -1;
    }
    if (avcodec_parameters_to_context(codec_ctx2, codec_params2) < 0) {
        std::cerr << "Failed to copy codec parameters to context!" << std::endl;
        return -1;
    }
    // codec_ctx1->time_base = (AVRational){1, 30}; // 25 FPS
    //     codec_ctx1->framerate = (AVRational){30, 1};
      
    //   // enc_context->time_base = {1, 25};
    //   codec_ctx1->width = 1280;  // 新的宽度
    //   codec_ctx1->height = 720;  // 新的高度
    //   codec_ctx1->flags = 1 ;
    //   codec_ctx1->gop_size = 30;
    //   // enc_context->bit_rate = 90000;
    //   // 设置编码器的像素格式
    //   codec_ctx1->pix_fmt = AV_PIX_FMT_YUV420P; 
    codec_ctx1->thread_count=4;
    codec_ctx2->thread_count=4;
    // 打开解码器
    if (avcodec_open2(codec_ctx1, codec1, nullptr) < 0) {
        std::cerr << "Failed to open codec!" << std::endl;
        return -1;
    }
    if (avcodec_open2(codec_ctx2, codec2, nullptr) < 0) {
        std::cerr << "Failed to open codec!" << std::endl;
        return -1;
    }
    AVPacket* packet1 = av_packet_alloc();
    AVPacket* packet2 = av_packet_alloc();
  //内置视频
  // cv::VideoCapture cap2("uav/1.mp4"); // 使用视频文件
  // cv::VideoCapture cap3("uav/2.mp4"); // 使用视频文件
  int ret, i;
  int in_stream_index = 0, out_stream_index = 0;
  int stream_mapping_size = 0;
  int64_t pts = 0;
  uint8_t *buffer = NULL, *avio_ctx_buffer = NULL;
  size_t buffer_size=0;
  size_t avio_ctx_buffer_size = 4096;
  //设置为tcp传输，默认为udp
  AVDictionary* options = nullptr;
  const AVBitStreamFilter* abs_filter = NULL;
  //AVBSFContext是FFmpeg中用于比特流过滤器（Bitstream Filter）的上下文结构体，
  //用于对音频和视频比特流进行处理。比特流过滤器可以用于添加、删除、修改、解析和分析比特流中的元数据或其他信息。
  AVBSFContext* abs_ctx = NULL;
  //比特流过滤器作用是将H.264码流从MP4容器格式转换为H.264码流的裸流格式，
  //即将MP4封装中的H.264码流解析出来，添加H.264码流的起始码（start code）以形成H.264裸流格式（Annex B格式）。
  abs_filter = av_bsf_get_by_name("null");//h264_mp4toannexb hevc_mp4toannexb null
  //分配并初始化
  av_bsf_alloc(abs_filter, &abs_ctx);

  avio_ctx_buffer = (uint8_t*)av_malloc(avio_ctx_buffer_size);
  if (!avio_ctx_buffer) {
    ret = AVERROR(ENOMEM);
    goto end;
  }
  //1表示avio_ctx可写，opaque用于在写入数据时传递额外的信息，NULL表示不使用读取数据的回调函数。
  //WriteRtpCallback用于在写入RTP数据时调用，NULL表示不使用设置I/O位置的回调函数。
  //这段代码的作用是创建一个AVIOContext对象，用于写入RTP数据，并指定了一个回调函数WriteRtpCallback来实际执行写入操作。
  avio_ctx = avio_alloc_context(avio_ctx_buffer, avio_ctx_buffer_size, 1, opaque, NULL,
                                WriteRtpCallback, NULL);
  if (!avio_ctx) {
    ret = AVERROR(ENOMEM);
    goto end;
  }
  avio_ctx->max_packet_size = 1400;//不能设置太大
  //AVPacket 是用于存储压缩后的音视频数据的结构体
  pkt = av_packet_alloc();
  if (!pkt) {
    fprintf(stderr, "Could not allocate AVPacket\n");
    return 1;
  }

  //ifmt_ctx存储了音视频文件的全部信息,最后两个参数通常为NULL，表示不指定特定的输入格式，让FFmpeg库自行检测。
  av_dict_set(&options, "rtsp_transport", "udp", 0);
  if ((ret = avformat_open_input(&ifmt_ctx, in_filename, 0, &options)) < 0) {
    fprintf(stderr, "Could not open input file '%s'", in_filename);
    goto end;
  }
  //avformat_find_stream_info用于从输入文件中获取流的信息
  if ((ret = avformat_find_stream_info(ifmt_ctx, 0)) < 0) {
    fprintf(stderr, "Failed to retrieve input stream information");
    goto end;
  }
  // 打印信息
  printf("输入流信息:\n");
  av_dump_format(ifmt_ctx, 0, in_filename, 0);
  //创建输出容器
  avformat_alloc_output_context2(&ofmt_ctx, NULL, "rtp", NULL);
  
  if (!ofmt_ctx) {
    fprintf(stderr, "Could not create output context\n");
    ret = AVERROR_UNKNOWN;
    goto end;
  }
  //输出信息
  ofmt = ofmt_ctx->oformat;
  //key=ssrc value=12345678
  av_opt_set_int(ofmt_ctx->priv_data, "ssrc", 12345678, 0);
  //输入流的数量
  for (i = 0; i < ifmt_ctx->nb_streams; i++) {
    AVStream* out_stream;
    AVStream* in_stream = ifmt_ctx->streams[i];
    AVCodecParameters* in_codecpar = in_stream->codecpar;
    if (in_codecpar->codec_type != AVMEDIA_TYPE_VIDEO) {
      continue;
    }
    if(in_codecpar->codec_id == AV_CODEC_ID_H264) {
      printf("视频流为H264编码");
      //编码器
      //https://blog.csdn.net/li_wen01/article/details/62889494
      codec = avcodec_find_encoder(AV_CODEC_ID_H264);
      // codec = avcodec_find_encoder_by_name("h264_nvenc");
      if (!codec) {
            fprintf(stderr, "Codec not found\n");
            exit(1);
        }
      enc_context = avcodec_alloc_context3(codec);
        if (!enc_context) {
            fprintf(stderr, "Could not allocate video codec context\n");
            exit(1);
        }
        enc_context->time_base = (AVRational){1, 25}; // 25 FPS
        enc_context->framerate = (AVRational){25, 1};
      
      // enc_context->time_base = {1, 25};
      enc_context->width = 1280;  // 新的宽度
      enc_context->height = 720;  // 新的高度
      enc_context->flags = 1 ;
      enc_context->gop_size = 30;
      // enc_context->bit_rate = 90000;
      // 设置编码器的像素格式
      enc_context->pix_fmt = AV_PIX_FMT_YUV420P; 
      enc_context->thread_count=4;
      av_opt_set(enc_context->priv_data, "preset", "superfast", 0); //转码速度越快，视频也越模糊
      av_opt_set(enc_context->priv_data, "tune", "zerolatency", 0); //编码器实时编码，必须放在open前，零延迟，用在需要非常低的延迟的情况下，比如电视电话会议的编码。
        // if (av_frame_get_buffer(frame, 0) < 0) {
        //     exit(1);
        // }
        // if (av_frame_make_writable(frame) < 0) {
        //     exit(1);
        // }
      avcodec_open2(enc_context,codec,NULL);
      
      //解码器
      // codec = NULL;
      codec=avcodec_find_decoder(AV_CODEC_ID_H264);
      // codec=avcodec_find_decoder_by_name("h264_cuvid");
        if (!codec) {
            fprintf(stderr, "Codec not found\n");
            exit(1);
        }
      dec_context = avcodec_alloc_context3(codec);
        if (!dec_context) {
            fprintf(stderr, "Could not allocate video codec context\n");
            exit(1);
        }
      dec_context->thread_count=4;
      dec_context->width=1280;
      dec_context->height=720;
      dec_context->pix_fmt = AV_PIX_FMT_YUV420P;
      avcodec_open2(dec_context,codec,NULL);
      out_stream = avformat_new_stream(ofmt_ctx, NULL);
      if (!out_stream) {
        fprintf(stderr, "Failed allocating output stream\n");
        ret = AVERROR_UNKNOWN;
        goto end;
      }
      ret = avcodec_parameters_copy(out_stream->codecpar, in_codecpar);
      if (ret < 0) {
        fprintf(stderr, "Failed to copy codec parameters\n");
        goto end;
      }

      avcodec_parameters_copy(abs_ctx->par_in, in_codecpar);
      //仅仅是让abs_ctx能够正常工作，不会初始化输入流与输出流的编解码器
      av_bsf_init(abs_ctx);
      avcodec_parameters_copy(out_stream->codecpar, abs_ctx->par_out);

      in_stream_index = i;
      out_stream_index = out_stream->index;
      break;
    }
    if(in_codecpar->codec_id == AV_CODEC_ID_HEVC) {
      printf("视频流为H265编码");
          //https://blog.csdn.net/li_wen01/article/details/62889494
      codec = avcodec_find_encoder(AV_CODEC_ID_H264);
      // codec = avcodec_find_encoder_by_name("h264_nvenc");
      if (!codec) {
            fprintf(stderr, "Codec not found\n");
            exit(1);
        }
      enc_context = avcodec_alloc_context3(codec);
        if (!enc_context) {
            fprintf(stderr, "Could not allocate video codec context\n");
            exit(1);
        }
        enc_context->time_base = (AVRational){1, 25}; // 25 FPS
        enc_context->framerate = (AVRational){25, 1};
      
      // enc_context->time_base = {1, 25};
      enc_context->width = 1280;  // 新的宽度
      enc_context->height = 720;  // 新的高度
      enc_context->flags = 1 ;
      enc_context->gop_size = 30;
      // enc_context->bit_rate = 90000;
      // 设置编码器的像素格式
      enc_context->pix_fmt = AV_PIX_FMT_YUV420P; 
      enc_context->thread_count=2;
      av_opt_set(enc_context->priv_data, "preset", "superfast", 0); //转码速度越快，视频也越模糊
      av_opt_set(enc_context->priv_data, "tune", "zerolatency", 0); //编码器实时编码，必须放在open前，零延迟，用在需要非常低的延迟的情况下，比如电视电话会议的编码。
        // if (av_frame_get_buffer(frame, 0) < 0) {
        //     exit(1);
        // }
        // if (av_frame_make_writable(frame) < 0) {
        //     exit(1);
        // }
      avcodec_open2(enc_context,codec,NULL);
      
      //解码器
      // codec = NULL;
      codec=avcodec_find_decoder(AV_CODEC_ID_H265);
      // codec=avcodec_find_decoder_by_name("h264_cuvid");
        if (!codec) {
            fprintf(stderr, "Codec not found\n");
            exit(1);
        }
      dec_context = avcodec_alloc_context3(codec);
        if (!dec_context) {
            fprintf(stderr, "Could not allocate video codec context\n");
            exit(1);
        }
      dec_context->thread_count=2;
      dec_context->width=1280;
      dec_context->height=720;
      dec_context->pix_fmt = AV_PIX_FMT_YUV420P;
      avcodec_open2(dec_context,codec,NULL);
      out_stream = avformat_new_stream(ofmt_ctx, NULL);
      if (!out_stream) {
        fprintf(stderr, "Failed allocating output stream\n");
        ret = AVERROR_UNKNOWN;
        goto end;
      }
      ret = avcodec_parameters_copy(out_stream->codecpar, in_codecpar);
      if (ret < 0) {
        fprintf(stderr, "Failed to copy codec parameters\n");
        goto end;
      }

      avcodec_parameters_copy(abs_ctx->par_in, in_codecpar);
      av_bsf_init(abs_ctx);
      avcodec_parameters_copy(out_stream->codecpar, abs_ctx->par_out);
      // out_stream->codecpar->sample_rate=90000;
      // // //设置输出流的参数
      // out_stream->codecpar->codec_type = AVMEDIA_TYPE_VIDEO;
      out_stream->codecpar->codec_id = AV_CODEC_ID_H264;
      
      // // //设置其他 H.264 相关参数，例如分辨率、帧率、比特率等
      // out_stream->codecpar->width = in_stream->codecpar->width;
      // out_stream->codecpar->height = in_stream->codecpar->height;
      // out_stream->codecpar->bit_rate = in_stream->codecpar->bit_rate;
      // // out_stream->codecpar->sample_rate = in_codecpar->sample_rate;
      // out_stream->codecpar->codec_tag = in_stream->codecpar->codec_tag;
      // out_stream->avg_frame_rate = in_stream->avg_frame_rate;
      // // //设置像素格式为 YUVJ420P
      // out_stream->codecpar->format = AV_PIX_FMT_YUVJ420P;

      // // //设置色彩空间为 bt470bg
      // out_stream->codecpar->color_range = AVCOL_RANGE_JPEG;
      // out_stream->codecpar->color_primaries = AVCOL_PRI_BT709;
      // out_stream->codecpar->color_trc = AVCOL_TRC_BT709 ;
      // out_stream->codecpar->color_space = AVCOL_SPC_BT709;
      in_stream_index = i;
      out_stream_index = out_stream->index;

      break;
    }
  }
  printf("输出流信息:\n");
  av_dump_format(ofmt_ctx, 0, NULL, 1);

  ofmt_ctx->pb = avio_ctx;

  ret = avformat_write_header(ofmt_ctx, NULL);
  if (ret < 0) {
    fprintf(stderr, "Error occurred when opening output file\n");
    goto end;
  }
  AVStream *in_stream, *out_stream;
  out_stream = ofmt_ctx->streams[0];
  // in_stream->codecpar->width=640;
  // in_stream->codecpar->height=480;
  // out_stream->codecpar->width=160;
  // out_stream->codecpar->height=90;
  in_stream = ifmt_ctx->streams[in_stream_index];
  
  while (1) {
    //能保证读出来的是完整的一帧
    ret = av_read_frame(ifmt_ctx, pkt);
    if (ret < 0) {
      for (size_t i = 0; i < ifmt_ctx->nb_streams; i++) {
        av_seek_frame(ifmt_ctx, i, 0, AVSEEK_FLAG_BYTE);
      }
      continue;
    }
    if (pkt->stream_index != in_stream_index) {
      // printf("%i",pkt->stream_index);
      // printf("%i\n",in_stream_index);
      continue;
    }
    
    pts += 40;
    // log_packet(ifmt_ctx, pkt, "in");
    // pkt->pts = pts;
    // pkt->dts = pts;
    // av_bsf_send_packet(abs_ctx, pkt);
    // av_bsf_receive_packet(abs_ctx, pkt);
    // /* copy packet */
    // av_packet_rescale_ts(pkt, in_stream->time_base, out_stream->time_base);
    // pkt->pos = -1;
    // log_packet(ofmt_ctx, pkt, "out");
    
     //注意&的用法是指向指针的指针
      AVFrame *frame=NULL;
      frame = av_frame_alloc();
      frame->format = AV_PIX_FMT_YUV420P;
      frame->width = 1280;
      frame->height = 720;
      AVFrame *newframe=NULL;
      newframe = av_frame_alloc();
      newframe->format = AV_PIX_FMT_YUV420P;
      newframe->width = 1280;
      newframe->height = 720;
      decode(dec_context, frame, pkt);
      av_packet_unref(pkt);
      // if(!(pkt->flags & AV_PKT_FLAG_KEY)){
      //   continue;
      // }
      // if(frame->width!=0)
      // ScaleImg(frame->height,frame->width,frame,frame,90,160);
      // frame->width=160;
      // frame->height=90;
      if(frame->width==0){
      printf("111111\n");
      continue;
      }
      // printf("frame:%i\n",frame->width);
        // cv::Mat mattem;
        if(*weather_number == 0)
            ;
        else if(*weather_number == 1){

          //   if (!shmbuf.readable()) {
          // //               shm_unlink(shm_name);
          // // SharedMemoryBuffer shmbuf(shm_name, 845*480*3*100 + 12);
          //     printf("111111111");
          //     usleep(10000);  // 100ms
          //     continue;
          // }
          //   std::string data;
          //   uint32_t len = shmbuf.read_shm(data);
          //   if (len == 845*480*3) {
          //   cv::Mat frame(480, 845, CV_8UC3, const_cast<char*>(data.data()));
          //   if (!frame.empty()) {
          //   cv::resize(frame,mattem,cv::Size(1280,720));
          //   }
          //   else{
          //     continue;
          //   }
          //   }else{
          //     continue;
          //   }
          ;
        }else if (*weather_number == 15 || *weather_number == 24 || *weather_number == 25) {
           // 读取一个数据包
        while(1){
            ret = av_read_frame(format_ctx1, packet1);
            if (ret < 0) {
                for (size_t i = 0; i < format_ctx1->nb_streams; i++) {
                    av_seek_frame(format_ctx1, i, 0, AVSEEK_FLAG_BYTE);
                }
                std::cout << "视频播放结束，重置到第一帧..." << std::endl;
                av_seek_frame(format_ctx1, video_stream_idx1, 0, AVSEEK_FLAG_BACKWARD);
                avcodec_flush_buffers(codec_ctx1);
                continue;
             }
            // 只处理视频流
            if (packet1->stream_index == video_stream_idx1) {
                // 发送数据包到解码器

                    // 从解码器接收解码后的帧
                        // if(avcodec_receive_frame(codec_ctx1, frame) != 0)
                        // {
                        //   continue;
                        // }
                        decode(codec_ctx1,newframe,packet1);
                        if (!newframe || !newframe->data[0]) {
                            std::cerr << "newframe is not properly allocated or contains no data!" << std::endl;
                            continue;

                        }
                        // av_frame_copy(frame,newframe);
                        // 将 newframe 的内容引用到 frame
                        av_frame_unref(frame);
                        if (av_frame_ref(frame, newframe) < 0) {
                            std::cerr << "Failed to reference newframe to frame!" << std::endl;
                            // 处理错误
                        }
                        // avcodec_send_packet(codec_ctx1, packet1);
                        // avcodec_receive_frame(codec_ctx1, frame);
                        // 处理每一帧 YUV 格式的 AVFrame
                        // std::cout << "Decoded frame " << codec_ctx1->frame_number << std::endl;
                        break;
                        // 在这里对 YUV 格式的帧进行处理或保存（如果需要）
            }else{
                continue;
            }

            // 释放数据包
            av_packet_unref(packet1);
        }
}else if(*weather_number == 27){
           while(1){
            ret = av_read_frame(format_ctx2, packet2);
            if (ret < 0) {
                for (size_t i = 0; i < format_ctx2->nb_streams; i++) {
                    av_seek_frame(format_ctx2, i, 0, AVSEEK_FLAG_BYTE);
                }
                std::cout << "视频播放结束，重置到第一帧..." << std::endl;
                av_seek_frame(format_ctx2, video_stream_idx2, 0, AVSEEK_FLAG_BACKWARD);
                avcodec_flush_buffers(codec_ctx2);
                continue;
             }
            // 只处理视频流
            if (packet2->stream_index == video_stream_idx2) {
                // 发送数据包到解码器

                    // 从解码器接收解码后的帧
                        // if(avcodec_receive_frame(codec_ctx1, frame) != 0)
                        // {
                        //   continue;
                        // }
                        decode(codec_ctx2,newframe,packet2);
                        if (!newframe || !newframe->data[0]) {
                            std::cerr << "newframe is not properly allocated or contains no data!" << std::endl;
                            continue;

                        }
                        // av_frame_copy(frame,newframe);
                        // 将 newframe 的内容引用到 frame
                        av_frame_unref(frame);
                        if (av_frame_ref(frame, newframe) < 0) {
                            std::cerr << "Failed to reference newframe to frame!" << std::endl;
                            // 处理错误
                        }
                        // avcodec_send_packet(codec_ctx1, packet1);
                        // avcodec_receive_frame(codec_ctx1, frame);
                        // 处理每一帧 YUV 格式的 AVFrame
                        // std::cout << "Decoded frame " << codec_ctx1->frame_number << std::endl;
                        break;
                        // 在这里对 YUV 格式的帧进行处理或保存（如果需要）
            }else{
                continue;
            }

            // 释放数据包
            av_packet_unref(packet2);
        }
        }
        // YOLOv8* yolov8=(*yolo_map)["people1"];
        // cv::Size size        = cv::Size{640, 640};
        // int      num_labels  = 40;
        // int      topk        = 100;
        // float    score_thres = 0.7f;
        // float    iou_thres   = 0.7f;//超参数
        // std::vector<Object> objs;
        // objs.clear();
        // cv::Mat  res;
        // yolov8->copy_from_Mat(mattem, size);
        // yolov8->infer();
        // yolov8->postprocess(objs, score_thres, iou_thres, topk, num_labels);
        // yolov8->draw_objects(mattem, res, objs, CLASS_NAMES, COLORS, "11");
        // cv::imwrite("3.jpg",mattem);
    //   // Convert cv::Mat to numpy array
      
      
      // 不能resize，否则需要修改参数后重新打开编码器
      // 导入 Python 模块
    
    // input = cvmat_to_pyopencv(mattem);
    auto start = std::chrono::high_resolution_clock::now();
    previmg = originalinput;               
    originalinput = avframe_to_pyobject(frame,1280,720);
    input = originalinput;
    Py_INCREF(input);
    if (!input) {
        fprintf(stderr, "Failed to create one or more Python objects.\n");
        return 1;
    }
    if (!pModule) {
        PyErr_Print();
        std::cerr << "Failed to import Python module" << std::endl;
        return 1;
    }
    mtx.lock();
      for (const auto subalgorithm : *algorithm) {
        if(subalgorithm.name=="weather")
        { 
            // PyObject* weather_name=PyUnicode_FromString(subalgorithm.tracker_name.c_str());
            PyObject* weather = PyObject_GetAttrString(pModuleweather, "weather");
            if (!weather || !PyCallable_Check(weather)) {
                PyErr_Print();
                std::cerr << "Cannot find function 'process_image'" << std::endl;
                Py_DECREF(weather);
                return 1;
            }
            PyObject *newput;
            PyObject* pArgs = PyTuple_Pack(2,input,input); // 传递参数

            newput = PyObject_CallObject(weather, pArgs); // 调用函数
            // Py_DECREF(weather_name);
            Py_DECREF(pArgs);
            Py_DECREF(input);
            input=newput;
            // Py_DECREF(output);
            //释放之前的会导致最后一次的未释放
            Py_DECREF(weather);
            if (!input) {
              PyErr_Print();
              std::cerr << "Call failed" << std::endl;
              return 1;
          }
        }
        if(subalgorithm.name=="weather_detect")
        { 
            // PyObject* weather_name=PyUnicode_FromString(subalgorithm.tracker_name.c_str());
            PyObject* weather_detect = PyObject_GetAttrString(pModuleweather, "fake_weather_detect");
            if (!weather_detect || !PyCallable_Check(weather_detect)) {
                PyErr_Print();
                std::cerr << "Cannot find function 'process_image'" << std::endl;
                Py_DECREF(weather_detect);
                return 1;
            }
            PyObject *newput;
            PyObject* pArgs = PyTuple_Pack(2,input,input); // 传递参数

            newput = PyObject_CallObject(weather_detect, pArgs); // 调用函数
            // Py_DECREF(weather_name);
            Py_DECREF(pArgs);
            Py_DECREF(input);
            input=newput;
            // Py_DECREF(output);
            //释放之前的会导致最后一次的未释放
            Py_DECREF(weather_detect);
            if (!input) {
              PyErr_Print();
              std::cerr << "Call failed" << std::endl;
              return 1;
          }
        }
        if(subalgorithm.name=="mean_filter")
        {
            PyObject* width=PyLong_FromLong(subalgorithm.width);
            PyObject* height=PyLong_FromLong(subalgorithm.height);
            PyObject* mean_filter = PyObject_GetAttrString(pModule, "mean_filter");
            if (!mean_filter || !PyCallable_Check(mean_filter)) {
                PyErr_Print();
                std::cerr << "Cannot find function 'process_image'" << std::endl;
                Py_DECREF(mean_filter);
                return 1;
            }
             PyObject *newput;
            PyObject* pArgs = PyTuple_Pack(3, input,width,height); // 传递参数
            newput = PyObject_CallObject(mean_filter, pArgs); // 调用函数
            Py_DECREF(width);
            Py_DECREF(height);
            Py_DECREF(pArgs);
            Py_DECREF(input);
            input=newput;
            //释放之前的会导致最后一次的未释放
            Py_DECREF(mean_filter);
            if (!input) {
              PyErr_Print();
              std::cerr << "Call failed" << std::endl;
              return 1;
          }
        }
        if(subalgorithm.name=="adjust_contrast_and_lightness")
        
        {   
            PyObject* lights=PyLong_FromLong(subalgorithm.lights);
            PyObject* contrasts=PyLong_FromLong(subalgorithm.contrasts);
            PyObject* adjust_contrast_and_lightness = PyObject_GetAttrString(pModule, "adjust_contrast_and_lightness");
            if (!adjust_contrast_and_lightness || !PyCallable_Check(adjust_contrast_and_lightness)) {
                PyErr_Print();
                std::cerr << "Cannot find function 'process_image'" << std::endl;
                Py_DECREF(adjust_contrast_and_lightness);
                return 1;
            }
            PyObject *newput;
            PyObject* pArgs = PyTuple_Pack(3, input,lights,contrasts); // 传递参数
            newput = PyObject_CallObject(adjust_contrast_and_lightness, pArgs); // 调用函数
            Py_DECREF(lights);
            Py_DECREF(contrasts);
            Py_DECREF(pArgs);
            Py_DECREF(input);
            input=newput;
            Py_DECREF(adjust_contrast_and_lightness);
            if (!input) {
              PyErr_Print();
              std::cerr << "Call failed" << std::endl;
              return 1;
          }
        }
        if(subalgorithm.name=="box_filter") 
        {    
            PyObject* width=PyLong_FromLong(subalgorithm.width);
            PyObject* height=PyLong_FromLong(subalgorithm.height);
            PyObject* box_filter = PyObject_GetAttrString(pModule, "box_filter");
            if (!box_filter || !PyCallable_Check(box_filter)) {
                PyErr_Print();
                std::cerr << "Cannot find function 'process_image'" << std::endl;
                Py_DECREF(box_filter);
                return 1;
            }
            PyObject *newput;
            PyObject* pArgs = PyTuple_Pack(3, input,width,height); // 传递参数
            newput = PyObject_CallObject(box_filter, pArgs); // 调用函数
            Py_DECREF(width);
            Py_DECREF(height);
            Py_DECREF(pArgs);
            Py_DECREF(input);
            input=newput;
            Py_DECREF(box_filter);
            if (!input) {
              PyErr_Print();
              std::cerr << "Call failed" << std::endl;
              return 1;
          }
        }
        if(subalgorithm.name=="gaussian_filter") 
        {   
            PyObject* width=PyLong_FromLong(subalgorithm.width);
            PyObject* height=PyLong_FromLong(subalgorithm.height);
            PyObject* gaussian_filter = PyObject_GetAttrString(pModule, "gaussian_filter");
            if (!gaussian_filter || !PyCallable_Check(gaussian_filter)) {
                PyErr_Print();
                std::cerr << "Cannot find function 'process_image'" << std::endl;
                Py_DECREF(gaussian_filter);
                return 1;
            }
            PyObject *newput;
            PyObject* pArgs = PyTuple_Pack(3, input,width,height); // 传递参数
            newput = PyObject_CallObject(gaussian_filter, pArgs); // 调用函数
            Py_DECREF(width);
            Py_DECREF(height);
            Py_DECREF(pArgs);
            Py_DECREF(input);
            input=newput;
            Py_DECREF(gaussian_filter);
            if (!input) {
              PyErr_Print();
              std::cerr << "Call failed" << std::endl;
              return 1;
          }
        }
        if(subalgorithm.name=="updown") 
        {   
            PyObject* updown = PyObject_GetAttrString(pModule, "updown");
            if (!updown || !PyCallable_Check(updown)) {
                PyErr_Print();
                std::cerr << "Cannot find function 'process_image'" << std::endl;
                Py_DECREF(updown);
                return 1;
            }
            PyObject *newput;
            PyObject* pArgs = PyTuple_Pack(1, input); // 传递参数
            newput = PyObject_CallObject(updown, pArgs); // 调用函数
            Py_DECREF(pArgs);
            Py_DECREF(input);
            input=newput;
            Py_DECREF(updown);
            if (!input) {
              PyErr_Print();
              std::cerr << "Call failed" << std::endl;
              return 1;
          }
        }
        if(subalgorithm.name=="hist_equal") 
        {   
            PyObject* hist_equal = PyObject_GetAttrString(pModule, "hist_equal");
            if (!hist_equal || !PyCallable_Check(hist_equal)) {
                PyErr_Print();
                std::cerr << "Cannot find function 'process_image'" << std::endl;
                Py_DECREF(hist_equal);
                return 1;
            }
            PyObject *newput;
            PyObject* pArgs = PyTuple_Pack(1, input); // 传递参数
            newput = PyObject_CallObject(hist_equal, pArgs); // 调用函数
            Py_DECREF(pArgs);
            Py_DECREF(input);
            input=newput;
            Py_DECREF(hist_equal);
            if (!input) {
              PyErr_Print();
              std::cerr << "Call failed" << std::endl;
              return 1;
          }
        }
        if(subalgorithm.name=="bilateral_filter") 
        {    
            PyObject* d=PyLong_FromLong(subalgorithm.d);
            PyObject* color=PyLong_FromLong(subalgorithm.color);
            PyObject* space=PyLong_FromLong(subalgorithm.space);
            PyObject* bilateral_filter = PyObject_GetAttrString(pModule, "bilateral_filter");
            if (!bilateral_filter || !PyCallable_Check(bilateral_filter)) {
                PyErr_Print();
                std::cerr << "Cannot find function 'process_image'" << std::endl;
                Py_DECREF(bilateral_filter);
                return 1;
            }
            PyObject *newput;
            PyObject* pArgs = PyTuple_Pack(4,input,d,color,space); // 传递参数
            newput = PyObject_CallObject(bilateral_filter, pArgs); // 调用函数
            Py_DECREF(d);
            Py_DECREF(color);
            Py_DECREF(space);
            Py_DECREF(pArgs);
            Py_DECREF(input);
            input=newput;
            Py_DECREF(bilateral_filter);
            if (!input) {
              PyErr_Print();
              std::cerr << "Call failed" << std::endl;
              return 1;
          }
        }
        if(subalgorithm.name=="alpha_filter") 
        {    
            PyObject* d=PyLong_FromLong(subalgorithm.d);
            PyObject* width=PyLong_FromLong(subalgorithm.width);
            PyObject* height=PyLong_FromLong(subalgorithm.height);
            PyObject* alpha_filter = PyObject_GetAttrString(pModule, "alpha_filter");
            if (!alpha_filter || !PyCallable_Check(alpha_filter)) {
                PyErr_Print();
                std::cerr << "Cannot find function 'process_image'" << std::endl;
                Py_DECREF(alpha_filter);
                return 1;
            }
            PyObject *newput;
            PyObject* pArgs = PyTuple_Pack(4,input,d,width,height); // 传递参数
            newput = PyObject_CallObject(alpha_filter, pArgs); // 调用函数
            Py_DECREF(d);
            Py_DECREF(width);
            Py_DECREF(height);
            Py_DECREF(pArgs);
            Py_DECREF(input);
            input=newput;
            Py_DECREF(alpha_filter);
            if (!input) {
              PyErr_Print();
              std::cerr << "Call failed" << std::endl;
              return 1;
          }
        }
        if(subalgorithm.name=="median_filter") 
        {    
            PyObject* width=PyLong_FromLong(subalgorithm.width);
            PyObject* height=PyLong_FromLong(subalgorithm.height);
            PyObject* median_filter = PyObject_GetAttrString(pModule, "median_filter");
            if (!median_filter || !PyCallable_Check(median_filter)) {
                PyErr_Print();
                std::cerr << "Cannot find function 'process_image'" << std::endl;
                Py_DECREF(median_filter);
                return 1;
            }
            PyObject *newput;
            PyObject* pArgs = PyTuple_Pack(3,input,width,height); // 传递参数
            newput = PyObject_CallObject(median_filter, pArgs); // 调用函数
            Py_DECREF(width);
            Py_DECREF(height);
            Py_DECREF(pArgs);
            Py_DECREF(input);
            input=newput;
            Py_DECREF(median_filter);
            if (!input) {
              PyErr_Print();
              std::cerr << "Call failed" << std::endl;
              return 1;
          }
        }
        if(subalgorithm.name=="max_filter") 
        {    
            PyObject* width=PyLong_FromLong(subalgorithm.width);
            PyObject* height=PyLong_FromLong(subalgorithm.height);
            PyObject* max_filter = PyObject_GetAttrString(pModule, "max_filter");
            if (!max_filter || !PyCallable_Check(max_filter)) {
                PyErr_Print();
                std::cerr << "Cannot find function 'process_image'" << std::endl;
                Py_DECREF(max_filter);
                return 1;
            }
            PyObject *newput;
            PyObject* pArgs = PyTuple_Pack(3,input,width,height); // 传递参数
            newput = PyObject_CallObject(max_filter, pArgs); // 调用函数
            Py_DECREF(width);
            Py_DECREF(height);
            Py_DECREF(pArgs);
            Py_DECREF(input);
            input=newput;
            Py_DECREF(max_filter);
            if (!input) {
              PyErr_Print();
              std::cerr << "Call failed" << std::endl;
              return 1;
          }
        }
        if(subalgorithm.name=="min_filter") 
        {    
            PyObject* width=PyLong_FromLong(subalgorithm.width);
            PyObject* height=PyLong_FromLong(subalgorithm.height);
            PyObject* min_filter = PyObject_GetAttrString(pModule, "min_filter");
            if (!min_filter || !PyCallable_Check(min_filter)) {
                PyErr_Print();
                std::cerr << "Cannot find function 'process_image'" << std::endl;
                Py_DECREF(min_filter);
                return 1;
            }
            PyObject *newput;
            PyObject* pArgs = PyTuple_Pack(3,input,width,height); // 传递参数
            newput = PyObject_CallObject(min_filter, pArgs); // 调用函数
            Py_DECREF(width);
            Py_DECREF(height);
            Py_DECREF(pArgs);
            Py_DECREF(input);
            input=newput;
            Py_DECREF(min_filter);
            if (!input) {
              PyErr_Print();
              std::cerr << "Call failed" << std::endl;
              return 1;
          }
        }
        if(subalgorithm.name=="middle_filter") 
        {    
            PyObject* width=PyLong_FromLong(subalgorithm.width);
            PyObject* height=PyLong_FromLong(subalgorithm.height);
            PyObject* middle_filter = PyObject_GetAttrString(pModule, "middle_filter");
            if (!middle_filter || !PyCallable_Check(middle_filter)) {
                PyErr_Print();
                std::cerr << "Cannot find function 'process_image'" << std::endl;
                Py_DECREF(middle_filter);
                return 1;
            }
            PyObject *newput;
            PyObject* pArgs = PyTuple_Pack(3,input,width,height); // 传递参数
            newput = PyObject_CallObject(middle_filter, pArgs); // 调用函数
            Py_DECREF(width);
            Py_DECREF(height);
            Py_DECREF(pArgs);
            Py_DECREF(input);
            input=newput;
            Py_DECREF(middle_filter);
            if (!input) {
              PyErr_Print();
              std::cerr << "Call failed" << std::endl;
              return 1;
          }
        }
        // if(subalgorithm.name=="hist_match") 
        // {   
        //     PyObject* ref = cvmat_to_pyopencv(mat_hist_match);
        //     PyObject* hist_match = PyObject_GetAttrString(pModule, "hist_match");
        //     if (!hist_match || !PyCallable_Check(hist_match)) {
        //         PyErr_Print();
        //         std::cerr << "Cannot find function 'process_image'" << std::endl;
        //         Py_DECREF(hist_match);
        //         return 1;
        //     }
        //     PyObject *newput;
        //     PyObject* pArgs = PyTuple_Pack(2,input,ref); // 传递参数
        //     newput = PyObject_CallObject(hist_match, pArgs); // 调用函数
        //     auto end = std::chrono::high_resolution_clock::now();//计时结束
        //     std::chrono::duration<double> elapsed = end - start;
        //     std::cout << "The run time is: " << elapsed.count() << "s" << std::endl;
        //     Py_DECREF(ref);
        //     Py_DECREF(pArgs);
        //     Py_DECREF(input);
        //     input=newput;
        //     Py_DECREF(hist_match);
        //     if (!input) {
        //       PyErr_Print();
        //       std::cerr << "Call failed" << std::endl;
        //       return 1;
        //   }
        // }
        // if(subalgorithm.name=="curve_adjust") 
        // {   
        //     PyObject* r_1=PyLong_FromLong(subalgorithm.r_1);
        //     PyObject* r_2=PyLong_FromLong(subalgorithm.r_2);
        //     PyObject* g_1=PyLong_FromLong(subalgorithm.g_1);
        //     PyObject* g_2=PyLong_FromLong(subalgorithm.g_2);
        //     PyObject* b_1=PyLong_FromLong(subalgorithm.b_1);
        //     PyObject* b_2=PyLong_FromLong(subalgorithm.b_2);
        //     PyObject* curve_adjust = PyObject_GetAttrString(pModule, "curve_adjust");
        //     if (!curve_adjust || !PyCallable_Check(curve_adjust)) {
        //         PyErr_Print();
        //         std::cerr << "Cannot find function 'process_image'" << std::endl;
        //         Py_DECREF(curve_adjust);
        //         return 1;
        //     }
        //     PyObject *newput;
        //     PyObject* pArgs = PyTuple_Pack(7,input,r_1,r_2,g_1,g_2,b_1,b_2); // 传递参数
        //     newput = PyObject_CallObject(curve_adjust, pArgs); // 调用函数
        //     Py_DECREF(r_1);
        //     Py_DECREF(r_2);
        //     Py_DECREF(g_1);
        //     Py_DECREF(g_2);
        //     Py_DECREF(b_1);
        //     Py_DECREF(b_2);
        //     Py_DECREF(pArgs);
        //     Py_DECREF(input);
        //     input=newput;
        //     Py_DECREF(curve_adjust);
        //     if (!input) {
        //       PyErr_Print();
        //       std::cerr << "Call failed" << std::endl;
        //       return 1;
        //   }
        // }
        if(subalgorithm.name=="dehaze_clahe") 
        {   
            PyObject* limits=PyLong_FromLong(subalgorithm.limits);
            PyObject* grids=PyLong_FromLong(subalgorithm.grids);
            PyObject* dehaze_clahe = PyObject_GetAttrString(pModule, "dehaze_clahe");
            if (!dehaze_clahe || !PyCallable_Check(dehaze_clahe)) {
                PyErr_Print();
                std::cerr << "Cannot find function 'process_image'" << std::endl;
                Py_DECREF(dehaze_clahe);
                return 1;
            }
            PyObject *newput;
            PyObject* pArgs = PyTuple_Pack(3,input,limits,grids); // 传递参数
            newput = PyObject_CallObject(dehaze_clahe, pArgs); // 调用函数
            Py_DECREF(limits);
            Py_DECREF(grids);
            Py_DECREF(pArgs);
            Py_DECREF(input);
            input=newput;
            Py_DECREF(dehaze_clahe);
            if (!input) {
              PyErr_Print();
              std::cerr << "Call failed" << std::endl;
              return 1;
          }
        }
        if(subalgorithm.name=="dehaze_Pkel") 
        {   
            PyObject* ratios=PyLong_FromLong(subalgorithm.ratios);
            PyObject* radii=PyLong_FromLong(subalgorithm.radii);
            PyObject* dehaze_Pkel = PyObject_GetAttrString(pModule, "dehaze_Pkel");
            if (!dehaze_Pkel || !PyCallable_Check(dehaze_Pkel)) {
                PyErr_Print();
                std::cerr << "Cannot find function 'process_image'" << std::endl;
                Py_DECREF(dehaze_Pkel);
                return 1;
            }
            
            PyObject* pArgs = PyTuple_Pack(3,input,ratios,radii); // 传递参数
            PyObject *newput;
            newput = PyObject_CallObject(dehaze_Pkel, pArgs); // 调用函数
            Py_DECREF(ratios);
            Py_DECREF(radii);
            Py_DECREF(pArgs);
            Py_DECREF(input);
            input=newput;
            Py_DECREF(dehaze_Pkel);
            if (!input) {
              PyErr_Print();
              std::cerr << "Call failed" << std::endl;
              return 1;
          }
        }
        if(subalgorithm.name=="dehaze_xybc") 
        {   
            PyObject* weights=PyLong_FromLong(subalgorithm.ratios);
            PyObject* kers=PyLong_FromLong(subalgorithm.radii);
            PyObject* dehaze_xybc = PyObject_GetAttrString(pModule, "dehaze_xybc");
            if (!dehaze_xybc || !PyCallable_Check(dehaze_xybc)) {
                PyErr_Print();
                std::cerr << "Cannot find function 'process_image'" << std::endl;
                Py_DECREF(dehaze_xybc);
                return 1;
            }
            
            PyObject* pArgs = PyTuple_Pack(3,input,weights,kers); // 传递参数
            PyObject *newput;
            newput = PyObject_CallObject(dehaze_xybc, pArgs); // 调用函数
            Py_DECREF(weights);
            Py_DECREF(kers);
            Py_DECREF(pArgs);
            Py_DECREF(input);
            input=newput;
            Py_DECREF(dehaze_xybc);
            if (!input) {
              PyErr_Print();
              std::cerr << "Call failed" << std::endl;
              return 1;
          }
        }
        if(subalgorithm.name=="edgedetect_Scharr") 
        {   
            PyObject* edgedetect_Scharr = PyObject_GetAttrString(pModule, "edgedetect_Scharr");
            if (!edgedetect_Scharr || !PyCallable_Check(edgedetect_Scharr)) {
                PyErr_Print();
                std::cerr << "Cannot find function 'process_image'" << std::endl;
                Py_DECREF(edgedetect_Scharr);
                return 1;
            }
            
            PyObject* pArgs = PyTuple_Pack(1,input); // 传递参数
            PyObject *newput;
            newput = PyObject_CallObject(edgedetect_Scharr, pArgs); // 调用函数
            Py_DECREF(pArgs);
            Py_DECREF(input);
            input=newput;
            Py_DECREF(edgedetect_Scharr);
            if (!input) {
              PyErr_Print();
              std::cerr << "Call failed" << std::endl;
              return 1;
          }
        }
        if(subalgorithm.name=="edgedetect_Canny") 
        {   
            PyObject* edgedetect_Canny = PyObject_GetAttrString(pModule, "edgedetect_Canny");
            if (!edgedetect_Canny || !PyCallable_Check(edgedetect_Canny)) {
                PyErr_Print();
                std::cerr << "Cannot find function 'process_image'" << std::endl;
                Py_DECREF(edgedetect_Canny);
                return 1;
            }
            
            PyObject* pArgs = PyTuple_Pack(1,input); // 传递参数
            PyObject *newput;
            newput = PyObject_CallObject(edgedetect_Canny, pArgs); // 调用函数
            Py_DECREF(pArgs);
            Py_DECREF(input);
            input=newput;
            Py_DECREF(edgedetect_Canny);
            if (!input) {
              PyErr_Print();
              std::cerr << "Call failed" << std::endl;
              return 1;
          }
        }
        if(subalgorithm.name=="significance_test") 
        {   
            PyObject* significance_test = PyObject_GetAttrString(pModule, "significance_test");
            if (!significance_test || !PyCallable_Check(significance_test)) {
                PyErr_Print();
                std::cerr << "Cannot find function 'process_image'" << std::endl;
                Py_DECREF(significance_test);
                return 1;
            }
            
            PyObject* pArgs = PyTuple_Pack(1,input); // 传递参数
            PyObject *newput;
            newput = PyObject_CallObject(significance_test, pArgs); // 调用函数
            Py_DECREF(pArgs);
            Py_DECREF(input);
            input=newput;
            Py_DECREF(significance_test);
            if (!input) {
              PyErr_Print();
              std::cerr << "Call failed" << std::endl;
              return 1;
          }
        }
        if(subalgorithm.name=="edgedetect_LOG") 
        {   
            PyObject* edgedetect_LOG = PyObject_GetAttrString(pModule, "edgedetect_LOG");
            if (!edgedetect_LOG || !PyCallable_Check(edgedetect_LOG)) {
                PyErr_Print();
                std::cerr << "Cannot find function 'process_image'" << std::endl;
                Py_DECREF(edgedetect_LOG);
                return 1;
            }
            
            PyObject* pArgs = PyTuple_Pack(1,input); // 传递参数
            PyObject *newput;
            newput = PyObject_CallObject(edgedetect_LOG, pArgs); // 调用函数
            Py_DECREF(pArgs);
            Py_DECREF(input);
            input=newput;
            Py_DECREF(edgedetect_LOG);
            if (!input) {
              PyErr_Print();
              std::cerr << "Call failed" << std::endl;
              return 1;
          }
        }
        if(subalgorithm.name=="binarization") 
        {   
            PyObject* methods=PyLong_FromLong(subalgorithm.radii);
            PyObject* binarization = PyObject_GetAttrString(pModule, "binarization");
            if (!binarization || !PyCallable_Check(binarization)) {
                PyErr_Print();
                std::cerr << "Cannot find function 'process_image'" << std::endl;
                Py_DECREF(binarization);
                return 1;
            }
            
            PyObject* pArgs = PyTuple_Pack(2,input,methods); // 传递参数
            PyObject *newput;
            newput = PyObject_CallObject(binarization, pArgs); // 调用函数
            Py_DECREF(methods);
            Py_DECREF(pArgs);
            Py_DECREF(input);
            input=newput;
            Py_DECREF(binarization);
            if (!input) {
              PyErr_Print();
              std::cerr << "Call failed" << std::endl;
              return 1;
          }
        }
        if(subalgorithm.name=="sharpen_canny") 
        {   
            PyObject* thre_1=PyLong_FromLong(subalgorithm.width);
            PyObject* thre_2=PyLong_FromLong(subalgorithm.height);
            PyObject* sharpenthes=PyLong_FromLong(subalgorithm.contrasts);
            PyObject* sharpen_canny = PyObject_GetAttrString(pModule, "sharpen_canny");
            if (!sharpen_canny || !PyCallable_Check(sharpen_canny)) {
                PyErr_Print();
                std::cerr << "Cannot find function 'process_image'" << std::endl;
                Py_DECREF(sharpen_canny);
                return 1;
            }
            
            PyObject* pArgs = PyTuple_Pack(4,input,thre_1,thre_2,sharpenthes); // 传递参数
            PyObject *newput;
            newput = PyObject_CallObject(sharpen_canny, pArgs); // 调用函数
            Py_DECREF(thre_1);
            Py_DECREF(thre_2);
            Py_DECREF(sharpenthes);
            Py_DECREF(pArgs);
            Py_DECREF(input);
            input=newput;
            Py_DECREF(sharpen_canny);
            if (!input) {
              PyErr_Print();
              std::cerr << "Call failed" << std::endl;
              return 1;
          }
        }
        if(subalgorithm.name=="sharpen_PreLOG") 
        {   
            PyObject* weights=PyLong_FromLong(subalgorithm.width);
            PyObject* sizes=PyLong_FromLong(subalgorithm.height);
            PyObject* sharpenthes=PyLong_FromLong(subalgorithm.contrasts);
            PyObject* sharpen_PreLOG = PyObject_GetAttrString(pModule, "sharpen_PreLOG");
            if (!sharpen_PreLOG || !PyCallable_Check(sharpen_PreLOG)) {
                PyErr_Print();
                std::cerr << "Cannot find function 'process_image'" << std::endl;
                Py_DECREF(sharpen_PreLOG);
                return 1;
            }
            
            PyObject* pArgs = PyTuple_Pack(4,input,weights,sizes,sharpenthes); // 传递参数
            PyObject *newput;
            newput = PyObject_CallObject(sharpen_PreLOG, pArgs); // 调用函数
            Py_DECREF(weights);
            Py_DECREF(sizes);
            Py_DECREF(sharpenthes);
            Py_DECREF(pArgs);
            Py_DECREF(input);
            input=newput;
            Py_DECREF(sharpen_PreLOG);
            if (!input) {
              PyErr_Print();
              std::cerr << "Call failed" << std::endl;
              return 1;
          }
        }
        if(subalgorithm.name=="sharpen_sos") 
        {   
            PyObject* weights=PyLong_FromLong(subalgorithm.width);
            PyObject* sharpenthes=PyLong_FromLong(subalgorithm.contrasts);
            PyObject* sharpen_sos = PyObject_GetAttrString(pModule, "sharpen_sos");
            if (!sharpen_sos || !PyCallable_Check(sharpen_sos)) {
                PyErr_Print();
                std::cerr << "Cannot find function 'process_image'" << std::endl;
                Py_DECREF(sharpen_sos);
                return 1;
            }
            
            PyObject* pArgs = PyTuple_Pack(3,input,weights,sharpenthes); // 传递参数
            PyObject *newput;
            newput = PyObject_CallObject(sharpen_sos, pArgs); // 调用函数
            Py_DECREF(weights);
            Py_DECREF(sharpenthes);
            Py_DECREF(pArgs);
            Py_DECREF(input);
            input=newput;
            Py_DECREF(sharpen_sos);
            if (!input) {
              PyErr_Print();
              std::cerr << "Call failed" << std::endl;
              return 1;
          }
        }
        if(subalgorithm.name=="moving_average_pic")
        {   
            int grayNo = subalgorithm.height;
            PyObject* alpha=PyFloat_FromDouble(static_cast<double>(subalgorithm.contrasts));
            PyObject* moving_average_pic = PyObject_GetAttrString(pModule, "moving_average_pic");
            if (!moving_average_pic || !PyCallable_Check(moving_average_pic)) {
                PyErr_Print();
                std::cerr << "Cannot find function 'process_image'" << std::endl;
                Py_DECREF(moving_average_pic);
                return 1;
            }
            
            PyObject* pArgs = PyTuple_Pack(3,input,gray,alpha); // 传递参数
            PyObject *newput;
            newput = PyObject_CallObject(moving_average_pic, pArgs); // 调用函数
            Py_DECREF(alpha);
            Py_DECREF(pArgs);
            Py_DECREF(input);
            input=newput;
            Py_DECREF(moving_average_pic);
            if (!input) {
              PyErr_Print();
              std::cerr << "Call failed" << std::endl;
              return 1;
          }
        }
        if(subalgorithm.name=="PCA_infrared_color_fusion")
        {   
            int grayNo = subalgorithm.height;
            PyObject* alpha=PyFloat_FromDouble(static_cast<double>(subalgorithm.contrasts));
            PyObject* PCA_infrared_color_fusion = PyObject_GetAttrString(pModule, "PCA_infrared_color_fusion");
            if (!PCA_infrared_color_fusion || !PyCallable_Check(PCA_infrared_color_fusion)) {
                PyErr_Print();
                std::cerr << "Cannot find function 'process_image'" << std::endl;
                Py_DECREF(PCA_infrared_color_fusion);
                return 1;
            }
            
            PyObject* pArgs = PyTuple_Pack(3,input,gray,alpha); // 传递参数
            PyObject *newput;
            newput = PyObject_CallObject(PCA_infrared_color_fusion, pArgs); // 调用函数
            Py_DECREF(alpha);
            Py_DECREF(pArgs);
            Py_DECREF(input);
            input=newput;
            Py_DECREF(PCA_infrared_color_fusion);
            if (!input) {
              PyErr_Print();
              std::cerr << "Call failed" << std::endl;
              return 1;
          }
        }
        if(subalgorithm.name=="CDDFuse")
        {   
            int grayNo = subalgorithm.height;
            PyObject* wavelet=PyUnicode_FromString(subalgorithm.tracker_name.c_str());
            PyObject* level=PyLong_FromLong(subalgorithm.r_1);
            PyObject* CDDFuse = PyObject_GetAttrString(pModule, "CDDFuse");
            if (!CDDFuse || !PyCallable_Check(CDDFuse)) {
                PyErr_Print();
                std::cerr << "Cannot find function 'process_image'" << std::endl;
                Py_DECREF(CDDFuse);
                return 1;
            }
            
            PyObject* pArgs = PyTuple_Pack(3,input,gray,wavelet,level); // 传递参数
            PyObject *newput;
            newput = PyObject_CallObject(CDDFuse, pArgs); // 调用函数
            Py_DECREF(wavelet);
            Py_DECREF(level);
            Py_DECREF(pArgs);
            Py_DECREF(input);
            input=newput;
            Py_DECREF(CDDFuse);
            if (!input) {
              PyErr_Print();
              std::cerr << "Call failed" << std::endl;
              return 1;
          }
        }
        if(subalgorithm.name=="FusionGAN")
        {   
            int grayNo = subalgorithm.height;
            PyObject* levels=PyLong_FromLong(subalgorithm.r_1);
            PyObject* alpha=PyFloat_FromDouble(static_cast<double>(subalgorithm.contrasts));
            PyObject* FusionGAN = PyObject_GetAttrString(pModule, "FusionGAN");
            if (!FusionGAN || !PyCallable_Check(FusionGAN)) {
                PyErr_Print();
                std::cerr << "Cannot find function 'process_image'" << std::endl;
                Py_DECREF(FusionGAN);
                return 1;
            }
            
            PyObject* pArgs = PyTuple_Pack(3,input,gray,levels,alpha); // 传递参数
            PyObject *newput;
            newput = PyObject_CallObject(FusionGAN, pArgs); // 调用函数
            Py_DECREF(alpha);
            Py_DECREF(levels);
            Py_DECREF(pArgs);
            Py_DECREF(input);
            input=newput;
            Py_DECREF(FusionGAN);
            if (!input) {
              PyErr_Print();
              std::cerr << "Call failed" << std::endl;
              return 1;
          }
        }
        if(subalgorithm.name=="min_max_blend")
        {   
            int grayNo = subalgorithm.height;
            int minmax = subalgorithm.width;
            std::string minmaxstr;
            if(minmax == 0)
            {
              minmaxstr = "max";
            }else{
              minmaxstr = "min";
            }
            PyObject* min_max_str = PyUnicode_FromString(minmaxstr.c_str());
            PyObject* min_max_blend = PyObject_GetAttrString(pModule, "min_max_blend");
            if (!min_max_blend || !PyCallable_Check(min_max_blend)) {
                PyErr_Print();
                std::cerr << "Cannot find function 'process_image'" << std::endl;
                Py_DECREF(min_max_blend);
                return 1;
            }
            
            PyObject* pArgs = PyTuple_Pack(3,input,gray,min_max_str); // 传递参数
            PyObject *newput;
            newput = PyObject_CallObject(min_max_blend, pArgs); // 调用函数
            Py_DECREF(min_max_str);
            Py_DECREF(pArgs);
            Py_DECREF(input);
            input=newput;
            Py_DECREF(min_max_blend);
            if (!input) {
              PyErr_Print();
              std::cerr << "Call failed" << std::endl;
              return 1;
          }
        }
        if(subalgorithm.name=="detect")
        {   
            PyObject* detect = PyObject_GetAttrString(pModuleDetect, "detect");
            PyObject* method=PyUnicode_FromString(subalgorithm.tracker_name.c_str());
            if (!detect || !PyCallable_Check(detect)) {
                PyErr_Print();
                std::cerr << "Cannot find function 'process_image'" << std::endl;
                Py_DECREF(detect);
                return 1;
            }
            
            PyObject* pArgs = PyTuple_Pack(2,input,method); // 传递参数
            PyObject *newput;
            newput = PyObject_CallObject(detect, pArgs); // 调用函数
            Py_DECREF(method);
            Py_DECREF(pArgs);
            Py_DECREF(input);
            input=newput;
            Py_DECREF(detect);
            if (!input) {
              PyErr_Print();
              std::cerr << "Call failed" << std::endl;
              return 1;
          }
        }
        if(subalgorithm.name=="mmtracking")
        {   
            PyObject* detect = PyObject_GetAttrString(pModulemmtracker, "mmtracking");
            if (!detect || !PyCallable_Check(detect)) {
                PyErr_Print();
                std::cerr << "Cannot find function 'process_image'" << std::endl;
                Py_DECREF(detect);
                return 1;
            }
            
            PyObject* pArgs = PyTuple_Pack(1,input); // 传递参数
            PyObject *newput;
            newput = PyObject_CallObject(detect, pArgs); // 调用函数
            Py_DECREF(pArgs);
            Py_DECREF(input);
            input=newput;
            Py_DECREF(detect);
            if (!input) {
              PyErr_Print();
              std::cerr << "Call failed" << std::endl;
              return 1;
          }
        }
        if(subalgorithm.name=="mmtrack")
        {   
            PyObject* detect = PyObject_GetAttrString(pModuleDetect, "mmtracking");
            PyObject* method = PyUnicode_FromString(subalgorithm.tracker_name.c_str());
            if (!detect || !PyCallable_Check(detect)) {
                PyErr_Print();
                std::cerr << "Cannot find function 'process_image'" << std::endl;
                Py_DECREF(detect);
                return 1;
            }
            
            PyObject* pArgs = PyTuple_Pack(2,input,method); // 传递参数
            PyObject *newput;
            newput = PyObject_CallObject(detect, pArgs); // 调用函数
            Py_DECREF(method);
            Py_DECREF(pArgs);
            Py_DECREF(input);
            input=newput;
            Py_DECREF(detect);
            if (!input) {
              PyErr_Print();
              std::cerr << "Call failed" << std::endl;
              return 1;
          }
        }
        if(subalgorithm.name=="SOT")
        {   
            PyObject* height=PyLong_FromLong(subalgorithm.height);
            PyObject* width=PyLong_FromLong(subalgorithm.width);
            PyObject* x=PyLong_FromLong(subalgorithm.d);
            PyObject* y=PyLong_FromLong(subalgorithm.color);
            PyObject* learning_rate=PyFloat_FromDouble(static_cast<double>(subalgorithm.contrasts));
            PyObject* tracker_name=PyUnicode_FromString(subalgorithm.tracker_name.c_str());
            PyObject* SOT = PyObject_GetAttrString(pModuleSOT, "SOT");
            if (!SOT || !PyCallable_Check(SOT)) {
                PyErr_Print();
                std::cerr << "Cannot find function 'process_image'" << std::endl;
                Py_DECREF(SOT);
                return 1;
            }
            
            PyObject* pArgs = PyTuple_Pack(7,input,tracker_name,learning_rate,x,y,width,height); // 传递参数
            PyObject *newput;
            newput = PyObject_CallObject(SOT, pArgs); // 调用函数
            Py_DECREF(width);
            Py_DECREF(height);
            Py_DECREF(x);
            Py_DECREF(y);
            Py_DECREF(learning_rate);
            Py_DECREF(tracker_name);
            Py_DECREF(pArgs);
            Py_DECREF(input);
            input=newput;
            Py_DECREF(SOT);
            if (!input) {
              PyErr_Print();
              std::cerr << "Call failed" << std::endl;
              return 1;
          }
        }
        if(subalgorithm.name=="saoshe"){
            
            PyObject* seq_stitch = PyObject_GetAttrString(pModuleStitch, "seq_stitch");
            if (!seq_stitch || !PyCallable_Check(seq_stitch)) {
                PyErr_Print();
                std::cerr << "Cannot find function 'process_image'" << std::endl;
                Py_DECREF(seq_stitch);
                return 1;
            }
            PyObject *newput;
            PyObject* pArgs = PyTuple_Pack(2,previmg,input); // 传递参数
            newput = PyObject_CallObject(seq_stitch, pArgs); // 调用函数
            Py_DECREF(pArgs);
            Py_DECREF(input);
            input=newput;
            Py_DECREF(seq_stitch);
            if (!input) {
              PyErr_Print();
              std::cerr << "Call failed" << std::endl;
              return 1;
          }
        }
        if(subalgorithm.name=="SOTR")
        {   
            PyObject* height=PyLong_FromLong(subalgorithm.height);
            PyObject* width=PyLong_FromLong(subalgorithm.width);
            PyObject* x=PyLong_FromLong(subalgorithm.d);
            PyObject* y=PyLong_FromLong(subalgorithm.color);
            PyObject* tracker_name=PyUnicode_FromString(subalgorithm.tracker_name.c_str());
            PyObject* SOTR = PyObject_GetAttrString(pModuleSOTR, "track_webcam");
            if (!SOTR || !PyCallable_Check(SOTR)) {
                PyErr_Print();
                std::cerr << "Cannot find function 'process_image'" << std::endl;
                Py_DECREF(SOTR);
                return 1;
            }
            
            PyObject* pArgs = PyTuple_Pack(6,input,tracker_name,x,y,width,height); // 传递参数
            PyObject *newput;
            newput = PyObject_CallObject(SOTR, pArgs); // 调用函数
            Py_DECREF(width);
            Py_DECREF(height);
            Py_DECREF(x);
            Py_DECREF(y);
            Py_DECREF(tracker_name);
            Py_DECREF(pArgs);
            Py_DECREF(input);
            input=newput;
            Py_DECREF(SOTR);
            if (!input) {
              PyErr_Print();
              std::cerr << "Call failed" << std::endl;
              return 1;
          }
        }
        // if(subalgorithm.name=="suidonggenzong")
        // {   
            
        //     PyObject* box_x=PyLong_FromLong(subalgorithm.height);
        //     PyObject* box_y=PyLong_FromLong(subalgorithm.width);
        //     PyObject* SOTR = PyObject_GetAttrString(pModuleSOTR, "track_webcam");
        //     if (!SOTR || !PyCallable_Check(SOTR)) {
        //         PyErr_Print();
        //         std::cerr << "Cannot find function 'process_image'" << std::endl;
        //         Py_DECREF(SOTR);
        //         return 1;
        //     }
            
        //     PyObject* pArgs = PyTuple_Pack(6,input,tracker_name,x,y,width,height); // 传递参数
        //     PyObject *newput;
        //     newput = PyObject_CallObject(SOTR, pArgs); // 调用函数
        //     Py_DECREF(width);
        //     Py_DECREF(height);
        //     Py_DECREF(x);
        //     Py_DECREF(y);
        //     Py_DECREF(tracker_name);
        //     Py_DECREF(pArgs);
        //     Py_DECREF(input);
        //     input=newput;
        //     Py_DECREF(SOTR);
        //     if (!input) {
        //       PyErr_Print();
        //       std::cerr << "Call failed" << std::endl;
        //       return 1;
        //   }
        // }
      }
      mtx.unlock();
      // cv::Mat mat = pyopencv_to_cvmat(input);
      // if(mat.channels()==1){
      //   CvGrayMatToAVFrame(mat,frame);
      // }else{
      // CvMatToAVFrame(mat,frame);
      // }
      PyObjectToAVFrame(input,frame,1280,720);
      // Py_DECREF(input); input和mat用的是同一个内存地址，释放掉后，mat也用不了了
      // CvMatToAVFrame(mat,frame);
      
      // std::thread worker_thread(thread_func, enc_context, frame, pkt, abs_ctx, ofmt_ctx, in_stream, out_stream,  input,pts);
      //crypto.cc 第75行，资源竞争加锁，解决的段错误
      encode(enc_context,frame,pkt);
      // worker_thread.join();
      // auto end = std::chrono::high_resolution_clock::now();//计时结束
      // std::chrono::duration<double> elapsed = end - start;
      // std::cout << "The run time is: " << elapsed.count() << "s" << std::endl;
      // decode(dec_context, frame, pkt);
      pkt->pos = -1;
      pkt->pts = pts;
      pkt->dts = pts;
      // pkt->stream_index = 0;
      // av_bsf_send_packet(abs_ctx, pkt);
      // av_bsf_receive_packet(abs_ctx, pkt);
      /* copy packet */
      av_packet_rescale_ts(pkt, in_stream->time_base, out_stream->time_base);
      
       // int dataSize = 0;
    // for (int i = 0; i < AV_NUM_DATA_POINTERS; ++i) {
    //     if (frame->data[i] && frame->linesize[i]) {
    //         dataSize += frame->linesize[i] * frame->height;
    //     }
    // }
    // frame->width=1920;
    // frame->height=1080;
    // printf("%i \n",dataSize);

    // memcpy(shared_pkt,&pkt->size,4);
    //std::cout <<pkt->size<<std::endl;
    // memcpy(shared_pkt+4, pkt->data, pkt->size);
    // pkt->data=shared_pkt;
    auto end = std::chrono::high_resolution_clock::now();//计时结束
    std::chrono::duration<double> elapsed = end - start;
    // std::cout << "The run time is: " << elapsed.count() << "s" << std::endl;
    if(elapsed.count()>0.04){
        frame_number++;
        if(frame_number>=300){
            frame_number = 0;
            algorithm->clear();
            printf("清空算法\n");
        }
    }
    // if(pkt->flags & AV_PKT_FLAG_KEY){
    ret = av_interleaved_write_frame(ofmt_ctx, pkt);
    av_frame_free(&newframe);
    av_frame_free(&frame);
    // Py_DECREF(pModule);
        if(prevlabel){
        Py_DECREF(previmg);
    }else{
    prevlabel=true;
    }
      Py_DECREF(input);
    }

      // Py_DECREF(gray);
    // if (ret < 0) {
    //   fprintf(stderr, "Error muxing packet\n");
    //   break;
    // }
    // std::this_thread::sleep_for(std::chrono::milliseconds(10));
    // auto start = std::chrono::high_resolution_clock::now();//计时结束   
    //  avformat_close_input(&ifmt_ctx);
    //  avformat_open_input(&ifmt_ctx, in_filename, 0, &options);
    // auto end = std::chrono::high_resolution_clock::now();//计时结束
    // std::chrono::duration<double> elapsed = end - start;
    // std::cout << "The run time is: " << elapsed.count() << "s" << std::endl;
    
    // }
    /* pkt is now blank (av_interleaved_write_frame() takes ownership of
     * its contents and resets pkt), so that no unreferencing is necessary.
     * This would be different if one used av_write_frame(). */
    
  

  av_write_trailer(ofmt_ctx);
end:
  Py_Finalize();
  avcodec_free_context(&dec_context);
  avcodec_free_context(&codec_ctx1);
  av_dict_free(&options);
  av_packet_free(&pkt);
  avformat_close_input(&ifmt_ctx);
  /* close output */
  if (ofmt_ctx && !(ofmt->flags & AVFMT_NOFILE)) avio_closep(&ofmt_ctx->pb);
  avformat_free_context(ofmt_ctx);

  av_bsf_free(&abs_ctx);
  av_dict_free(&options);
  if (ret < 0 && ret != AVERROR_EOF) {
    fprintf(stderr, "Error occurred: %d\n", ret);
    return 1;
  }

  return 0;
  
}
int main(int argc, char* argv[]) {
  // ffmpeg -re -f lavfi -i testsrc2=size=640*480:rate=25 -vcodec libx264 -profile:v baseline
  // -keyint_min 60 -g 60 -sc_threshold 0 -f rtp rtp://127.0.0.1:56000
  // ffmpeg -re -stream_loop -1 -i  test.mp4 -vcodec copy -bsf:v h264_mp4toannexb -ssrc 12345678 -f
  // rtp rtp://127.0.0.1:56000
    // cudaSetDevice(0);
    // const std::string engine_file_path="person_5l_640_v6.0_20221030.trtmodel";
    // YOLOv8* yolov81 = new YOLOv8(engine_file_path);
    // yolo_map["people1"]=yolov81;
    // yolov81->make_pipe(true);
    // // cudaSetDevice(1);
    // YOLOv8* yolov82 = new YOLOv8(engine_file_path);
    // yolo_map["people2"]=yolov82;
    // yolov82->make_pipe(true);
    // delete yolov82;
    // YOLOv8* yolov83 = new YOLOv8(engine_file_path);
    // yolo_map["people3"]=yolov83;
    // yolov83->make_pipe(true);
    // delete yolov83;
    // YOLOv8* yolov84 = new YOLOv8(engine_file_path);
    // yolo_map["people3"]=yolov84;
    // yolov84->make_pipe(true);
    // delete yolov84;
    // cv::Mat  res, image;
    // cv::Size size        = cv::Size{640, 640};
    // int      num_labels  = 40;
    // int      topk        = 100;
    // float    score_thres = 0.7f;
    // float    iou_thres   = 0.7f;//超参数
    // std::vector<Object> objs;
    // objs.clear();
    // yolov8->copy_from_Mat(image, size);
    // yolov8->infer();
    // yolov8->postprocess(objs, score_thres, iou_thres, topk, num_labels);
    // yolov8->draw_objects(image, res, objs, CLASS_NAMES, COLORS, "11");
    // delete yolov8;
  std::string ip("127.0.0.1");
  char *rtsp="";
  uint16_t port = 10000;
  uint16_t httpport = 8000;
  if (argc == 5) {
    ip = argv[1];
    port = atoi(argv[2]);
    httpport = atoi(argv[3]);
    rtsp = argv[4];
    // port2 = atoi(argv[3]);
    // port3 = atoi(argv[4]);
    // port4 = atoi(argv[5]);
    // port5 = atoi(argv[6]);
  }else{
    return 0;
  }
  Utils::Crypto::ClassInit();
  RTC::DtlsTransport::ClassInit();
  RTC::DepLibSRTP::ClassInit();
  RTC::SrtpSession::ClassInit();
  EventLoop loop;
  WebRTCSessionFactory* webrtc_session_factory = new WebRTCSessionFactory();

  // pid_t pid1 = fork();
  // if (pid1 < 0) {
  //           std::cerr << "Fork failed!" << std::endl;
  //           return 1;
  //       } else if (pid1 == 0) {
  //           H2642Rtp("rtsp://admin:df15001500@172.20.63.193:554/Streaming/Channels/101", webrtc_session_factory1, &algorithm1);
  //           exit(0); // 确保子进程在完成后正常退出
  //       } 
  // pid_t pid2 = fork();
  // if (pid2 < 0) {
  //           std::cerr << "Fork failed!" << std::endl;
  //           return 1;
  //       } else if (pid2 == 0) {
  //           // 子进程
  //           H2642Rtp("rtsp://admin:admin12345@172.20.63.195:554", &webrtc_session_factory2, &algorithm2);
  //           exit(0); // 确保子进程在完成后正常退出
  //       } 
  std::thread flv_2_rtp_thread(
      [webrtc_session_factory,&algorithm,rtsp,&weather_number]() { H2642Rtp(rtsp, webrtc_session_factory,&algorithm,&weather_number); }
      );
  // std::thread flv_2_rtp_thread2(
  //     [&webrtc_session_factory2,&algorithm2]() { H2642Rtp("rtsp://admin:admin123456@172.20.63.129:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif", &webrtc_session_factory2,&algorithm2); }
  //     );
  // std::thread flv_2_rtp_thread3(
  //     [&webrtc_session_factory3,&algorithm3]() { H2642Rtp("rtsp://admin:admin12345@172.20.63.195:554", &webrtc_session_factory3,&algorithm3); }
  //     );
  // std::thread flv_2_rtp_thread4(
  //     [&webrtc_session_factory4,&algorithm4]() { H2642Rtp("rtsp://admin:admin123456@172.20.63.129:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif", &webrtc_session_factory4,&algorithm4); }
  //     );
  // std::thread flv_2_rtp_thread5(
  //     [&webrtc_session_factory5,&algorithm5]() { H2642Rtp("rtsp://admin:admin123456@172.20.63.129:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif", &webrtc_session_factory5,&algorithm5); }
  //     );
  // std::thread flv_2_rtp_thread6(
  //     [&webrtc_session_factory]() { H2642Rtp("rtsp://admin:df15001500@172.20.63.193:554/Streaming/Channels/101", &webrtc_session_factory); }
  //     );
  // std::thread flv_2_rtp_thread7(
  //     [&webrtc_session_factory]() { H2642Rtp("rtsp://admin:admin123456@172.20.63.129:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif", &webrtc_session_factory); }
  //     );
  // std::thread flv_2_rtp_thread8(
  //     [&webrtc_session_factory]() { H2642Rtp("rtsp://admin:admin123456@172.20.63.129:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif", &webrtc_session_factory); }
  //     );
  //rtsp://admin:df15001500@172.20.63.193:554/Streaming/Channels/101
  //rtsp://admin:admin123456@172.20.63.129:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif
  UdpServer rtc_server(&loop, muduo::net::InetAddress("0.0.0.0", port), "rtc_server", 2);
  HttpServer http_server(&loop, muduo::net::InetAddress("0.0.0.0", httpport), "http_server",
                         TcpServer::kReusePort);

  rtc_server.SetPacketCallback([webrtc_session_factory](UdpServer* server, const uint8_t* buf,
                                                         size_t len,
                                                         const muduo::net::InetAddress& peer_addr,
                                                         muduo::Timestamp timestamp) {
    WebRTCSessionFactory::HandlePacket(webrtc_session_factory, server, buf, len, peer_addr,
                                       timestamp);
  });
  http_server.setHttpCallback(
      [&loop, webrtc_session_factory, port, ip](const HttpRequest& req, HttpResponse* resp) {  
        try
        {
        
             
        std::cout<<req.getbody()<<std::endl;
        if (req.path() == "/webrtc") {
          resp->setStatusCode(HttpResponse::k200Ok);
          resp->setStatusMessage("OK");
          resp->setContentType("text/plain");
          resp->addHeader("Access-Control-Allow-Origin", "*");
          auto rtc_session = webrtc_session_factory->CreateWebRTCSession(ip, port);
          resp->setBody(rtc_session->webrtc_transport()->GetLocalSdp());
          std::cout << rtc_session->webrtc_transport()->GetLocalSdp() << std::endl;
        }
        if (req.path().find("/image/") == 0) {
        // 先解码路径
        std::string decodedPath = urlDecode(req.path());
        std::string filePath = "/home/easy_webrtc_server/build/" + decodedPath.substr(7);
        // std::string filePath = "/home/easy_webrtc_server/build/" + req.path().substr(7);
        printf(filePath.c_str());
        std::ifstream file(filePath, std::ios::binary);

        if (file) {
            std::ostringstream responseStream;
            responseStream << file.rdbuf();
            std::string body = responseStream.str();
            resp->setStatusCode(muduo::net::HttpResponse::k200Ok);
            resp->setContentType("image/jpeg");
            resp->setBody(body);
        } else {
            resp->setStatusCode(muduo::net::HttpResponse::k404NotFound);
            resp->setStatusMessage("Not Found");
            resp->setCloseConnection(true);
        }
    } 
       if (req.path().find("/video/") == 0) {
    std::string filePath = "/home/easy_webrtc_server/build/" + req.path().substr(7);
    std::ifstream file(filePath, std::ios::binary);

    if (file) {
        resp->setStatusCode(muduo::net::HttpResponse::k200Ok);
        resp->setContentType("video/mp4");
        resp->addHeader("Cache-Control", "public, max-age=31536000");
        resp->addHeader("X-Content-Type-Options", "nosniff");

        std::string body;
        char buffer[4096]; // 每次读取 4KB 数据
        while (file.read(buffer, sizeof(buffer)) || file.gcount() > 0) {
            body.append(buffer, file.gcount()); // 将缓冲区内容附加到 body 中
        }
        resp->setBody(body); // 设置响应的 body
    } else {
        resp->setStatusCode(muduo::net::HttpResponse::k404NotFound);
        resp->setStatusMessage("Not Found");
        resp->setCloseConnection(true);
    }
}

        if (req.path() == "/algorithm") {
                // 检查POST请求中是否包含参数数据
                    algorithm.clear();
                    weather_number = 0;
                    const char * jsonData=req.getbody().data();
                    // 在这里处理参数数据，根据实际情况进行解析和处理
                    Json::CharReaderBuilder builder;
                    Json::CharReader* reader = builder.newCharReader();
                    Json::Value root;
                    std::string errs;
                    bool parsingSuccessful = reader->parse(jsonData, jsonData + strlen(jsonData), &root, &errs);
                    delete reader;
                    if (!parsingSuccessful) {
                        std::cerr << "Error parsing JSON: " << errs << std::endl;
                        return 1;
                    }
                    if (root.isMember("algorithm") && root["algorithm"].isArray()){
                        const Json::Value algorithms = root["algorithm"];
                        for (const auto& algo : algorithms) {
                          if (algo.isMember("subalgorithm")&& algo.isMember("tracker_name")&&algo.isMember("learning_rate"))
                            {
                                mtx.lock();
                                //猜测是因为SOT函数最后一次的调用还未完毕
                                PyObject* pheight=PyLong_FromLong(algo["height"].asInt());
                                PyObject* pwidth=PyLong_FromLong(algo["width"].asInt());
                                PyObject* px=PyLong_FromLong(algo["x"].asInt());
                                PyObject* py=PyLong_FromLong(algo["y"].asInt());
                                PyObject* pArgs = PyTuple_Pack(4,px,py,pwidth,pheight);
                                PyObject* pModuleSOT = PyImport_ImportModule("HJJX_TEST");
                                PyObject* stopSOT = PyObject_GetAttrString(pModuleSOT, "stopSOT");
                                if (!stopSOT || !PyCallable_Check(stopSOT)) {
                                    PyErr_Print();
                                    std::cerr << "Cannot find function 'process_image'" << std::endl;
                                    Py_DECREF(stopSOT);
                                    return 1;
                                }
                                PyObject *newput;
                                newput = PyObject_CallObject(stopSOT, pArgs); // 调用函数
                                mtx.unlock();
                                Py_DECREF(pwidth);
                                Py_DECREF(pheight);
                                Py_DECREF(pArgs);
                                Py_DECREF(py);
                                Py_DECREF(pArgs);
                                Py_DECREF(pModuleSOT);
                                Py_DECREF(stopSOT);
                                Py_DECREF(newput);
                                std::string subalgorithm = algo["subalgorithm"].asString();
                                std::string tracker_name = algo["tracker_name"].asString();
                                float learning_rate = algo["learning_rate"].asFloat();
                                int x=algo["x"].asInt();
                                int y=algo["y"].asInt();
                                int width=algo["width"].asInt();
                                int height=algo["height"].asInt();
                                algorithm.emplace_back(subalgorithm,height,width,x,y,0,0,learning_rate,0,0,0,0,0,0,0,0,0,0,tracker_name);
                            }
                            else if (algo.isMember("subalgorithm")&& algo.isMember("tracker_name")) {
                                // std::this_thread::sleep_for(std::chrono::milliseconds(500));//多线程导致资源冲突
                                mtx.lock();
                                //猜测是因为SOT函数最后一次的调用还未完毕
                                PyObject* pheight=PyLong_FromLong(algo["height"].asInt());
                                PyObject* pwidth=PyLong_FromLong(algo["width"].asInt());
                                PyObject* px=PyLong_FromLong(algo["x"].asInt());
                                PyObject* py=PyLong_FromLong(algo["y"].asInt());
                                PyObject* pArgs = PyTuple_Pack(4,px,py,pwidth,pheight);
                                PyObject* pModuleSOT = PyImport_ImportModule("siamese_tracking.run_given_videoflow_ONNX_RT");
                                PyObject* stopSOT = PyObject_GetAttrString(pModuleSOT, "stopSOT");
                                printf("成功\n");
                                if (!stopSOT || !PyCallable_Check(stopSOT)) {
                                    PyErr_Print();
                                    std::cerr << "Cannot find function 'process_image'" << std::endl;
                                    Py_DECREF(stopSOT);
                                    return 1;
                                }
                                PyObject *newput;
                                newput = PyObject_CallObject(stopSOT, pArgs); // 调用函数
                                Py_DECREF(pwidth);
                                Py_DECREF(pheight);
                                Py_DECREF(pArgs);
                                Py_DECREF(py);
                                Py_DECREF(pArgs);
                                Py_DECREF(pModuleSOT);
                                Py_DECREF(stopSOT);
                                Py_DECREF(newput);
                                mtx.unlock();
                                std::string subalgorithm = algo["subalgorithm"].asString();
                                std::string tracker_name = algo["tracker_name"].asString();
                                // float learning_rate = algo["learning_rate"].asFloat();
                                int x=algo["x"].asInt();
                                int y=algo["y"].asInt();
                                int width=algo["width"].asInt();
                                int height=algo["height"].asInt();
                                algorithm.emplace_back(subalgorithm,height,width,x,y,0,0,0,0,0,0,0,0,0,0,0,0,0,tracker_name);
                            }
                            else if (algo.isMember("subalgorithm") && algo.isMember("thre_1") && algo.isMember("thre_2")&& algo.isMember("sharpenthes")) {
                                std::string subalgorithm = algo["subalgorithm"].asString();
                                int height = algo["thre_2"].asInt();
                                int width = algo["thre_1"].asInt();
                                float sharpenthes = algo["sharpenthes"].asFloat();
                                algorithm.emplace_back(subalgorithm,height,width,0,0,0,0,sharpenthes);
                            }else if (algo.isMember("subalgorithm") && algo.isMember("weather_number")) {
                                std::string subalgorithm = algo["subalgorithm"].asString();
                                // PyObject* weather_name=PyUnicode_FromString(algo["weather_name"].asString().c_str());
                                // PyObject* kind=PyLong_FromLong(algo["kind"].asInt());
                                // PyObject* pArgs = PyTuple_Pack(2,weather_name,kind);
                                // PyObject* weather = PyImport_ImportModule("task6");
                                // PyObject* weather_control = PyObject_GetAttrString(weather, "weather_control");
                                // if (!weather_control || !PyCallable_Check(weather_control)) {
                                //     PyErr_Print();
                                //     std::cerr << "Cannot find function 'process_image'" << std::endl;
                                //     Py_DECREF(weather_control);
                                //     return 1;
                                // }
                                // PyObject *newput;
                            
                                // newput=PyObject_CallObject(weather_control, pArgs); // 调用函数
                                // Py_DECREF(weather_name);
                                // Py_DECREF(kind);
                                // Py_DECREF(pArgs);
                                // Py_DECREF(weather);
                                // Py_DECREF(weather_control);
                                // Py_DECREF(newput);
                                // algorithm.emplace_back(subalgorithm);
                                weather_number = algo["weather_number"].asInt();
                                
                            }else if (algo.isMember("subalgorithm") && algo.isMember("weights") && algo.isMember("sizes")&& algo.isMember("sharpenthes")) {
                                std::string subalgorithm = algo["subalgorithm"].asString();
                                int weights = algo["weights"].asInt();
                                int sizes = algo["sizes"].asInt();
                                float sharpenthes = algo["sharpenthes"].asFloat();
                                algorithm.emplace_back(subalgorithm,sizes,weights,0,0,0,0,sharpenthes);
                            }else if (algo.isMember("subalgorithm") && algo.isMember("weights") && algo.isMember("sharpenthes")) {
                                std::string subalgorithm = algo["subalgorithm"].asString();
                                int width = algo["weights"].asInt();
                                float sharpenthes = algo["sharpenthes"].asFloat();
                                algorithm.emplace_back(subalgorithm,0,width,0,0,0,0,sharpenthes);
                            }
                            else if (algo.isMember("subalgorithm") && algo.isMember("height") && algo.isMember("width")) {
                                std::string subalgorithm = algo["subalgorithm"].asString();
                                int height = algo["height"].asInt();
                                int width = algo["width"].asInt();
                                algorithm.emplace_back(subalgorithm,height,width);
                            } else if(algo.isMember("subalgorithm") && algo.isMember("d") && algo.isMember("color") && algo.isMember("space")) {
                                std::string subalgorithm = algo["subalgorithm"].asString();
                                int d = algo["d"].asInt();
                                int color = algo["color"].asInt();
                                int space = algo["space"].asInt();
                                algorithm.emplace_back(subalgorithm,0,0,d,color,space);
                            } else if(algo.isMember("subalgorithm") && algo.isMember("d") && algo.isMember("width") && algo.isMember("height")) {
                                std::string subalgorithm = algo["subalgorithm"].asString();
                                int d = algo["d"].asInt();
                                int width = algo["width"].asInt();
                                int height = algo["height"].asInt();
                                algorithm.emplace_back(subalgorithm,height,width,d);
                            } else if(algo.isMember("subalgorithm") && algo.isMember("lights") && algo.isMember("contrasts")) {
                                std::string subalgorithm = algo["subalgorithm"].asString();
                                int lights = algo["lights"].asInt();
                                float contrasts = algo["contrasts"].asFloat();
                                algorithm.emplace_back(subalgorithm,0,0,0,0,0,lights,contrasts);
                            } 
                            else if (algo.isMember("subalgorithm")&& algo.isMember("r_1")&& algo.isMember("r_2")&& algo.isMember("g_1")&& algo.isMember("g_2")&& algo.isMember("b_1")&& algo.isMember("b_2"))
                            {
                                std::string subalgorithm = algo["subalgorithm"].asString();
                                int r_1 = algo["r_1"].asInt();
                                int r_2 = algo["r_2"].asInt();
                                int g_1 = algo["g_1"].asInt();
                                int g_2 = algo["g_2"].asInt();
                                int b_1 = algo["b_1"].asInt();
                                int b_2 = algo["b_2"].asInt();
                                algorithm.emplace_back(subalgorithm);
                            }else if (algo.isMember("subalgorithm")&& algo.isMember("limits")&& algo.isMember("grids"))
                            {
                                std::string subalgorithm = algo["subalgorithm"].asString();
                                float limits = algo["limits"].asFloat();
                                int grids = algo["grids"].asInt();
                                algorithm.emplace_back(subalgorithm,0,0,0,0,0,0,0,0,0,0,0,0,0,limits,grids);
                            }else if (algo.isMember("subalgorithm")&& algo.isMember("ratios")&& algo.isMember("radii"))
                            {
                                std::string subalgorithm = algo["subalgorithm"].asString();
                                int ratios = algo["ratios"].asInt();
                                int radii = algo["radii"].asInt();
                                algorithm.emplace_back(subalgorithm,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,ratios,radii);
                            }else if (algo.isMember("subalgorithm")&& algo.isMember("weights")&& algo.isMember("kers"))
                            {
                                std::string subalgorithm = algo["subalgorithm"].asString();
                                int weights = algo["weights"].asInt();
                                int kers = algo["kers"].asInt();
                                algorithm.emplace_back(subalgorithm,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,weights,kers);
                            }else if (algo.isMember("subalgorithm")&& algo.isMember("methods"))
                            {
                                std::string subalgorithm = algo["subalgorithm"].asString();
                                int methods = algo["methods"].asInt();
                                algorithm.emplace_back(subalgorithm,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,methods);
                            }else if (algo.isMember("subalgorithm")&& algo.isMember("code")&& algo.isMember("wavelet")&& algo.isMember("level"))
                            {
                                std::string subalgorithm = algo["subalgorithm"].asString();
                                int infrared_image = algo["code"].asInt();
                                std::string wavelet = algo["wavelet"].asString();
                                int level = algo["level"].asInt();
                                algorithm.emplace_back(subalgorithm,infrared_image,0,0,0,0,0,0,level,0,0,0,0,0,0,0,0,0,wavelet);
                            }else if (algo.isMember("subalgorithm")&& algo.isMember("code")&& algo.isMember("alpha")&& algo.isMember("levels"))
                            {
                                std::string subalgorithm = algo["subalgorithm"].asString();
                                int infrared_image = algo["code"].asInt();
                                float alpha = algo["alpha"].asFloat();
                                int levels = algo["levels"].asInt();
                                algorithm.emplace_back(subalgorithm,infrared_image,0,0,0,0,0,alpha,levels);
                            }else if (algo.isMember("subalgorithm")&& algo.isMember("code")&& algo.isMember("alpha"))
                            {
                                std::string subalgorithm = algo["subalgorithm"].asString();
                                int infrared_image = algo["code"].asInt();
                                float alpha = algo["alpha"].asFloat();
                                algorithm.emplace_back(subalgorithm,infrared_image,0,0,0,0,0,alpha);
                            }else if (algo.isMember("subalgorithm")&& algo.isMember("code")&& algo.isMember("method"))
                            {
                                std::string subalgorithm = algo["subalgorithm"].asString();
                                int infrared_image = algo["code"].asInt();
                                int method = algo["method"].asInt();
                                algorithm.emplace_back(subalgorithm,infrared_image,method,0,0,0,0);
                            }else if (algo.isMember("subalgorithm")&& algo.isMember("method"))
                            {   
                                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                                std::string subalgorithm = algo["subalgorithm"].asString();
                                std::string method = algo["method"].asString();
                                // PyObject* pModuleDetect = PyImport_ImportModule("sdk_python");
                                // PyObject* setframe_id = PyObject_GetAttrString(pModuleDetect, "setframe_id");
                                // if (!setframe_id || !PyCallable_Check(setframe_id)) {
                                //     PyErr_Print();
                                //     std::cerr << "Cannot find function 'process_image'" << std::endl;
                                //     Py_DECREF(setframe_id);
                                //     return 1;
                                // }
                                // PyObject_CallObject(setframe_id, nullptr); // 调用函数
                                // Py_DECREF(pModuleDetect);
                                // Py_DECREF(setframe_id);
                                algorithm.emplace_back(subalgorithm,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,method);
                            }else if (algo.isMember("subalgorithm")&& algo.isMember("kind"))
                            {   
                                mtx.lock();
                                PyObject* weather_name=PyUnicode_FromString(algo["weather_name"].asString().c_str());
                                PyObject* kind=PyLong_FromLong(algo["kind"].asInt());
                                PyObject* pArgs = PyTuple_Pack(2,weather_name,kind);
                                PyObject* weather = PyImport_ImportModule("task6");
                                PyObject* weather_control = PyObject_GetAttrString(weather, "weather_control");
                                if (!weather_control || !PyCallable_Check(weather_control)) {
                                    PyErr_Print();
                                    std::cerr << "Cannot find function 'process_image'" << std::endl;
                                    Py_DECREF(weather_control);
                                    return 1;
                                }
                                PyObject_CallObject(weather_control, pArgs); // 调用函数
                                mtx.unlock();
                                Py_DECREF(weather_name);
                                Py_DECREF(kind);
                                Py_DECREF(pArgs);
                                Py_DECREF(weather);
                                Py_DECREF(weather_control);
                                
                            }else if (algo.isMember("subalgorithm"))
                            {   
                                std::string subalgorithm = algo["subalgorithm"].asString();
                                if(subalgorithm == "saoshe")
                                {
                                    mtx.lock();
                                    PyObject* pModuleStitch = PyImport_ImportModule("sdk_python_stitch");

                                    PyObject* set_first = PyObject_GetAttrString(pModuleStitch, "set_first");
                                    if (!set_first || !PyCallable_Check(set_first)) {
                                        PyErr_Print();
                                        std::cerr << "Cannot find function 'process_image'" << std::endl;
                                        Py_DECREF(set_first);
                                        return 1;
                                    }
                                    PyObject_CallObject(set_first, nullptr); // 调用函数
                                    mtx.unlock();
                                    Py_DECREF(pModuleStitch);
                                    Py_DECREF(set_first);
                                }
                                algorithm.emplace_back(subalgorithm);
                            }
                            else if (algo.isMember("box_x")&& algo.isMember("box_y"))
                            {
                                PyObject* box_x=PyLong_FromLong(algo["box_x"].asInt());
                                PyObject* box_y=PyLong_FromLong(algo["box_y"].asInt());
                                PyObject* label=PyLong_FromLong(algo["label"].asBool());
                                PyObject* pArgs = PyTuple_Pack(3,box_x,box_y,label);
                                PyObject* sdk_python = PyImport_ImportModule("sdk_python");
                                PyObject* setxy = PyObject_GetAttrString(sdk_python, "setxy");
                                if (!setxy || !PyCallable_Check(setxy)) {
                                    PyErr_Print();
                                    std::cerr << "Cannot find function 'process_image'" << std::endl;
                                    Py_DECREF(setxy);
                                    return 1;
                                }
                                PyObject_CallObject(setxy, pArgs); // 调用函数
                                Py_DECREF(box_x);
                                Py_DECREF(box_y);
                                Py_DECREF(label);
                                Py_DECREF(pArgs);
                                Py_DECREF(sdk_python);
                                Py_DECREF(setxy);
                            }
                               
                        }
                        resp->setStatusCode(HttpResponse::k200Ok);
                        resp->setStatusMessage("OK");
                        resp->setContentType("text/plain");
                        resp->addHeader("Access-Control-Allow-Origin", "*");
                    }
                    else {
                    // 如果请求体为空，则返回错误响应
                    resp->setStatusCode(HttpResponse::k400BadRequest);
                    resp->setStatusMessage("Bad Request: No POST data");
                }
        // std::this_thread::sleep_for(std::chrono::milliseconds(5000));//多线程导致资源冲突
        }
        if (req.path() == "/picAnalyse")
        {
            ;
        }
        /* code */
        }
        catch(const std::exception& e)
        {
          std::cerr << e.what() << '\n';
        }
      });
  loop.runInLoop([&]() {
    rtc_server.Start();
    http_server.start();
  });
  loop.loop();
  printf("end");
}