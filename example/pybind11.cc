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
}
#include <json.h>
#include <common/uavAlgorithm.h>
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
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
using namespace muduo;
using namespace muduo::net;
namespace py = pybind11;
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
struct Algorithm {
    std::string name;
    int height;
    int width;
    // 构造函数
    Algorithm(std::string name,int height, int width) : name(name),height(height), width(width) {}
};
// 创建一个 vector 来存储 Algorithm 对象
std::vector<Algorithm> algorithm1;
std::vector<Algorithm> algorithm2;
std::vector<Algorithm> algorithm3;
std::vector<Algorithm> algorithm4;
std::vector<Algorithm> algorithm5;
cv::Mat pyopencv_to_cvmat(py::array_t<unsigned char>& input) {
    auto buf = input.request();
    int rows = buf.shape[0];
    int cols = buf.shape[1];
    int channels = buf.shape[2];

    cv::Mat mat(rows, cols, CV_8UC3, (unsigned char*)buf.ptr);
    return mat; // Ensure data integrity by cloning
}

// py::array_t<unsigned char> cvmat_to_pyopencv(const cv::Mat& mat) {
//     std::vector<size_t> shape = { (size_t)mat.rows, (size_t)mat.cols, (size_t)mat.channels() };
//     std::vector<size_t> strides = { (size_t)mat.step[0], (size_t)mat.step[1], (size_t)mat.elemSize() };
    
//     return py::array_t<unsigned char>(shape, strides, mat.data);
// }
py::array_t<unsigned char> cvmat_to_pyopencv(const cv::Mat& mat) {
    // 获取图像的数据指针、形状和步幅
    void* ptr = mat.data;
    std::vector<size_t> shape = { (size_t)mat.rows, (size_t)mat.cols, (size_t)mat.channels() };

    // 使用 py::array_t 构造数组对象并返回
    return py::array_t<unsigned char>(
        shape,                             // 形状
        (unsigned char*)ptr                // 数据指针
    );
}
void CvMatToAVFrame(const cv::Mat& input_mat, AVFrame* out_avframe)
{
    int image_width = input_mat.cols;
    int image_height = input_mat.rows;
    int cvLinesizes[1];
    cvLinesizes[0] = input_mat.step1();

    SwsContext* openCVBGRToAVFrameSwsContext = sws_getContext(
        image_width,
        image_height,
        AVPixelFormat::AV_PIX_FMT_BGR24,
        image_width,
        image_height,
        AVPixelFormat::AV_PIX_FMT_YUV420P,
        SWS_FAST_BILINEAR,
        nullptr, nullptr, nullptr
    );

    sws_scale(openCVBGRToAVFrameSwsContext,
        &input_mat.data,
        cvLinesizes,
        0,
        image_height,
        out_avframe->data,
        out_avframe->linesize);

    if (openCVBGRToAVFrameSwsContext != nullptr)
    {
        sws_freeContext(openCVBGRToAVFrameSwsContext);
        openCVBGRToAVFrameSwsContext = nullptr;
    }
}
void CvMatGrayToAVFrame(const cv::Mat& input_mat, AVFrame* out_avframe)
{
    int image_width = input_mat.cols;
    int image_height = input_mat.rows;
    int cvLinesizes[1];
    cvLinesizes[0] = input_mat.step1();

    SwsContext* openCVGrayToAVFrameSwsContext = sws_getContext(
        image_width,
        image_height,
        AVPixelFormat::AV_PIX_FMT_GRAY8, // Input format is grayscale
        image_width,
        image_height,
        AVPixelFormat::AV_PIX_FMT_YUV420P,
        SWS_FAST_BILINEAR,
        nullptr, nullptr, nullptr
    );

    sws_scale(openCVGrayToAVFrameSwsContext,
        &input_mat.data,
        cvLinesizes,
        0,
        image_height,
        out_avframe->data,
        out_avframe->linesize);

    if (openCVGrayToAVFrameSwsContext != nullptr)
    {
        sws_freeContext(openCVGrayToAVFrameSwsContext);
        openCVGrayToAVFrameSwsContext = nullptr;
    }
}
cv::Mat AVFrameToCvMat(AVFrame* input_avframe)
{
    int image_width = input_avframe->width;
    int image_height = input_avframe->height;

    cv::Mat resMat(image_height, image_width, CV_8UC3);
    int cvLinesizes[1];
    cvLinesizes[0] = resMat.step1();

    SwsContext* avFrameToOpenCVBGRSwsContext = sws_getContext(
        image_width,
        image_height,
        AVPixelFormat::AV_PIX_FMT_YUV420P,
        image_width,
        image_height,
        AVPixelFormat::AV_PIX_FMT_BGR24,
        SWS_FAST_BILINEAR,
        nullptr, nullptr, nullptr
    );

    sws_scale(avFrameToOpenCVBGRSwsContext,
        input_avframe->data,
        input_avframe->linesize,
        0,
        image_height,
        &resMat.data,
        cvLinesizes);

    if (avFrameToOpenCVBGRSwsContext != nullptr)
    {
        sws_freeContext(avFrameToOpenCVBGRSwsContext);
        avFrameToOpenCVBGRSwsContext = nullptr;
    }

    return resMat;
}

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
static int H2642Rtp(const char* in_filename, void* opaque,std::vector<Algorithm> *algorithm) {
  // python
  py::scoped_interpreter guard{}; // start the interpreter and keep it alive
  // Add the directory containing the Python script to the Python path
  py::module sys = py::module::import("sys");
  sys.attr("path").attr("append")("../python/YoloStitch");
  // Initialize the Python script
  py::module python_algorithm = py::module::import("mystitch");
  // Call the Python function
  py::object process_image = python_algorithm.attr("stitch");
  const AVOutputFormat* ofmt = NULL;
  AVFrame *frame=NULL;

  frame = av_frame_alloc();
  frame->format = AV_PIX_FMT_YUV420P;
    frame->width = 1280;
    frame->height = 720;
  AVIOContext* avio_ctx = NULL;
  //用于打开、读取、写入音视频文件，并维护了音视频格式的全局信息
  AVFormatContext *ifmt_ctx = NULL, *ofmt_ctx = NULL;
  AVPacket* pkt = NULL;
  AVCodecContext* dec_context = nullptr;
  AVCodecContext* enc_context = nullptr;
  const AVCodec* codec = nullptr;
  avcodec_register_all;
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
  enc_context->bit_rate = 90000;
  // 设置编码器的像素格式
  enc_context->pix_fmt = AV_PIX_FMT_YUV420P;
  enc_context->thread_count=2;
  av_opt_set(enc_context->priv_data, "preset", "fast", 0); //转码速度最快，视频也最模糊
  av_opt_set(enc_context->priv_data, "tune", "zerolatency", 0); //编码器实时编码，必须放在open前，零延迟，用在需要非常低的延迟的情况下，比如电视电话会议的编码。
    if (av_frame_get_buffer(frame, 0) < 0) {
        exit(1);
    }
    if (av_frame_make_writable(frame) < 0) {
        exit(1);
    }
  avcodec_open2(enc_context,codec,NULL);
  
  //解码器
  codec = avcodec_find_decoder(AV_CODEC_ID_H264);
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
  // 创建共享内存
  // int fd = shm_open("/avpacket_shm", O_CREAT | O_RDWR, 0666);
  // if (fd == -1) {
  //     std::cerr << "Error opening shared memory" << std::endl;
  //     return -1;
  // }
  // ftruncate(fd, 100*1024);
  // 映射共享内存
  // uint8_t *shared_pkt = (uint8_t *)mmap(NULL, 100*1024, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  // if (shared_pkt == MAP_FAILED) {
  //     std::cerr << "Failed to mmap shared memory" << std::endl;
  //     return -1;
  // }
 
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
  av_dict_set(&options, "rtsp_transport", "tcp", 0);
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
  while (1) {
    //能保证读出来的是完整的一帧
    ret = av_read_frame(ifmt_ctx, pkt);
    if (ret < 0) {
      for (size_t i = 0; i < ifmt_ctx->nb_streams; i++) {
        av_seek_frame(ifmt_ctx, i, 0, AVSEEK_FLAG_BYTE);
      }
      printf("contiue");
      continue;
    }
    if (pkt->stream_index != in_stream_index) {
      //  printf("错误\n");
      // printf("%i",pkt->stream_index);
      // printf("%i\n",in_stream_index);
      continue;
    }
    in_stream = ifmt_ctx->streams[in_stream_index];
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
     
      decode(dec_context, frame, pkt);
      // if(frame->width!=0)
      // ScaleImg(frame->height,frame->width,frame,frame,90,160);
      // frame->width=160;
      // frame->height=90;
      if(frame->width==0){
      continue;
      }
      // printf("frame:%i\n",frame->width);
      cv::Mat mattem;// mattem是一个智能指针
      mattem=AVFrameToCvMat(frame);
      // Convert cv::Mat to numpy array
      auto start = std::chrono::high_resolution_clock::now();
      py::array_t<unsigned char> input = cvmat_to_pyopencv(mattem);
      // py::array_t<unsigned char> result = process_image(input).cast<py::array_t<unsigned char>>();
      // Convert numpy array back to cv::Mat
      cv::Mat mat = pyopencv_to_cvmat(result);
      auto end = std::chrono::high_resolution_clock::now();//计时结束
      std::chrono::duration<double> elapsed = end - start;
      std::cout << "The run time is: " << elapsed.count() << "s" << std::endl;
      // cv::Mat mat;
      // cv::resize(mattem ,mat, cv::Size(640,360));
      // 不能resize，否则需要修改参数后重新打开编码器
    //   for (const auto subalgorithm : *algorithm) {
    //     if(subalgorithm.name=="mean_filter")
    //     {
    //       mattem=mean_filter(mattem,subalgorithm.height,subalgorithm.width);
    //     }
    // }

      // mattem=cv::imread("1.jpg",1);
      // if(flags.test(0)){
      // cv::Mat matgray;
      // cv::cvtColor(mattem, matgray, cv::COLOR_BGR2GRAY);
      // CvMatGrayToAVFrame(matgray,frame);
      // }else{
      // cv::Mat mat=mean_filter(matgray,2,12);
      // // cv::Mat mat;
      // // cv::Mat newmat;
      // // 改变图像大小
      // // cv::resize(matgray ,newmat, cv::Size(384,288));
      // // mat=alphaFilter(newmat,1,2,12);
      // bool isSuccess = cv::imwrite("1.jpg",mattem);
      // if(isSuccess){
      // printf("保存成功");
      // // break;
      // }else{
      //   printf("保存失败");
      // }
      CvMatToAVFrame(mat,frame);
      printf("%i\n",8);
      // }
      encode(enc_context,frame,pkt);
      
      // decode(dec_context, frame, pkt);
      pkt->pts = pts;
      pkt->dts = pts;
      // pkt->stream_index = 0;
      av_bsf_send_packet(abs_ctx, pkt);
      av_bsf_receive_packet(abs_ctx, pkt);
      /* copy packet */
      av_packet_rescale_ts(pkt, in_stream->time_base, out_stream->time_base);
      pkt->pos = -1;
      // pkt->pts = pts;
      // pkt->dts = pts;
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
//     auto end = std::chrono::high_resolution_clock::now();//计时结束
//     std::chrono::duration<double> elapsed = end - start;
// std::cout << "The run time is: " << elapsed.count() << "s" << std::endl;
    // if(pkt->flags & AV_PKT_FLAG_KEY){
    ret = av_interleaved_write_frame(ofmt_ctx, pkt);
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
    
  }

  av_write_trailer(ofmt_ctx);
end:
  avcodec_free_context(&dec_context);
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

  std::string ip("127.0.0.1");
  uint16_t port1 = 10000,port2 = 10001,port3 = 10002,port4 = 10003,port5 = 10004;
  if (argc == 2) {
    ip = argv[1];
    // port1 = atoi(argv[2]);
    // port2 = atoi(argv[3]);
    // port3 = atoi(argv[4]);
    // port4 = atoi(argv[5]);
    // port5 = atoi(argv[6]);
  }
  Utils::Crypto::ClassInit();
  RTC::DtlsTransport::ClassInit();
  RTC::DepLibSRTP::ClassInit();
  RTC::SrtpSession::ClassInit();
  EventLoop loop;
  WebRTCSessionFactory webrtc_session_factory1,webrtc_session_factory2,webrtc_session_factory3,webrtc_session_factory4,webrtc_session_factory5;

  std::thread flv_2_rtp_thread1(
      [&webrtc_session_factory1,&algorithm1]() { H2642Rtp("rtsp://admin:df15001500@172.20.63.193:554/Streaming/Channels/101", &webrtc_session_factory1,&algorithm1); }
      );
  std::thread flv_2_rtp_thread2(
      [&webrtc_session_factory2,&algorithm2]() { H2642Rtp("rtsp://admin:admin123456@172.20.63.129:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif", &webrtc_session_factory2,&algorithm2); }
      );
  std::thread flv_2_rtp_thread3(
      [&webrtc_session_factory3,&algorithm3]() { H2642Rtp("rtsp://admin:admin12345@172.20.63.195:554", &webrtc_session_factory3,&algorithm3); }
      );
  std::thread flv_2_rtp_thread4(
      [&webrtc_session_factory4,&algorithm4]() { H2642Rtp("rtsp://admin:admin123456@172.20.63.129:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif", &webrtc_session_factory4,&algorithm4); }
      );
  std::thread flv_2_rtp_thread5(
      [&webrtc_session_factory5,&algorithm5]() { H2642Rtp("rtsp://admin:admin123456@172.20.63.129:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif", &webrtc_session_factory5,&algorithm5); }
      );
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
  UdpServer rtc_server1(&loop, muduo::net::InetAddress("0.0.0.0", port1), "rtc_server", 2);
  HttpServer http_server(&loop, muduo::net::InetAddress("0.0.0.0", 8000), "http_server",
                         TcpServer::kReusePort);

  rtc_server1.SetPacketCallback([&webrtc_session_factory1](UdpServer* server, const uint8_t* buf,
                                                         size_t len,
                                                         const muduo::net::InetAddress& peer_addr,
                                                         muduo::Timestamp timestamp) {
    WebRTCSessionFactory::HandlePacket(&webrtc_session_factory1, server, buf, len, peer_addr,
                                       timestamp);
  });
  UdpServer rtc_server2(&loop, muduo::net::InetAddress("0.0.0.0", port2), "rtc_server", 2);
  rtc_server2.SetPacketCallback([&webrtc_session_factory2](UdpServer* server, const uint8_t* buf,
                                                         size_t len,
                                                         const muduo::net::InetAddress& peer_addr,
                                                         muduo::Timestamp timestamp) {
    WebRTCSessionFactory::HandlePacket(&webrtc_session_factory2, server, buf, len, peer_addr,
                                       timestamp);
  });
  UdpServer rtc_server3(&loop, muduo::net::InetAddress("0.0.0.0", port3), "rtc_server", 2);
  rtc_server3.SetPacketCallback([&webrtc_session_factory3](UdpServer* server, const uint8_t* buf,
                                                         size_t len,
                                                         const muduo::net::InetAddress& peer_addr,
                                                         muduo::Timestamp timestamp) {
    WebRTCSessionFactory::HandlePacket(&webrtc_session_factory3, server, buf, len, peer_addr,
                                       timestamp);
  });
  UdpServer rtc_server4(&loop, muduo::net::InetAddress("0.0.0.0", port4), "rtc_server", 2);
  rtc_server4.SetPacketCallback([&webrtc_session_factory4](UdpServer* server, const uint8_t* buf,
                                                         size_t len,
                                                         const muduo::net::InetAddress& peer_addr,
                                                         muduo::Timestamp timestamp) {
    WebRTCSessionFactory::HandlePacket(&webrtc_session_factory4, server, buf, len, peer_addr,
                                       timestamp);
  });
  UdpServer rtc_server5(&loop, muduo::net::InetAddress("0.0.0.0", port5), "rtc_server", 2);
  rtc_server5.SetPacketCallback([&webrtc_session_factory5](UdpServer* server, const uint8_t* buf,
                                                         size_t len,
                                                         const muduo::net::InetAddress& peer_addr,
                                                         muduo::Timestamp timestamp) {
    WebRTCSessionFactory::HandlePacket(&webrtc_session_factory5, server, buf, len, peer_addr,
                                       timestamp);
  });
  http_server.setHttpCallback(
      [&loop, &webrtc_session_factory1,&webrtc_session_factory2,&webrtc_session_factory3,&webrtc_session_factory4,&webrtc_session_factory5, port1,port2,port3,port4,port5, ip](const HttpRequest& req, HttpResponse* resp) {
        if (req.path() == "/webrtc1") {
          resp->setStatusCode(HttpResponse::k200Ok);
          resp->setStatusMessage("OK");
          resp->setContentType("text/plain");
          resp->addHeader("Access-Control-Allow-Origin", "*");
          auto rtc_session = webrtc_session_factory1.CreateWebRTCSession(ip, port1);
          resp->setBody(rtc_session->webrtc_transport()->GetLocalSdp());
          std::cout << rtc_session->webrtc_transport()->GetLocalSdp() << std::endl;
        }
        if (req.path() == "/webrtc2") {
          resp->setStatusCode(HttpResponse::k200Ok);
          resp->setStatusMessage("OK");
          resp->setContentType("text/plain");
          resp->addHeader("Access-Control-Allow-Origin", "*");
          auto rtc_session = webrtc_session_factory2.CreateWebRTCSession(ip, port2);
          resp->setBody(rtc_session->webrtc_transport()->GetLocalSdp());
          std::cout << rtc_session->webrtc_transport()->GetLocalSdp() << std::endl;
        }
        if (req.path() == "/webrtc3") {
          resp->setStatusCode(HttpResponse::k200Ok);
          resp->setStatusMessage("OK");
          resp->setContentType("text/plain");
          resp->addHeader("Access-Control-Allow-Origin", "*");
          auto rtc_session = webrtc_session_factory3.CreateWebRTCSession(ip, port3);
          resp->setBody(rtc_session->webrtc_transport()->GetLocalSdp());
          std::cout << rtc_session->webrtc_transport()->GetLocalSdp() << std::endl;
        }
        if (req.path() == "/webrtc4") {
          resp->setStatusCode(HttpResponse::k200Ok);
          resp->setStatusMessage("OK");
          resp->setContentType("text/plain");
          resp->addHeader("Access-Control-Allow-Origin", "*");
          auto rtc_session = webrtc_session_factory4.CreateWebRTCSession(ip, port4);
          resp->setBody(rtc_session->webrtc_transport()->GetLocalSdp());
          std::cout << rtc_session->webrtc_transport()->GetLocalSdp() << std::endl;
        }
        if (req.path() == "/webrtc5") {
          resp->setStatusCode(HttpResponse::k200Ok);
          resp->setStatusMessage("OK");
          resp->setContentType("text/plain");
          resp->addHeader("Access-Control-Allow-Origin", "*");
          auto rtc_session = webrtc_session_factory5.CreateWebRTCSession(ip, port5);
          resp->setBody(rtc_session->webrtc_transport()->GetLocalSdp());
          std::cout << rtc_session->webrtc_transport()->GetLocalSdp() << std::endl;
        }
        if (req.path() == "/algorithm1") {
                // 检查POST请求中是否包含参数数据
                    algorithm1.clear();
                    std::cout<<req.getbody()<<std::endl;
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
                        return;
                    }
                    if (root.isMember("algorithm") && root["algorithm"].isArray()){
                        const Json::Value algorithms = root["algorithm"];
                        for (const auto& algo : algorithms) {
                            if (algo.isMember("subalgorithm") && algo.isMember("height") && algo.isMember("width")) {
                                std::string subalgorithm = algo["subalgorithm"].asString();
                                int height = algo["height"].asInt();
                                int width = algo["width"].asInt();
                                algorithm1.emplace_back(subalgorithm,height,width);
                            } else {
                                std::cerr << "Error: Missing required fields in algorithm object." << std::endl;
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
        }
        if (req.path() == "/closemean_filter") {
          resp->setStatusCode(HttpResponse::k200Ok);
          resp->setStatusMessage("OK");
          resp->setContentType("text/plain");
          resp->addHeader("Access-Control-Allow-Origin", "*");
        }
      });
  loop.runInLoop([&]() {
    rtc_server1.Start();
    rtc_server2.Start();
    rtc_server3.Start();
    rtc_server4.Start();
    rtc_server5.Start();
    http_server.start();
  });
  loop.loop();
  printf("end");
}
