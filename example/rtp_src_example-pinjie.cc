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
//共享内存
#include "shm_buf.h"
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

std::mutex mtx1;
std::mutex mtx2;
std::mutex mtx3;
std::mutex mtx4;
// 定义一个自定义结构体
#include <vector>
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
const char *shm_name = "shm_name_test";
  SharedMemoryBuffer shmbuf(shm_name, 845*480*3*100 + 12);
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
AVFrame *frame1= av_frame_alloc();
AVFrame *frame2=av_frame_alloc();
AVFrame *frame3=av_frame_alloc();
AVFrame *frame4=av_frame_alloc();
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
static int H2642Rtp(const char* in_filename, void* opaque,std::vector<Algorithm> *algorithm,int * weather_number,AVFrame * frame1,std::mutex * mtx1) {
    // cv::Mat mat1=cv::imread("1.jpg");
    Py_Initialize();
    // 添加 Python 脚本目录到 Python 路径
    PyRun_SimpleString("import sys");
    // const wchar_t*  python_path = L"../python";
    // PySys_SetPath(python_path);
    PyRun_SimpleString("sys.path.append('/home/easy_webrtc_server/python')");
    PyObject* pModule = PyImport_ImportModule("algorithm");
    // PyObject* pModule = nullptr;
    PyRun_SimpleString("sys.path.append('/home/easy_webrtc_server/python/HJJX/model')");
    PyObject* pModuleSOT = PyImport_ImportModule("HJJX_TEST");
    PyRun_SimpleString("sys.path.append('/home/opencvdemo/4_area_search')");
    // PyObject* pModuleDetect = PyImport_ImportModule("sdk_python");
    // PyObject* pModulemmtracker = PyImport_ImportModule("mmtest");
    // PyRun_SimpleString("sys.path.append('/home/opencvdemo/SiamDW/siamese_tracking')");
    PyRun_SimpleString("sys.path.append('/home/opencvdemo/SiamDW')");
    // PyObject* pModuleSOTR = PyImport_ImportModule("siamese_tracking.run_given_videoflow_ONNX_RT");
    PyObject* pModuleSOTR = nullptr;
    PyRun_SimpleString("sys.path.append('/home/opencvdemo/skyAR/model/task6')");
    // PyObject* pModuleweather = PyImport_ImportModule("task6");
    PyRun_SimpleString("sys.path.append('/home/opencvdemo/stitch')");
    PyObject* pModuleStitch = PyImport_ImportModule("sdk_python_stitch");
    // PyObject* pModuleStitch_copy = PyImport_ImportModule("sdk_python_stitch_copy");
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
                        break;
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
      for (const auto subalgorithm : *algorithm) {
        if(subalgorithm.name=="stitch")
        {   
            try{
                mtx1->lock();
                mtx2.lock();
                mtx3.lock();
                mtx4.lock();
                if(frame1->width!=0&& frame2->width!=0&&frame3->width!=0&&frame4->width!=0){
                PyObject *pyframe2 =avframe_to_pyobject(frame1,1280,720);
                PyObject *pyframe3 =avframe_to_pyobject(frame2,1280,720);
                PyObject *pyframe4 =avframe_to_pyobject(frame3,1280,720);
                PyObject *pyframe5 =avframe_to_pyobject(frame4,1280,720);
                // 根据 参数 动态决定使用的帧
                std::vector<PyObject*> frames;
                if (subalgorithm.tracker_name.find('1') != std::string::npos) {
                frames.push_back(input); // 仅在 framesToUse 包含 '1' 时添加 frame1
            }
            if (subalgorithm.tracker_name.find('2') != std::string::npos) {
                frames.push_back(pyframe2); // 添加 frame2
            }
            if (subalgorithm.tracker_name.find('3') != std::string::npos) {
                frames.push_back(pyframe3); // 添加 frame3
            }
            if (subalgorithm.tracker_name.find('4') != std::string::npos) {
                frames.push_back(pyframe4); // 添加 frame4
            }
            if (subalgorithm.tracker_name.find('5') != std::string::npos) {
                frames.push_back(pyframe5); // 添加 frame5
            }
            PyObject* stitch_function;
            switch (frames.size()) {
            case 2:
                stitch_function = PyObject_GetAttrString(pModuleStitch, "stitch_two");
                break;
            case 3:
                stitch_function = PyObject_GetAttrString(pModuleStitch, "stitch_three");
                break;
            case 4:
                stitch_function = PyObject_GetAttrString(pModuleStitch, "stitch_four");
                break;
            case 5:
                stitch_function = PyObject_GetAttrString(pModuleStitch, "stitch_five");
                break;
            }

            //     PyObject* stitch_two = PyObject_GetAttrString(pModuleStitch, "stitch_five");
            //     // save_frame_as_jpeg(frame1,"stitch.jpg");
            //     if (!stitch_two || !PyCallable_Check(stitch_two)) {
            //         PyErr_Print();
            //         std::cerr << "Cannot find function 'process_image'" << std::endl;
            //         Py_DECREF(stitch_two);
            //         return 1;
            //     }
                PyObject *newput;
                // PyObject* pArgs = PyTuple_Pack(4,pyframe2,pyframe3,pyframe4,pyframe5); // 传递参数
                //创建参数元组，大小为 frames.size()
                printf("size%i\n",frames.size());
                PyObject* pArgs = PyTuple_New(frames.size());
                // 设置每个帧到参数元组中
                for (size_t i = 0; i < frames.size(); i++) {
                    
                    PyTuple_SetItem(pArgs, i, frames[i]); // 设置每个帧
                    Py_INCREF(frames[i]);//当你用 PyTuple_SetItem 设置元素时，Python 不会自动增加这些元素的引用计数
                }
                newput = PyObject_CallObject(stitch_function, pArgs); // 调用函数
                Py_DECREF(pyframe2);
                Py_DECREF(pyframe3);
                Py_DECREF(pyframe4);
                Py_DECREF(pyframe5);
                Py_DECREF(pArgs);
                Py_DECREF(input);
                input=newput;
                if (!input) {
                PyErr_Print();
                std::cerr << "Call failed" << std::endl;
                return 1;
            }
                Py_DECREF(stitch_function);

                }else{
                    printf("保存空\n");
                }
                mtx1->unlock();
                mtx2.unlock();
                mtx3.unlock();
                mtx4.unlock();
            }catch(...){
                printf("异常\n");
            }
        }
      }
      
      PyObjectToAVFrame(input,frame,1280,720);

      encode(enc_context,frame,pkt);
      
      pkt->pos = -1;
      pkt->pts = pts;
      pkt->dts = pts;

      av_packet_rescale_ts(pkt, in_stream->time_base, out_stream->time_base);
      
    // std::this_thread::sleep_for(std::chrono::milliseconds(5));//多线程导致资源冲突
    auto end = std::chrono::high_resolution_clock::now();//计时结束
    std::chrono::duration<double> elapsed = end - start;
// std::cout << "The run time is: " << elapsed.count() << "s" << std::endl;
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
static int H2642RtpSub(const char* in_filename, void* opaque,std::vector<Algorithm> *algorithm,int * weather_number,AVFrame * frame_tem,std::mutex * mtx) {
  const AVOutputFormat* ofmt = NULL;
  AVIOContext* avio_ctx = NULL;
  //用于打开、读取、写入音视频文件，并维护了音视频格式的全局信息
  AVFormatContext *ifmt_ctx = NULL, *ofmt_ctx = NULL;
  AVPacket* pkt = NULL;
  AVCodecContext* dec_context = nullptr;
  AVCodecContext* enc_context = nullptr;
  PyObject* input;
  PyObject* originalinput;
  bool prevlabel = false;
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
      out_stream->codecpar->codec_id = AV_CODEC_ID_H264;
      
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
      mtx->lock();
      frame_tem->format = AV_PIX_FMT_YUV420P;
      frame_tem->width = 1280;
      frame_tem->height = 720;
      AVFrame *newframe=NULL;
      newframe = av_frame_alloc();
      newframe->format = AV_PIX_FMT_YUV420P;
      newframe->width = 1280;
      newframe->height = 720;
      decode(dec_context, frame_tem, pkt);
      av_packet_unref(pkt);
      if(frame_tem->width==0){

      printf("111111\n");
      mtx->unlock();
      continue;
      }
      if(*weather_number == 0)
            ;
        else if(*weather_number == 1){
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
                        av_frame_unref(frame_tem);
                        if (av_frame_ref(frame_tem, newframe) < 0) {
                            std::cerr << "Failed to reference newframe to frame!" << std::endl;
                            // 处理错误
                        }
                        break;
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
                        av_frame_unref(frame_tem);
                        if (av_frame_ref(frame_tem, newframe) < 0) {
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
      // if(!(pkt->flags & AV_PKT_FLAG_KEY)){
      //   continue;
      // }
      // if(frame->width!=0)
      // ScaleImg(frame->height,frame->width,frame,frame,90,160);
      // frame->width=160;
      // frame->height=90;
      if(frame_tem->width==0){
      printf("111111\n");
      mtx->unlock();
      continue;
      }
    mtx->unlock();
    // std::this_thread::sleep_for(std::chrono::milliseconds(40));//多线程导致资源冲突
    // mtx->lock();
    // // av_frame_free(&frame1);
    // // frame1 = nullptr;
    // // av_frame_unref(frame_tem);
    // mtx->unlock();
    av_frame_free(&newframe);
  }
    // Py_DECREF(pModule);
end:
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

  std::string ip("127.0.0.1");
  char *rtsp="";
  uint16_t port = 10000;
  uint16_t httpport = 8000;
  if (argc == 5) {
    ip = argv[1];
    port = atoi(argv[2]);
    httpport = atoi(argv[3]);
    rtsp = argv[4];
  }else{
    return 0;
  }
  Utils::Crypto::ClassInit();
  RTC::DtlsTransport::ClassInit();
  RTC::DepLibSRTP::ClassInit();
  RTC::SrtpSession::ClassInit();
  EventLoop loop;
  WebRTCSessionFactory* webrtc_session_factory = new WebRTCSessionFactory();
WebRTCSessionFactory* webrtc_session_factory2 = new WebRTCSessionFactory();
WebRTCSessionFactory* webrtc_session_factory3 = new WebRTCSessionFactory();
  std::thread flv_2_rtp_thread(
      [webrtc_session_factory,&algorithm,rtsp,&weather_number,frame1,&mtx1]() { H2642Rtp(rtsp, webrtc_session_factory,&algorithm,&weather_number,frame1,&mtx1); }
      );
  std::thread flv_2_rtp_thread2(
      [webrtc_session_factory2,&algorithm,rtsp,&weather_number,frame1,&mtx1]() { H2642RtpSub("rtsp://admin:admin12345@172.20.63.195:554", webrtc_session_factory2,&algorithm,&weather_number,frame1,&mtx1); }
      );
  std::thread flv_2_rtp_thread3(
      [&webrtc_session_factory3,&algorithm,rtsp,&weather_number,frame2,&mtx2]() { H2642RtpSub("rtsp://admin:admin12345@172.20.63.195:554", &webrtc_session_factory3,&algorithm,&weather_number,frame2,&mtx2); }
      );
std::thread flv_2_rtp_thread4(
      [&webrtc_session_factory3,&algorithm,rtsp,&weather_number,frame3,&mtx3]() { H2642RtpSub("rtsp://admin:admin12345@172.20.63.195:554", &webrtc_session_factory3,&algorithm,&weather_number,frame3,&mtx3); }
      );
      std::thread flv_2_rtp_thread5(
      [&webrtc_session_factory3,&algorithm,rtsp,&weather_number,frame4,&mtx4]() { H2642RtpSub("rtsp://admin:admin12345@172.20.63.195:554", &webrtc_session_factory3,&algorithm,&weather_number,frame4,&mtx4); }
      );
// std::thread flv_2_rtp_thread4(
//       [&webrtc_session_factory3,&algorithm,rtsp,&weather_number,frame3,&mtx3]() { H2642RtpSub(rtsp, &webrtc_session_factory3,&algorithm,&weather_number,frame3,&mtx3); }
//       );
// std::thread flv_2_rtp_thread5(
//       [&webrtc_session_factory3,&algorithm,rtsp,&weather_number,frame4,&mtx4]() { H2642RtpSub("rtsp://admin:admin12345@172.20.63.195:554", &webrtc_session_factory3,&algorithm,&weather_number,frame4,&mtx4); }
//       );
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
                                std::this_thread::sleep_for(std::chrono::milliseconds(100));//多线程导致资源冲突
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
                                std::this_thread::sleep_for(std::chrono::milliseconds(100));//多线程导致资源冲突
                                //猜测是因为SOT函数最后一次的调用还未完毕
                                // PyObject* pheight=PyLong_FromLong(algo["height"].asInt());
                                // PyObject* pwidth=PyLong_FromLong(algo["width"].asInt());
                                // PyObject* px=PyLong_FromLong(algo["x"].asInt());
                                // PyObject* py=PyLong_FromLong(algo["y"].asInt());
                                // PyObject* pArgs = PyTuple_Pack(4,px,py,pwidth,pheight);
                                // PyObject* pModuleSOT = PyImport_ImportModule("siamese_tracking.run_given_videoflow_ONNX_RT");
                                // PyObject* stopSOT = PyObject_GetAttrString(pModuleSOT, "stopSOT");
                                // printf("成功\n");
                                // if (!stopSOT || !PyCallable_Check(stopSOT)) {
                                //     PyErr_Print();
                                //     std::cerr << "Cannot find function 'process_image'" << std::endl;
                                //     Py_DECREF(stopSOT);
                                //     return 1;
                                // }
                                // PyObject *newput;
                                // newput = PyObject_CallObject(stopSOT, pArgs); // 调用函数
                                // Py_DECREF(pwidth);
                                // Py_DECREF(pheight);
                                // Py_DECREF(pArgs);
                                // Py_DECREF(py);
                                // Py_DECREF(pArgs);
                                // Py_DECREF(pModuleSOT);
                                // Py_DECREF(stopSOT);
                                // Py_DECREF(newput);
                                std::string subalgorithm = algo["subalgorithm"].asString();
                                std::string tracker_name = algo["tracker_name"].asString();
                                // float learning_rate = algo["learning_rate"].asFloat();
                                int x=algo["x"].asInt();
                                int y=algo["y"].asInt();
                                int width=algo["width"].asInt();
                                int height=algo["height"].asInt();
                                algorithm.emplace_back(subalgorithm,height,width,x,y,0,0,0,0,0,0,0,0,0,0,0,0,0,tracker_name);
                            }
                            else if (algo.isMember("subalgorithm")&& algo.isMember("stitch_number")) {
                                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                                std::string subalgorithm = algo["subalgorithm"].asString();
                                int stitch_number = algo["stitch_number"].asInt();
                                std::string stitch_code = algo["stitch_code"].asString();
                                // float learning_rate = algo["learning_rate"].asFloat();
                                algorithm.emplace_back(subalgorithm,stitch_number,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,stitch_code);
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
                                algorithm.emplace_back(subalgorithm);
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
                                std::string subalgorithm = algo["subalgorithm"].asString();
                                std::string method = algo["method"].asString();
                                algorithm.emplace_back(subalgorithm,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,method);
                            }else if (algo.isMember("subalgorithm")&& algo.isMember("kind"))
                            {   
                                std::this_thread::sleep_for(std::chrono::milliseconds(100));
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
                                Py_DECREF(weather_name);
                                Py_DECREF(kind);
                                Py_DECREF(pArgs);
                                Py_DECREF(weather);
                                Py_DECREF(weather_control);
                                
                            }else if (algo.isMember("subalgorithm"))
                            {
                                std::string subalgorithm = algo["subalgorithm"].asString();
                                algorithm.emplace_back(subalgorithm);
                            }
                            // else if (algo.isMember("box_x")&& algo.isMember("box_y"))
                            // {
                            //     PyObject* box_x=PyLong_FromLong(algo["box_x"].asInt());
                            //     PyObject* box_y=PyLong_FromLong(algo["box_y"].asInt());
                            //     PyObject* label=PyLong_FromLong(algo["label"].asBool());
                            //     PyObject* pArgs = PyTuple_Pack(3,box_x,box_y,label);
                            //     PyObject* sdk_python = PyImport_ImportModule("sdk_python");
                            //     PyObject* setxy = PyObject_GetAttrString(sdk_python, "setxy");
                            //     if (!setxy || !PyCallable_Check(setxy)) {
                            //         PyErr_Print();
                            //         std::cerr << "Cannot find function 'process_image'" << std::endl;
                            //         Py_DECREF(setxy);
                            //         return 1;
                            //     }
                            //     PyObject_CallObject(setxy, pArgs); // 调用函数
                            //     Py_DECREF(box_x);
                            //     Py_DECREF(box_y);
                            //     Py_DECREF(label);
                            //     Py_DECREF(pArgs);
                            //     Py_DECREF(pModuleSOT);
                            //     Py_DECREF(sdk_python);
                            //     Py_DECREF(setxy);
                            // }
                               
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