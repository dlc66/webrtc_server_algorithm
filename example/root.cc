extern "C" {
#include <libavcodec/bsf.h>
#include <libavformat/avformat.h>
#include <libavutil/opt.h>
#include <libavutil/timestamp.h>
}

#include <atomic>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <map>
#include <mutex>
#include <thread>

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


using namespace muduo;
using namespace muduo::net;

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

static int H2642Rtp(const char* in_filename, void* opaque) {
  const AVOutputFormat* ofmt = NULL;
  AVIOContext* avio_ctx = NULL;
  //用于打开、读取、写入音视频文件，并维护了音视频格式的全局信息
  AVFormatContext *ifmt_ctx = NULL, *ofmt_ctx = NULL;
  AVPacket* pkt = NULL;
  int ret, i;
  int in_stream_index = 0, out_stream_index = 0;
  int stream_mapping_size = 0;
  int64_t pts = 0;
  uint8_t *buffer = NULL, *avio_ctx_buffer = NULL;
  size_t buffer_size, avio_ctx_buffer_size = 4096;
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
      out_stream->codecpar->codec_id = AV_CODEC_ID_H265;

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
  while (1) {
    AVStream *in_stream, *out_stream;
    ret = av_read_frame(ifmt_ctx, pkt);
    if (ret < 0) {
      for (size_t i = 0; i < ifmt_ctx->nb_streams; i++) {
        av_seek_frame(ifmt_ctx, i, 0, AVSEEK_FLAG_BYTE);
        printf("11");
      }
      printf("contiue");
      continue;
    }
    if (pkt->stream_index != in_stream_index) {
      printf("%i",pkt->stream_index);
      printf("%i\n",in_stream_index);
      continue;
    }

    in_stream = ifmt_ctx->streams[in_stream_index];
    out_stream = ofmt_ctx->streams[0];
    // log_packet(ifmt_ctx, pkt, "in");
    pts += 40;
    pkt->pts = pts;
    pkt->dts = pts;
    av_bsf_send_packet(abs_ctx, pkt);
    av_bsf_receive_packet(abs_ctx, pkt);
    /* copy packet */
    av_packet_rescale_ts(pkt, in_stream->time_base, out_stream->time_base);
    pkt->pos = -1;
    // log_packet(ofmt_ctx, pkt, "out");
    //std::this_thread::sleep_for(std::chrono::milliseconds(1));
    ret = av_interleaved_write_frame(ofmt_ctx, pkt);
    /* pkt is now blank (av_interleaved_write_frame() takes ownership of
     * its contents and resets pkt), so that no unreferencing is necessary.
     * This would be different if one used av_write_frame(). */
    if (ret < 0) {
      fprintf(stderr, "Error muxing packet\n");
      break;
    }
  }

  av_write_trailer(ofmt_ctx);
end:
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
  uint16_t port = 10004;
  if (argc == 3) {
    ip = argv[1];
    port = atoi(argv[2]);
  }

  Utils::Crypto::ClassInit();
  RTC::DtlsTransport::ClassInit();
  RTC::DepLibSRTP::ClassInit();
  RTC::SrtpSession::ClassInit();
  EventLoop loop;
  WebRTCSessionFactory webrtc_session_factory;

  std::thread flv_2_rtp_thread(
      [&webrtc_session_factory]() { H2642Rtp("rtsp://admin:admin123456@172.20.63.129:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif", &webrtc_session_factory); });
  //rtsp://admin:df15001500@172.20.63.193:554/Streaming/Channels/101
  //rtsp://admin:admin123456@172.20.63.129:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif
  UdpServer rtc_server(&loop, muduo::net::InetAddress("0.0.0.0", port), "rtc_server", 2);
  HttpServer http_server(&loop, muduo::net::InetAddress("0.0.0.0", 8004), "http_server",
                         TcpServer::kReusePort);

  rtc_server.SetPacketCallback([&webrtc_session_factory](UdpServer* server, const uint8_t* buf,
                                                         size_t len,
                                                         const muduo::net::InetAddress& peer_addr,
                                                         muduo::Timestamp timestamp) {
    WebRTCSessionFactory::HandlePacket(&webrtc_session_factory, server, buf, len, peer_addr,
                                       timestamp);
  });

  http_server.setHttpCallback(
      [&loop, &webrtc_session_factory, port, ip](const HttpRequest& req, HttpResponse* resp) {
        if (req.path() == "/webrtc") {
          resp->setStatusCode(HttpResponse::k200Ok);
          resp->setStatusMessage("OK");
          resp->setContentType("text/plain");
          resp->addHeader("Access-Control-Allow-Origin", "*");
          auto rtc_session = webrtc_session_factory.CreateWebRTCSession(ip, port);
          resp->setBody(rtc_session->webrtc_transport()->GetLocalSdp());
          std::cout << rtc_session->webrtc_transport()->GetLocalSdp() << std::endl;
        }
      });
  loop.runInLoop([&]() {
    rtc_server.Start();
    http_server.start();
  });
  loop.loop();
}
