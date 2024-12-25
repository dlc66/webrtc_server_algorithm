# webrtc_server_algorithm
基于webrtc流媒体服务器，添加编解码模块，添加算法调用模板


# 使用说明
ubuntu20.04安装依赖库openssl1.1以上、srtp、ffmpeg、muduo，srtp需要--enable-openssl
详细安装过程参考Dockerfile
```
mkdir build  
cd build   
cmake ..  
make  
```  
## rtp_src_example
* 运行程序，第一个参数为IP地址，第二个参数为udp端口，第三个参数为http端口号，第四个参数为输入的rtsp流地址：
* ./rtp_src_example 172.20.63.56 10004 8004 rtsp://admin:admin123456@172.20.63.129:554/cam/realmonitor?channel=1\&subtype=0\&unicast=true\&proto=Onvif
* 打开webrtchtml/index.html 输入IP地址和http端口，播放视频

## 致谢
本库webrtc部分来源于easy_webrtc_server项目，作者修改了多线程bug，并添加了编解码与混合编程算法调用模板以及一些其他功能。
- 特别感谢 [easy_webrtc_server](https://github.com/Mihawk086/easy_webrtc_server.git) 对本项目的支持。
