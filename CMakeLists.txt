cmake_minimum_required(VERSION 3.0)
project(webrtc_server)
add_definitions(-DAPI_EXPORTS)
add_definitions(-DGPU)
set(CMAKE_CXX_STANDARD 17)
SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall") 
add_compile_options(-g)   
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g")
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

file(GLOB RTC_LIST ./rtc/*.cc ./common/*.cc ./net/*.cc ./session/*.cc ./rtp/*.cc)

add_library(rtc ${RTC_LIST})

set(LINK_LIBRARYS 
srtp2
ssl 
crypto
dl pthread )
#需添加新的muduo代码后重新编译，原始代码为/home/muduo/build/lib/libmuduo_http.a->muduo_http,默认取/usr/local/lib/libmuduo_base.a
set(RTC_LIBRARYS rtc /home/muduo/build/lib/libmuduo_http.a /home/muduo/build/lib/libmuduo_net.a /home/muduo/build/lib/libmuduo_base.a)
# swscale
add_library(swscale SHARED IMPORTED)
set_target_properties(swscale PROPERTIES IMPORTED_LOCATION /usr/lib/x86_64-linux-gnu/libswscale.so) 

add_executable(rtp_src_example ./example/rtp_src_example.cc)
add_executable(rtp_src_example_pinjie ./example/rtp_src_example-pinjie.cc) #rtp_src_example-picAnalyse.cc  ele_number_num.cc 
add_executable(picAnalyse ./example/rtp_src_yc-picAnalyse.cc) 
# add_executable(rtp_src_example ./example/zzztest.cc)
target_link_libraries(rtp_src_example  ${RTC_LIBRARYS} ${LINK_LIBRARYS} avformat avcodec avutil rt swscale)
target_link_libraries(rtp_src_example_pinjie  ${RTC_LIBRARYS} ${LINK_LIBRARYS} avformat avcodec avutil rt swscale)
target_link_libraries(picAnalyse  ${RTC_LIBRARYS} ${LINK_LIBRARYS} avformat avcodec avutil rt swscale)
add_executable(publish ./example/publish.cc)
target_link_libraries(publish ${RTC_LIBRARYS} ${LINK_LIBRARYS} avformat avcodec avutil)
#jpeg
# 查找 JPEG 包
find_package(JPEG REQUIRED)
# 包含 JPEG 头文件
include_directories(${JPEG_INCLUDE_DIRS})
# 链接 JPEG 库
target_link_libraries(rtp_src_example ${JPEG_LIBRARIES})
target_link_libraries(rtp_src_example_pinjie ${JPEG_LIBRARIES})
#curl 
include_directories(/usr/local/include)
link_directories(/usr/local/lib)
target_link_libraries(rtp_src_example /usr/local/lib/libcurl.so)
target_link_libraries(rtp_src_example_pinjie /usr/local/lib/libcurl.so)
target_link_libraries(picAnalyse /usr/local/lib/libcurl.so)
#find_package(CURL REQUIRED)
#include_directories(${CURL_INCLUDE_DIRS})
#target_link_libraries(rtp_src_example ${CURL_LIBRARIES})

#opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARIES})
# other settings
# target_link_libraries(rtp_src_example ${OpenCV_LIBRARIES})
target_link_libraries(rtp_src_example_pinjie ${OpenCV_LIBRARIES})
target_link_libraries(picAnalyse ${OpenCV_LIBRARIES})
include_directories(./include)
target_sources(rtp_src_example PRIVATE
    sources/json_reader.cpp
    sources/json_value.cpp
    sources/json_writer.cpp
    sources/shm_buf.cpp
    include/ftplib.cc
    python/algorithm.py
)

target_sources(rtp_src_example_pinjie PRIVATE
    sources/json_reader.cpp
    sources/json_value.cpp
    sources/json_writer.cpp
    sources/shm_buf.cpp
    include/ftplib.cc
    python/algorithm.py
)

target_sources(picAnalyse PRIVATE
    sources/json_reader.cpp
    sources/json_value.cpp
    sources/json_writer.cpp
    sources/shm_buf.cpp
    include/ftplib.cc
    python/algorithm.py
)
#import python3
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
include_directories(${Python3_INCLUDE_DIRS})
include_directories(/usr/local/lib/python3.8/dist-packages/numpy/core/include/)
# include_directories(/usr/local/lib/python3.8/dist-packages/pybind11/include/)
target_link_libraries(rtp_src_example ${Python3_LIBRARIES})
target_link_libraries(rtp_src_example_pinjie ${Python3_LIBRARIES})
target_link_libraries(picAnalyse ${Python3_LIBRARIES})
# muduo
include_directories(/home/muduo)
set(SOURCES /home/muduo/muduo/net/http/*.cc)
# MySQL Connector/C++
include_directories(/home/easy_webrtc_server/mysql-connector-c++-1.1.12-linux-ubuntu18.04-x86-64bit/include)
link_directories(/home/easy_webrtc_server/mysql-connector-c++-1.1.12-linux-ubuntu18.04-x86-64bit/lib)
target_link_libraries(rtp_src_example ${RTC_LIBRARYS} ${LINK_LIBRARYS} avformat avcodec avutil rt swscale /home/easy_webrtc_server/mysql-connector-c++-1.1.12-linux-ubuntu18.04-x86-64bit/lib/libmysqlcppconn.so)
target_link_libraries(rtp_src_example_pinjie ${RTC_LIBRARYS} ${LINK_LIBRARYS} avformat avcodec avutil rt swscale /home/easy_webrtc_server/mysql-connector-c++-1.1.12-linux-ubuntu18.04-x86-64bit/lib/libmysqlcppconn.so)
target_link_libraries(picAnalyse ${RTC_LIBRARYS} ${LINK_LIBRARYS} avformat avcodec avutil rt swscale /home/easy_webrtc_server/mysql-connector-c++-1.1.12-linux-ubuntu18.04-x86-64bit/lib/libmysqlcppconn.so)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

#cuda
# include_directories(/home/easy_webrtc_server/lean/protobuf3.11.4/include/)
# link_libraries(/home/easy_webrtc_server/lean/protobuf3.11.4/lib)

include_directories(/home/easy_webrtc_server/lean/cuda-11.2/include/)
include_directories(/home/easy_webrtc_server/lean/TensorRT-8.0.3.4.cuda11.3.cudnn8.2/include/)
include_directories(/home/easy_webrtc_server/lean/cudnn8.2.2.26/include/)
# include_directories(/home/easy_webrtc_server/lean/cudnn-linux-x86_64-8.9.2.26_cuda12-archive/include/)
# # 切记，protobuf的lib目录一定要比tensorRT目录前面，因为tensorRTlib下带有protobuf的so文件
# # 这可能带来错误
set(RTPATH /home/easy_webrtc_server/lean/TensorRT-8.0.3.4.cuda11.3.cudnn8.2/lib/)
link_libraries(/home/easy_webrtc_server/lean/TensorRT-8.0.3.4.cuda11.3.cudnn8.2/lib)
link_libraries(/home/easy_webrtc_server/lean/cuda-11.2/lib64)
link_libraries(/home/easy_webrtc_server/lean/cuda-11.2/targets/x86_64-linux/lib/stubs)
link_libraries(/home/easy_webrtc_server/lean/cuda-11.2/targets/x86_64-linux/lib)
link_libraries(/home/easy_webrtc_server/lean/cudnn8.2.2.26/lib)
# link_libraries(/home/easy_webrtc_server/lean/cudnn-linux-x86_64-8.9.2.26_cuda12-archive/lib)
target_link_libraries(rtp_src_example ${RTPATH}/libnvinfer.so ${RTPATH}/libnvinfer_plugin.so ${RTPATH}/libcudnn.so /home/easy_webrtc_server/lean/cuda-11.2/targets/x86_64-linux/lib/stubs/libcuda.so /home/easy_webrtc_server/lean/cuda-11.2/targets/x86_64-linux/lib/libcublas.so /home/easy_webrtc_server/lean/cuda-11.2/targets/x86_64-linux/lib/libcudart.so)
target_link_libraries(rtp_src_example_pinjie ${RTPATH}/libnvinfer.so ${RTPATH}/libnvinfer_plugin.so ${RTPATH}/libcudnn.so /home/easy_webrtc_server/lean/cuda-11.2/targets/x86_64-linux/lib/stubs/libcuda.so /home/easy_webrtc_server/lean/cuda-11.2/targets/x86_64-linux/lib/libcublas.so /home/easy_webrtc_server/lean/cuda-11.2/targets/x86_64-linux/lib/libcudart.so)
target_link_libraries(picAnalyse ${RTPATH}/libnvinfer.so ${RTPATH}/libnvinfer_plugin.so ${RTPATH}/libcudnn.so /home/easy_webrtc_server/lean/cuda-11.2/targets/x86_64-linux/lib/stubs/libcuda.so /home/easy_webrtc_server/lean/cuda-11.2/targets/x86_64-linux/lib/libcublas.so /home/easy_webrtc_server/lean/cuda-11.2/targets/x86_64-linux/lib/libcudart.so)
#由于此时用的cuda11.2（在bashrc中添加的），torch版本固定为1.9.1,但其与系统cudnn版本冲突，固下载torch时，需要一并下载cudnn