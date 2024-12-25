#include <iostream>
#include <string>
#include "shm_buf.h"
#include <unistd.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// Suppose the video is 3-channel 1080p 
static const uint32_t WIDTH = 845;
static const uint32_t HEIGHT = 480;
static const uint32_t CHANNELS = 3;

void consumer()
{
    const char *shm_name = "shm_name_test";
    SharedMemoryBuffer shmbuf(shm_name, WIDTH*HEIGHT*CHANNELS*100 + 12);
    while (true) {
        if (!shmbuf.readable()) {
            printf("111111111");
            usleep(100000);  // 100ms
            continue;
        }
        std::string data;
        uint32_t len = shmbuf.read_shm(data);
        printf("%i\n",len);
        if (len == WIDTH*HEIGHT*CHANNELS) {
            cv::Mat frame(HEIGHT, WIDTH, CV_8UC3, const_cast<char*>(data.data()));
            if (!frame.empty()) {
                cv::imwrite("1.jpg", frame);
                printf("成功");
            }
        }
    }
}

int main(int argc, char *argv[])
{
    consumer();
    return 0;
}
