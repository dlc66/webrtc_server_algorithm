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
#include <cstdlib>
//imread 和 imwrite在c++调用python慎用
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
#include <numpy/ndarrayobject.h>
#include <thread>
#include <functional>
//加入检测模型
#include "chrono"
#include <experimental/filesystem>
#include "yolov8.hpp"
#include <TcpSimular.h>
#include <condition_variable>
using namespace muduo;
using namespace muduo::net;
std::mutex yolov8_mutex; //模型锁
std::vector<double> thread_durations; // 用于保存每个线程的执行时间
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

//tcp模拟接收tcp.json中的报文
TcpSimulator::TcpSimulator(const std::string& json_file) : json_file_(json_file) {}

bool TcpSimulator::LoadTasks() {
    std::ifstream file(json_file_);
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << json_file_ << std::endl;
        return false;
    }

    Json::Value root;
    file >> root;

    for (const auto& item : root) {
        std::string requestHostIp = item["requestHostIp"].asString();
        std::string requestHostPort = item["requestHostPort"].asString();

        for (const auto& object : item["objectList"]) {
            std::string objectId = object["objectId"].asString();
            const auto& typeList = object["typeList"];
            const auto& imagePathList = object["imagePathList"];

            for (int i = 0; i < 50; ++i) { // 每个任务复制 25 次，用来测试性能
                for (const auto& type : typeList) {
                    for (const auto& imagePath : imagePathList) {
                        Task task;
                        task.image_id = objectId + "_" + std::to_string(i);
                        task.task_type = type.asString(); // 使用 type 作为任务类型
                        task.url = imagePath.asString();

                        task_queue_.push(task);
                    }
                }
            }
        }
    }

    return true;
}




bool TcpSimulator::GetNextTask(Task& task) {
    if (task_queue_.empty()) {
        return false;
    }
    task = task_queue_.front();
    task_queue_.pop();
    return true;
}

void TcpSimulator::SendResult(const std::string& result) {
    std::ofstream result_file("result.txt", std::ios::app);
    if (result_file.is_open()) {
        result_file << result << std::endl;
    } else {
        std::cerr << "无法保存处理结果到文件" << std::endl;
    }
}



const std::string engine_file_path = "person_5s_640_v6.0_20240801.trtmodel";
std::queue<Task> task_queue;
std::mutex queue_mutex;
std::condition_variable queue_cond_var;
bool stop = false;
int start_count = 0; //调用次数

void process_image(std::vector<std::unique_ptr<YOLOv8>>& yolov8_models, std::vector<std::mutex>& model_mutexes, std::atomic<int>& tasks_processed) {
    const int max_retry_count = 10; // 最大重试次数

    while (true) {
        Task task;

        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            queue_cond_var.wait(lock, [] { return !task_queue.empty() || stop; });

            if (stop && task_queue.empty()) {
                return;
            }

            task = task_queue.front();
            task_queue.pop();
        }

        cv::Mat res, image;
        image = cv::imread(task.url);
        if (image.empty()) {
            std::cerr << "无法读取图像: " << task.url << std::endl;
            continue;
        }

        cv::Size size = cv::Size{640, 640};
        int num_labels = 40;
        int topk = 100;
        float score_thres = 0.7f;
        float iou_thres = 0.7f;
        std::vector<Object> objs;
        objs.clear();

        auto start_time = std::chrono::high_resolution_clock::now();

        bool processed = false;
        int retry_count = 0;
        while (!processed && retry_count < max_retry_count) {
            for (size_t i = 0; i < yolov8_models.size(); ++i) {
                if (model_mutexes[i].try_lock()) {
                    yolov8_models[i]->copy_from_Mat(image, size);
                    yolov8_models[i]->infer();
                    yolov8_models[i]->postprocess(objs, score_thres, iou_thres, topk, num_labels);
                    yolov8_models[i]->draw_objects(image, res, objs, CLASS_NAMES, COLORS, "11");
                    model_mutexes[i].unlock();
                    processed = true;
                    break;
                }
            }

            if (!processed) {
                retry_count++;
                std::this_thread::sleep_for(std::chrono::milliseconds(100));  // 等待一段时间再尝试
            }
        }

        if (!processed) {
            std::cerr << "任务 " << task.image_id << " 超过最大重试次数，重新加入队列" << std::endl;
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                task_queue.push(task);
            }
            queue_cond_var.notify_one();
            continue;
        }
        

         // std::string output_path = "results/" + task.image_id + ".jpg";//保存结果
           //cv::imwrite(output_path, res);



        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end_time - start_time;
        thread_durations.push_back(duration.count()); // 保存时间

        // 增加已处理任务数
         tasks_processed++;
        
    }
}






int main(int argc, char* argv[]) {
    auto program_start_time = std::chrono::high_resolution_clock::now(); // 进程开始时间

    TcpSimulator tcp_simulator("request.json");

    if (!tcp_simulator.LoadTasks()) {
        return 1;
    }

    Task task;
    int total_tasks = 0; // 总任务数
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        while (tcp_simulator.GetNextTask(task)) { // 将任务加到任务队列中
            task_queue.push(task);
            total_tasks++;
        }
    }

    std::atomic<int> tasks_processed(0); // 已处理任务数

    auto model_start_time = std::chrono::high_resolution_clock::now(); // 模型加载开始时间

    cudaSetDevice(0);
    const int num_models = 3;  // 模型池，加载模型数量
    std::vector<std::unique_ptr<YOLOv8>> yolov8_models;
    std::vector<std::mutex> model_mutexes(num_models);
    for (int i = 0; i < num_models; ++i) {
        yolov8_models.emplace_back(new YOLOv8(engine_file_path));
        yolov8_models.back()->make_pipe(true);
    }

    auto model_end_time = std::chrono::high_resolution_clock::now(); // 模型加载结束时间

    const int num_threads = 6; // 线程数量
    std::vector<std::thread> threads;

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(process_image, std::ref(yolov8_models), std::ref(model_mutexes), std::ref(tasks_processed)); // 每个线程循环执行 process_image，从队列中取出任务，并发执行
    }

    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop = true;
    }

    queue_cond_var.notify_all();

    for (auto& thread : threads) {
        thread.join();
    }

    

    // 打印每个线程的执行时间
    for (size_t i = 0; i < thread_durations.size(); ++i) {
        std::cout << "线程 " << i + 1 << " 执行时间: " << thread_durations[i] << " 秒" << std::endl;
    }

    auto program_end_time = std::chrono::high_resolution_clock::now(); // 进程结束时间
    std::chrono::duration<double> total_duration = program_end_time - program_start_time;
    std::chrono::duration<double> model_duration = model_end_time - model_start_time;
    std::chrono::duration<double> tcp_duration = model_start_time - program_start_time;
    std::cout << "任务数量： " << total_tasks << std::endl;
    std::cout << "线程数: " << num_threads << " 模型数: " << num_models << std::endl;
    std::cout << "总执行时间: " << total_duration.count() << " 秒" << std::endl;
    std::cout << "模型加载时间: " << model_duration.count() << " 秒" << std::endl;
    std::cout << "分配任务时间: " << tcp_duration.count() << " 秒" << std::endl;
    std::cout << "剩余任务数量： " << task_queue.size() << std::endl;
    std::cout << "已处理任务数量： " << tasks_processed << std::endl;
    return 0;
}









    /*
    const std::string engine_file_path="person_5s_640_v6.0_20240801.trtmodel";
    YOLOv8* yolov8 = new YOLOv8(engine_file_path);
    yolov8->make_pipe(true);
    cv::Mat  res, image;
    image = cv::imread("zidane.jpg");
    cv::Size size        = cv::Size{640, 640};
    int      num_labels  = 40;
    int      topk        = 100;
    float    score_thres = 0.7f;
    float    iou_thres   = 0.7f;//超参数
    std::vector<Object> objs;
    objs.clear();
    yolov8->copy_from_Mat(image, size);
    yolov8->infer();
    yolov8->postprocess(objs, score_thres, iou_thres, topk, num_labels);
    yolov8->draw_objects(image, res, objs, CLASS_NAMES, COLORS, "11");
    cv::imwrite("result.jpg",res);
    delete yolov8;
    */
