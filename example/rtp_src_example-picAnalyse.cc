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
#include <malloc.h>
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
#include <filesystem>
#include <future>  // 用于 std::async 和 std::future
//加入检测模型
#include "chrono"
#include <experimental/filesystem>
#include "yolov8.hpp"
#include "algorithm/daozha.hpp"
#include "algorithm/number_class.hpp"
#include "algorithm/defect_25nc.hpp"
#include "algorithm/air_switch.hpp"
#include "algorithm/ele_number/ele_number_bp.hpp"
#include "algorithm/ele_number/ele_number_num.hpp"
//
#include <global.h>
#include <condition_variable>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <curl/curl.h>  
#include <ftplib.h>//试了，但没用上
using namespace muduo;
using namespace muduo::net;

struct ImageResult {
    std::string task_id; 
    std::string request_id; 
    cv::Mat image;      

    // 增强的构造函数
    ImageResult(const std::string& id,const std::string& req_id, const cv::Mat& img) {
        try {
            if (req_id.empty()||id.empty() || img.empty()) {
                throw std::runtime_error("Invalid parameters for ImageResult construction.");
            }
            task_id = id;
            request_id = req_id;
            image = img.clone();  // 确保独立的拷贝，避免引用外部释放的数据
        } catch (const std::exception& e) {
            std::cerr << "Failed to create ImageResult: " << e.what() << std::endl;
            // 可以在这里初始化为默认状态或记录错误，根据具体情况处理
        }
    }

    // 默认构造函数
    ImageResult() : task_id(""),request_id(""), image() {}
};


struct ImageLoad {//下载图像结构体
    std::string task_url; 
    cv::Mat image;      
    int ref_count = 0;  // 新增引用计数器,避免多次任务重复图片释放内存误删
    // 构造函数，初始化task_id和cv::Mat
    ImageLoad(const std::string& url, const cv::Mat& img)
        : task_url(url), image(img) {}

    // 默认构造函数
    ImageLoad() : task_url(""), image() {}
};


struct TaskInfo {
    std::string requestId;       // 请求ID
    std::string imageUrl;        // 图像URL，用于标识图像
    ResultMessage resultMessage; // 存储结果信息
    int type_count;              // 需要处理的任务类型数量
};
std::vector<TaskInfo> taskInfoBuffer;//结果报文暂存区
// 添加任务到 vector
void addTaskToBuffer(const std::string& requestId, const std::string& imageUrl, const ResultMessage& resultMessage,
 int type_count, const std::string& objectId) {
    TaskInfo taskInfo;
    taskInfo.requestId = requestId;
    taskInfo.imageUrl = imageUrl;
    taskInfo.resultMessage = resultMessage;
    taskInfo.type_count = type_count;
    taskInfo.resultMessage.requestid = requestId;
    taskInfo.resultMessage.resultList.objectId = objectId;
    taskInfoBuffer.push_back(taskInfo);
}

// 查找并更新任务
TaskInfo* findTaskByRequestIdAndImageUrl(const std::string& requestId, const std::string& imageUrl) {
    for (auto& task : taskInfoBuffer) {
        if (task.requestId == requestId && task.imageUrl == imageUrl) {
            return &task;
        }
    }
    return nullptr; // 没有找到则返回空指针
}

bool model_flag = true;
std::deque<Task> buffered_tasks; // 任务缓冲区
std::queue<Task> task_queue;//任务队列
std::queue<std::shared_ptr<ImageResult>> result_queue;
std::vector<ImageLoad> imageLoads;//图像加载动态数组
std::vector<ResultMessage> messageBuffer;
std::mutex queue_mutex;
std::mutex buffer_mutex;
std::mutex taskinfo_mutex;
std::mutex mat_mutex;
std::condition_variable queue_cond_var;
std::mutex yolov8_mutex; //模型锁
std::vector<double> thread_durations; // 用于保存每个线程的执行时间

// 全局变量来存储模型实例
std::vector<std::unique_ptr<YOLOv8>> yolov8_1_models;
std::vector<std::unique_ptr<YOLOv8>> yolov8_2_models;
std::vector<std::unique_ptr<YOLOv8>> yolov8_3_models;
std::vector<std::unique_ptr<YOLOv8>> yolov8_4_models;
std::vector<std::unique_ptr<YOLOv8>> yolov8_5_models;
std::vector<std::unique_ptr<YOLOv5>> defect_25nc_1_models;
std::vector<std::unique_ptr<YOLOv5>> defect_25nc_2_models;
std::vector<std::unique_ptr<YOLOv5>> defect_25nc_3_models;
std::vector<std::unique_ptr<YOLOv5>> defect_25nc_4_models;
std::vector<std::unique_ptr<YOLOv5>> defect_25nc_5_models;
std::vector<std::unique_ptr<EleNumber_bp>> Elenumber_models_bp;
std::vector<std::unique_ptr<EleNumber_num>> Elenumber_models_num;
std::vector<std::unique_ptr<DaoZhaClassifier>> daozha_1_models;
std::vector<std::unique_ptr<number_class>> number_class_models;
std::vector<std::unique_ptr<air_switch>> air_switch_models;
std::vector<std::unique_ptr<DaoZhaClassifier>> daozha_2_models;
std::vector<std::unique_ptr<DaoZhaClassifier>> daozha_3_models;
std::vector<std::unique_ptr<DaoZhaClassifier>> daozha_4_models;
std::vector<std::unique_ptr<DaoZhaClassifier>> daozha_5_models;

const int num_models = 1;  // 模型池，加载模型数量
const int num_max = 80; 
std::vector<std::mutex> model_mutexes(80);


// FTPS 连接全局变量
CURL *curl;//上传通道
CURL *curl_1;//下载通道
bool ftps_connected = false;

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


const std::string engine_file_path_yolov8_1 = 
    "models/person_yolov8/person_5s_640_v6.0_20240801_1.trtmodel";
const std::string engine_file_path_yolov8_2 = 
    "models/person_yolov8/person_5s_640_v6.0_20240801_2.trtmodel";
const std::string engine_file_path_yolov8_3 = 
     "models/person_yolov8/person_5s_640_v6.0_20240801_3.trtmodel";
const std::string engine_file_path_yolov8_4 = 
    "models/person_yolov8/person_5s_640_v6.0_20240801_4.trtmodel";
const std::string engine_file_path_yolov8_5 = 
    "models/person_yolov8/person_5s_640_v6.0_20240801_5.trtmodel";
const std::string engine_file_path_defect_25nc_1 = 
    "models/defect_25nc/defect_25labels_v5x_640_20240530_nc25.trtmodel";
const std::string engine_file_path_daozha_1 = 
    "models/daozha_class/daozha_resnet101_bs224_nc2_20240621_1.trtmodel";
const std::string engine_file_path_daozha_2 = 
    "models/daozha_class/daozha_resnet101_bs224_nc2_20240621_2.trtmodel";
const std::string engine_file_path_daozha_3 = 
    "models/daozha_class/daozha_resnet101_bs224_nc2_20240621_3.trtmodel";
const std::string engine_file_path_daozha_4 = 
    "models/daozha_class/daozha_resnet101_bs224_nc2_20240621_4.trtmodel";
const std::string engine_file_path_daozha_5 = 
    "models/daozha_class/daozha_resnet101_bs224_nc2_20240621_5.trtmodel";
const std::string engine_file_path_air_switch = 
    "models/air_switch/kqkg_5l_640_nc2_v6.0_20240914.trtmodel";
const std::string engine_file_path_number_class =
    "models/number_class/classifier_digital_resnet18_128_20230530.trtmodel";
const std::string engine_file_path_ele_num_class = 
    "models/ele_number/szyb_num_6.0l_nc21_20240910.trtmodel";
const std::string engine_file_path_ele_bp_class = 
    "models/ele_number/szyb_bp_6.0l_nc3_20240906.trtmodel";



void initialize_models(const std::map<std::string, int>& task_type_count) {
    auto model_start_time = std::chrono::high_resolution_clock::now(); // 模型加载开始时间

    int total_tasks = 0;
    for (const auto& entry : task_type_count) {
        total_tasks += entry.second; // 计算所有任务的总数量
    }

    std::map<std::string, int> model_load_count; // 用于记录每个模型类型成功加载的数量

    for (const auto& entry : task_type_count) {
        const std::string& type = entry.first;
        int task_count = entry.second;

        // 根据任务数量的比例来决定加载的模型数量
        int load_count = static_cast<int>((static_cast<double>(task_count) / total_tasks) * num_models);

        if (load_count == 0 && task_count > 0) {
            load_count = 1; // 确保至少为每种类型分配一个模型
        }

        for (int i = 0; i < load_count; ++i) {
            if (type == "person_1") 
            {
                yolov8_1_models.emplace_back(new YOLOv8(engine_file_path_yolov8_1));
                yolov8_1_models.back()->make_pipe(true);
            } 
            else if (type == "person_2")
            {
                yolov8_2_models.emplace_back(new YOLOv8(engine_file_path_yolov8_2));
                yolov8_2_models.back()->make_pipe(true);
            } else if (type == "person_3") 
            {
                yolov8_3_models.emplace_back(new YOLOv8(engine_file_path_yolov8_3));
                yolov8_3_models.back()->make_pipe(true);
            } else if (type == "defect_25nc") 
            {
                cv::Size inputSize(640, 640);
                defect_25nc_1_models.emplace_back(new YOLOv5(engine_file_path_defect_25nc_1,inputSize));
                defect_25nc_1_models.back()->initialize();
            } 
            else if(type == "ele_number")
            {   
                cv::Size inputSize(640, 640);
                Elenumber_models_bp.emplace_back(new EleNumber_bp(engine_file_path_ele_bp_class,inputSize));
                 Elenumber_models_bp.back()->initialize();
                Elenumber_models_num.emplace_back(new EleNumber_num(engine_file_path_ele_num_class,inputSize));
                 Elenumber_models_num.back()->initialize();
            }  
            else if (type == "daozha_1") 
            {
                daozha_1_models.emplace_back(std::make_unique<DaoZhaClassifier>(engine_file_path_daozha_1));
            } 
             else if (type == "daozha_2") 
            {
                daozha_1_models.emplace_back(std::make_unique<DaoZhaClassifier>(engine_file_path_daozha_2));
            } 
            else if (type == "number_class") 
            {
                number_class_models.emplace_back(std::make_unique<number_class>(engine_file_path_number_class));
            } 
            else if (type == "air_switch") 
            {
               cv::Size inputSize(640, 640);
                air_switch_models.emplace_back(new air_switch(engine_file_path_air_switch,inputSize));
                air_switch_models.back()->initialize();
            } 
            else if (type == "daozha_3") 
            {
                daozha_3_models.emplace_back(std::make_unique<DaoZhaClassifier>(engine_file_path_daozha_3));
            } 
            else if (type == "daozha_4") 
            {
                daozha_4_models.emplace_back(std::make_unique<DaoZhaClassifier>(engine_file_path_daozha_4));
            } 
            else if (type == "daozha_5") 
            {
                daozha_5_models.emplace_back(std::make_unique<DaoZhaClassifier>(engine_file_path_daozha_5));
            }
             else
             {
                std::cerr << "Unknown model type: " << type << std::endl;
            }

            // 增加模型类型的计数
            model_load_count[type]++;
        }
    }

// 
    // 打印成功加载的模型及数量
    std::cout << "成功加载模型:" << std::endl;
    for (const auto& entry : model_load_count) {
        std::cout << entry.first << ": " << entry.second << " 个模型" << std::endl;
    }
    std::cout << "模型总数: " << num_models << std::endl;

    auto model_end_time = std::chrono::high_resolution_clock::now(); // 模型加载结束时间
    std::chrono::duration<double> model_duration = model_end_time - model_start_time;
    std::cout << "模型加载时间: " << model_duration.count() << " 秒" << std::endl;
}

// 调整模型的函数
void adject_models(const std::map<std::string, std::string>& adject_count) {
    for (const auto& entry : adject_count) {
        const std::string& type = entry.first;
        int change_count = std::stoi(entry.second); // 将字符串 "+1" 或 "-1" 转换为整数 

        if (change_count > 0) {
            // 加载更多的模型
            for (int i = 0; i < change_count; ++i) {
                if (type == "person_1") {
                    yolov8_1_models.emplace_back(new YOLOv8(engine_file_path_yolov8_1));
                    yolov8_1_models.back()->make_pipe(true);
                } else if (type == "person_2") {
                    yolov8_2_models.emplace_back(new YOLOv8(engine_file_path_yolov8_2));
                    yolov8_2_models.back()->make_pipe(true);
                } else if (type == "person_3") {
                    yolov8_3_models.emplace_back(new YOLOv8(engine_file_path_yolov8_3));
                    yolov8_3_models.back()->make_pipe(true);
                } else if (type == "daozha_1") {
                    daozha_1_models.emplace_back(std::make_unique<DaoZhaClassifier>(engine_file_path_daozha_1));
                } else if (type == "daozha_2") {
                    daozha_2_models.emplace_back(std::make_unique<DaoZhaClassifier>(engine_file_path_daozha_2));
                }
            }
        } else if (change_count < 0) {
            // 卸载模型
            change_count = -change_count; // 取绝对值
            for (int i = 0; i < change_count; ++i) {
                if (type == "person_1" && !yolov8_1_models.empty()) {
                    yolov8_1_models.pop_back();
                } else if (type == "person_2" && !yolov8_2_models.empty()) {
                    yolov8_2_models.pop_back();
                } else if (type == "person_3" && !yolov8_3_models.empty()) {
                    yolov8_3_models.pop_back();
                } else if (type == "daozha_1" && !daozha_1_models.empty()) {
                    daozha_1_models.pop_back();
                } else if (type == "daozha_2" && !daozha_2_models.empty()) {
                    daozha_2_models.pop_back();
                }
            }
        }

        // 仅打印调整过的模型数量
        if (type == "person_1") {
            std::cout << "调整为person_1: " << yolov8_1_models.size() << " 个模型" << std::endl;
        } else if (type == "person_2") {
            std::cout << "调整为person_2: " << yolov8_2_models.size() << " 个模型" << std::endl;
        } else if (type == "person_3") {
            std::cout << "调整为person_3: " << yolov8_3_models.size() << " 个模型" << std::endl;
        } else if (type == "daozha_1") {
            std::cout << "调整为daozha_1: " << daozha_1_models.size() << " 个模型" << std::endl;
        } else if (type == "daozha_2") {
            std::cout << "调整为daozha_2: " << daozha_2_models.size() << " 个模型" << std::endl;
        }
    }
    model_flag = true;
}



//tcp模拟接收tcp.json中的报文
/*TcpSimulator::TcpSimulator(const std::string& json_file) : json_file_(json_file) {}

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

            for (int i = 0; i < 10; ++i) { // 每个任务复制 25 次，用来测试性能
                for (const auto& type : typeList) {
                    for (const auto& imagePath : imagePathList) {
                        Task task;
                        task.image_id = imagePath.asString().substr(5, 7) + "_" + std::to_string(i);
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
*/

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


// 回调函数：将数据写入内存
size_t write_callback(void* contents, size_t size, size_t nmemb, std::vector<uchar>* user_data) {
    size_t total_size = size * nmemb;
    user_data->insert(user_data->end(), (uchar*)contents, (uchar*)contents + total_size);
    return total_size;
}

// 从FTPS服务器下载图像
ImageLoad download_image(const std::string& remote_file_path, CURL* curl) {
    CURL* curl_copy = curl_easy_duphandle(curl);  // 复制已有的CURL句柄,避免多线程竞争
    if (!curl_copy) {
        std::cerr << "CURL duplication failed" << std::endl;
        return ImageLoad();
    }

    CURLcode res;
    std::vector<uchar> buffer;

    // 设置URL和回调函数
    std::string full_remote_path = "ftps://172.20.63.203:10012/" + remote_file_path;
    curl_easy_setopt(curl_copy, CURLOPT_URL, full_remote_path.c_str());
    curl_easy_setopt(curl_copy, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl_copy, CURLOPT_WRITEDATA, &buffer);

    // 执行下载
    res = curl_easy_perform(curl_copy);
    if (res != CURLE_OK) {
        std::cerr << "Image download failed: " << curl_easy_strerror(res) << std::endl;
        curl_easy_cleanup(curl_copy);
        return ImageLoad();  // 返回一个空的ImageLoad对象
    }

    // 将下载的数据转换为cv::Mat
    cv::Mat image = cv::imdecode(buffer, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Failed to decode image" << std::endl;
        curl_easy_cleanup(curl_copy);
        return ImageLoad();  // 返回一个空的ImageLoad对象
    }

    curl_easy_cleanup(curl_copy);  // 清理复制的CURL句柄

    // 返回ImageLoad对象
    return ImageLoad(remote_file_path, image);
}


// 查找图像数据
cv::Mat find_image_by_url( std::vector<ImageLoad>& imageLoads,  std::string& url) {
    for (auto& imgLoad : imageLoads) {
        if (imgLoad.task_url == url && imgLoad.ref_count == 0) {
           
                imgLoad.ref_count = 1;  // 只将第一个找到的任务引用计数设置为 1
            
            return imgLoad.image; // 找到匹配的图像，返回它
        }
    }

    std::cerr << "Image with URL " << url << " not found." << std::endl;
    return cv::Mat(); // 返回一个空的cv::Mat表示未找到
}



//HTTP服务接收任务

bool output_done = false;
const int PORT = 8020;
int task_cout = 0;
bool tcp_start = false;
auto program_start_time = std::chrono::high_resolution_clock::now();


void monitor_queue() {//队列监控
    while (true) {
        std::unique_lock<std::mutex> lock(queue_mutex);
        queue_cond_var.wait(lock, [] {
            return !task_queue.empty() || !buffered_tasks.empty();
        });
       
        // 如果任务队列为空且缓冲区有任务，将缓冲区任务转移到任务队列
       if (task_queue.empty() && !buffered_tasks.empty()) {
    while (!buffered_tasks.empty()) {
        task_queue.push(std::move(buffered_tasks.front())); // 使用std::move避免拷贝
        buffered_tasks.pop_front();
    }
     std::cerr << "释放缓冲区前，缓冲区大小: " << buffered_tasks.size() << std::endl;
            buffered_tasks.clear(); // 清空缓冲区
     std::cerr << "释放缓冲区后，缓冲区大小: " << buffered_tasks.size() << std::endl;
    task_cout = task_queue.size(); // 记录本次累计任务数量
    std::cerr << "缓冲区任务转移到队列，当前队列任务数: " << task_queue.size() << std::endl;
} 
 
    }
}

void send_error_response(int client_socket, int code, const std::string& message) {
    Json::Value response_json;
    response_json["code"] = code;
    response_json["message"] = message;
    Json::StreamWriterBuilder writer;
    std::string response = Json::writeString(writer, response_json);

    std::string http_response = "HTTP/1.1 " + std::to_string(code) + " Error\r\n"
                                "Content-Type: application/json\r\n"
                                "Content-Length: " + std::to_string(response.length()) + "\r\n"
                                "\r\n" + response;
    send(client_socket, http_response.c_str(), http_response.length(), 0);
}

void handle_client(int client_socket) {
    
try {
    if (task_queue.empty() && buffered_tasks.empty()) { // 在接收任务之前，如果缓冲区和任务队列都为空，则释放图片数组
        std::cerr << "\n任务队列和缓冲区都为空" << std::endl;
        std::cerr << "释放 imageLoads" << std::endl;
        
            const int loadSize = imageLoads.size();
           for (auto& imgLoad : imageLoads) {
            imgLoad.image.release();  // 释放图像资源（如使用OpenCV的cv::Mat）
        }
        imageLoads.clear();
        imageLoads.shrink_to_fit(); // 释放不再需要的内存
        std::vector<ImageLoad>().swap(imageLoads);
           // std::cerr <<  imageLoads.size() << std::endl;

           if (loadSize>=1900) {//寄存区图片达到2000张，释放内存
           //malloc_trim(0);  // 将空闲内存还给操作系统
           }
    }

    char buffer[5242880] = {0}; // 分配报文缓存区
    int valread = read(client_socket, buffer, 5242880);
    std::string request_data(buffer, valread);

    std::string response;
    int response_code = 500; // 默认异常响应

    // 解析请求行并获取路径
    size_t path_end = request_data.find("HTTP/1.1");
    if (path_end == std::string::npos) {
        std::cerr << "Invalid HTTP request." << std::endl;
        response_code = 400; // 请求错误
    } else {
        std::string request_line = request_data.substr(0, path_end);
        size_t path_start = request_line.find(" ");
    if (path_start == std::string::npos) {
        std::cerr << "Invalid HTTP request." << std::endl;
        response_code = 400; // 请求错误
    } else {
        std::string path = request_line.substr(path_start + 1, path_end - path_start - 1);
        // 移除路径首尾的空白字符
        path.erase(path.find_last_not_of(" \n\r\t") + 1);
        path.erase(0, path.find_first_not_of(" \n\r\t"));
        std::cerr << "当前端口请求路径为 '" << path << "'" << std::endl;
        // 检查路径是否为 /picAnalyse
        if (path == "/picAnalyse") {
            // 查找 POST 数据的起始位置
            std::string post_data;
            size_t pos = request_data.find("\r\n\r\n");
            if (pos != std::string::npos) {
                post_data = request_data.substr(pos + 4); // 获取 POST 数据部分
                }

                Json::Value root;
                Json::CharReaderBuilder reader;
                std::string errs;
                std::istringstream s(post_data);

            if (Json::parseFromStream(reader, s, &root, &errs)) {
                try {
                    std::unique_lock<std::mutex> lock(queue_mutex);
                    std::deque<Task> new_tasks;
                    int taskCount = 0; // 计数器初始化为0
                    int maxTasks = 100; // 最大任务数量,限制每次接收的任务数量，超出则丢弃
                    auto start_time = std::chrono::high_resolution_clock::now();
                     // 处理接收到的JSON数据 
                    if (imageLoads.size()<=2000) {//设置图片寄存区大小
                      
                     for (const auto& item : root) {
                        std::string requestHostIp = item["requestHostIp"].asString();
                        std::string requestHostPort = item["requestHostPort"].asString();
                        std::string requestId = item["requestId"].asString();

                      for (const auto& object : item["objectList"]) {
                         std::string objectId = object["objectId"].asString();
                         const auto& typeList = object["typeList"];
                         const auto& imagePathList = object["imagePathList"];

                        for (const auto& type : typeList) {

                         std::vector<std::future<ImageLoad>> futures;  // 存储异步任务

                        for (const auto& imagePath : imagePathList) {
                         if (taskCount >= maxTasks) {
                                break; // 如果任务数量达到限制，跳出内层循环
                            }

                            Task task;
                            task.task_type = type.asString();
                            task.requestId = requestId;
                            task.type_count = typeList.size();
                            std::filesystem::path p(imagePath.asString());
                            task.image_id = p.stem().string() + "_" + task.task_type;
                            task.url = imagePath.asString();
                            task.objectId = objectId;

                                // 检查是否已经存在相同的任务
                            TaskInfo* existingTask = findTaskByRequestIdAndImageUrl(task.requestId, task.url);
                            if (existingTask == nullptr) {
                                // 如果任务不存在，创建新的任务
                             addTaskToBuffer(task.requestId, task.url, ResultMessage(), task.type_count,task.objectId);
                            } else {
                                 // 如果任务已存在，可以选择更新现有任务或直接跳过
                             std::cout << "任务已存在，跳过创建" << std::endl;
                                }

                               // 使用 std::async 并行下载图片
                               futures.push_back(std::async(std::launch::async, download_image, task.url, curl_1));

                                new_tasks.push_back(task); // 将任务放入新任务列表
                                taskCount++; // 计数器增加
                                    }

                             // 等待所有异步下载任务完成
                            for (auto& future : futures) {
                            imageLoads.push_back(future.get());  // 获取下载结果并添加到 imageLoads
                            }

 
                            if (taskCount >= maxTasks) {
                             break; // 如果任务数量达到限制，跳出外层循环
                                 }
                            }

                            if (taskCount >= maxTasks) {
                             break; // 如果任务数量达到限制，跳出object循环
                                }
                            }

                            if (taskCount >= maxTasks) {
                                break; // 如果任务数量达到限制，跳出item循环
                            }
                       
                      }
                        
                    }else{
                            std::cerr << "缓存区已满，请稍后重试" << std::endl;
                            response_code = 500; // 处理请求时出现异常
                        }

                        auto end_time = std::chrono::high_resolution_clock::now();
                         std::chrono::duration<double> download_duration = end_time - start_time;
                         std::cerr << "本次图片的下载总时间为: " << download_duration.count() << " 秒" << std::endl;


                        // 处理任务队列
                        if (task_queue.empty()) {
                            while (!new_tasks.empty()) {
                                task_queue.push(new_tasks.front());
                                new_tasks.pop_front();
                            }

                            std::cerr << "\n新任务入队，目前任务数: " << task_queue.size() << std::endl;

                        } else {
                            while (!new_tasks.empty()) {
                                buffered_tasks.push_back(new_tasks.front());
                                new_tasks.pop_front();
                            }
                            std::cerr << "当前任务正在执行，新任务进入缓冲区 " << std::endl;
                        }

                        task_cout = task_queue.size(); // 记录本次任务数量
                        program_start_time = std::chrono::high_resolution_clock::now(); // 任务开始时间
                        output_done = false;
                        tcp_start = true;
                        queue_cond_var.notify_all();
                        response_code = 200; // 请求成功
                    } catch (...) {
                        std::cerr << "处理请求时发生异常" << std::endl;
                        response_code = 500; // 处理请求时出现异常
                    }
                } else {
                    std::cerr << "Failed to parse JSON: " << errs << std::endl;
                    response_code = 400; // 请求语法错误
                }
            } else if (path == "/preloadModels") {
    // 新增的 /preloadModels 处理逻辑
    // 查找 POST 数据的起始位置
    std::string post_data;
    size_t pos = request_data.find("\r\n\r\n");
    if (pos != std::string::npos) {
        post_data = request_data.substr(pos + 4); // 获取 POST 数据部分
    }

    Json::Value root;
    Json::CharReaderBuilder reader;
    std::string errs;
    std::istringstream s(post_data);

    if (Json::parseFromStream(reader, s, &root, &errs)) {
        try {
            std::map<std::string, int> task_type_count; // 使用map来存储任务类型和对应的任务数量

            for (const auto& item : root) {
                const auto& taskTypeCount = item["taskTypeCount"];
                for (const auto& type : taskTypeCount.getMemberNames()) {
                    task_type_count[type] = taskTypeCount[type].asInt(); // 将任务类型和数量添加到map中
                }
            }

            // 你可以在此处对 task_type_count 进行任何需要的操作
            initialize_models(task_type_count);

            response_code = 200; // 请求成功
        } catch (...) {
            std::cerr << "处理请求时发生异常" << std::endl;
            response_code = 500; // 处理请求时出现异常
        }
    } else {
        std::cerr << "Failed to parse JSON: " << errs << std::endl;
        response_code = 400; // 请求语法错误
    }
}else if (path == "/adjectModels") {
    // 新增的 /preloadModels 处理逻辑
    // 查找 POST 数据的起始位置
     model_flag = false;//暂停任务处理
    std::string post_data;
    size_t pos = request_data.find("\r\n\r\n");
    if (pos != std::string::npos) {
        post_data = request_data.substr(pos + 4); // 获取 POST 数据部分
    }

    Json::Value root;
    Json::CharReaderBuilder reader;
    std::string errs;
    std::istringstream s(post_data);

    if (Json::parseFromStream(reader, s, &root, &errs)) {
        try {
            std::map<std::string, string> adject_count; // 使用map来存储任务类型和对应的任务数量

            for (const auto& item : root) {
                const auto& adjectCount = item["adjectCount"];
                for (const auto& type : adjectCount.getMemberNames()) {
                    adject_count[type] = adjectCount[type].asString(); // 将任务类型和数量添加到map中
                }
            }

            // 可以在此处对 task_type_count 进行任何需要的操作
            adject_models(adject_count);
           for (const auto& entry : adject_count) {
    std::cout << "Task Type: " << entry.first << ", Adjustment: " << entry.second << std::endl;
}


            response_code = 200; // 请求成功
        } catch (...) {
            std::cerr << "处理请求时发生异常" << std::endl;
            response_code = 500; // 处理请求时出现异常
        }
    } else {
        std::cerr << "Failed to parse JSON: " << errs << std::endl;
        response_code = 400; // 请求语法错误
    }
}


            else {
                std::cerr << "Invalid path: " << path << std::endl;
                response_code = 404; // 未找到路径
            }
        }
    }

    // 构建响应消息
    Json::Value response_json;
    response_json["code"] = response_code;
    Json::StreamWriterBuilder writer;
    response = Json::writeString(writer, response_json);

    // 发送响应
    std::string http_response = "HTTP/1.1 " + std::to_string(response_code) + " OK\r\n"
                                "Content-Type: application/json\r\n"
                                "Content-Length: " + std::to_string(response.size()) + "\r\n"
                                "\r\n" + response;
    send(client_socket, http_response.c_str(), http_response.size(), 0);
    close(client_socket);

    } catch (const std::exception& e) {
        std::cerr << "Exception occurred: " << e.what() << std::endl;
        send_error_response(client_socket, 500, e.what());
    } catch (...) {
        std::cerr << "An unknown exception occurred." << std::endl;
        send_error_response(client_socket, 500, "An unknown error occurred.");
    }

    close(client_socket);
}



void tcp_server() {
    int server_fd, client_socket;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);

    // 创建套接字
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }

    // 设置套接字选项
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt))) {
        perror("setsockopt");
        exit(EXIT_FAILURE);
    }

    // 绑定地址和端口
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);

    if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
        perror("bind failed");
        exit(EXIT_FAILURE);
    }

    // 监听连接
    if (listen(server_fd, 3) < 0) {
        perror("listen");
        exit(EXIT_FAILURE);
    }

    // 接受并处理客户端连接（多线程模式）
    while (true) {
        if ((client_socket = accept(server_fd, (struct sockaddr*)&address, (socklen_t*)&addrlen)) < 0) {
            perror("accept");
            continue;
        }

        // 在这里处理客户端连接
       // handle_client(client_socket);

        // 关闭客户端连接
       // close(client_socket);

            // 使用新线程处理客户端连接
        std::thread(handle_client, client_socket).detach();

    }
}



// 每个任务结束后返回处理结果的POST请求的函数
void send_post_request(const std::string& url, const ResultMessage& resultMessage) {
    CURL *curl_2;
    CURLcode res;

    curl_2 = curl_easy_init();
    if (curl_2) {
        Json::Value jsonData;
        jsonData["requestid"] = resultMessage.requestid;  // 设置 requestid

        Json::Value resultListJson;
        resultListJson["objectId"] = resultMessage.resultList.objectId;  // 设置 objectId

        Json::Value resultsArray(Json::arrayValue);  // 用于存储 results 数组

        // 遍历 resultList 中的所有 Result
        for (const auto& resultPtr : resultMessage.resultList.results) {
            Json::Value resultJson;
            resultJson["type"] = resultPtr->type;
            resultJson["value"] = resultPtr->value;
            resultJson["code"] = resultPtr->code;
            resultJson["resImagePath"] = resultPtr->resImagePath;

            // pos 保持为字符串类型，直接插入，不解析为 JSON
            resultJson["pos"] = resultPtr->pos;  // 这里将 pos 作为字符串传输

            resultJson["conf"] = resultPtr->conf;
            resultJson["desc"] = resultPtr->desc;

            // 将单个 result 加入 results 数组
            resultsArray.append(resultJson);
        }

        // 将 results 数组加入 resultList
        resultListJson["results"] = resultsArray;

        // 将 resultList 加入根节点
        jsonData["resultList"] = resultListJson;

        // 将 JSON 数据转换为字符串
        Json::StreamWriterBuilder writer;
        std::string jsonString = Json::writeString(writer, jsonData);

        // 设置 CURL 参数，准备发送 POST 请求
        curl_easy_setopt(curl_2, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl_2, CURLOPT_POSTFIELDS, jsonString.c_str());
        curl_easy_setopt(curl_2, CURLOPT_HTTPHEADER, curl_slist_append(nullptr, "Content-Type: application/json"));

        // 执行请求
        res = curl_easy_perform(curl_2);
        if (res != CURLE_OK) {
            std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
        }

        // 清理
        curl_easy_cleanup(curl_2);
    }
}




//建立FTPS连接
void initialize_ftps(const std::string& ftp_url, const std::string& username, const std::string& password) {
    CURLcode res;

    curl_global_init(CURL_GLOBAL_ALL);
    curl = curl_easy_init();//上传通道
    curl_1 = curl_easy_init();//下载通道
    
    if (curl) {
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L); // 禁用 SSL 证书验证
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L); // 禁用主机名验证

        // 设置 FTP URL（使用 ftps:// 前缀表示 FTPS）
        curl_easy_setopt(curl, CURLOPT_URL, ftp_url.c_str());

        // 启用 FTPS（隐式）
        curl_easy_setopt(curl, CURLOPT_USE_SSL, CURLUSESSL_ALL);

        // 设置 FTP 用户名和密码
        curl_easy_setopt(curl, CURLOPT_USERNAME, username.c_str());
        curl_easy_setopt(curl, CURLOPT_PASSWORD, password.c_str());

        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);  // 设置超时时间为 300 秒
        curl_easy_setopt(curl, CURLOPT_BUFFERSIZE, 32768L); 

        // 测试连接
        res = curl_easy_perform(curl);
        if (res != CURLE_OK) {
            std::cerr << "FTPS connection failed: " << curl_easy_strerror(res) << std::endl;
        } else {
            std::cout << "Connected successfully to " << ftp_url << std::endl;
            ftps_connected = true;
        }
    } else {
        std::cerr << "Failed to initialize curl" << std::endl;
    }

    if (curl_1) {
        curl_easy_setopt(curl_1, CURLOPT_SSL_VERIFYPEER, 0L); // 禁用 SSL 证书验证
        curl_easy_setopt(curl_1, CURLOPT_SSL_VERIFYHOST, 0L); // 禁用主机名验证

        // 设置 FTP URL（使用 ftps:// 前缀表示 FTPS）
        curl_easy_setopt(curl_1, CURLOPT_URL, ftp_url.c_str());

        // 启用 FTPS（隐式）
        curl_easy_setopt(curl_1, CURLOPT_USE_SSL, CURLUSESSL_ALL);

        // 设置 FTP 用户名和密码
        curl_easy_setopt(curl_1, CURLOPT_USERNAME, username.c_str());
        curl_easy_setopt(curl_1, CURLOPT_PASSWORD, password.c_str());

        curl_easy_setopt(curl_1, CURLOPT_TIMEOUT, 30L);  // 设置超时时间为 300 秒
        curl_easy_setopt(curl_1, CURLOPT_BUFFERSIZE, 32768L); 

        // 测试连接
        res = curl_easy_perform(curl_1);
        if (res != CURLE_OK) {
            std::cerr << "FTPS connection failed: " << curl_easy_strerror(res) << std::endl;
        } else {
            std::cout << "Connected successfully to " << ftp_url << std::endl;
            ftps_connected = true;
        }
    } else {
        std::cerr << "Failed to initialize curl_1" << std::endl;
    }
}

//上传结果到服务器

size_t read_callback(void* ptr, size_t size, size_t nmemb, void* stream) {
    std::vector<uint8_t>* data = static_cast<std::vector<uint8_t>*>(stream);
    size_t total_size = size * nmemb;

    if (data->empty()) return 0;

    size_t to_copy = std::min(total_size, data->size());
    std::memcpy(ptr, data->data(), to_copy);
    data->erase(data->begin(), data->begin() + to_copy);
    return to_copy;
}
//上传结果
void upload_file(const cv::Mat& image, const std::string& remote_file_path) {
    if (!ftps_connected) {
        std::cerr << "FTPS not connected. Cannot upload file." << std::endl;
        return;
    }

    CURLcode res;
    std::vector<uint8_t> image_data;

    // 图像编码为JPG并压缩
    std::vector<int> params;
    params.push_back(cv::IMWRITE_JPEG_QUALITY);
    params.push_back(90); // 设置压缩质量,curl版本问题，7.68.0版本文件上传不完整

    std::vector<uchar> buf;
    if (!cv::imencode(".jpg", image, buf, params)) {
        std::cerr << "Image encoding failed." << std::endl;
        return;
    }

    image_data.assign(buf.begin(), buf.end());
   // std::cout << remote_file_path << " 上传大小为： " << image_data.size() << std::endl;
 //std::cerr <<imageLoads.size() << std::endl;
    // 设置上传url
    std::string full_remote_path = "ftps://172.20.63.203:10012/" + remote_file_path;
    curl_easy_setopt(curl, CURLOPT_URL, full_remote_path.c_str());
    curl_easy_setopt(curl, CURLOPT_UPLOAD, 1L);
    curl_easy_setopt(curl, CURLOPT_READFUNCTION, read_callback);
    curl_easy_setopt(curl, CURLOPT_READDATA, &image_data);
    curl_easy_setopt(curl, CURLOPT_INFILESIZE_LARGE, (curl_off_t)image_data.size());

    // 详细输出
   // curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);

    // 执行文件上传
    res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        std::cerr << "Image upload failed: " << curl_easy_strerror(res) << std::endl;
       
    } else {
        //std::cout << "Image uploaded successfully." << std::endl;
    }
      
}

/*************************************************各类图像算法*****************************************************/

bool stop = false;
int start_count = 0; //处理函数调用次数
 auto program_end_time = std::chrono::high_resolution_clock::now();

// 处理 "person" 任务的逻辑
bool process_person_task(const cv::Mat& image, cv::Mat& result_image, std::shared_ptr<Result>& result) {
    cv::Size size = cv::Size{640, 640};
    int num_labels = 40;
    int topk = 100;
    float score_thres = 0.7f;
    float iou_thres = 0.7f;
    std::vector<Object> objs;
    objs.clear();
    bool analysis_successful = false;

    for (size_t i = 0; i < yolov8_1_models.size(); ++i) {
        if (model_mutexes[i].try_lock()) {
            try {
                yolov8_1_models[i]->copy_from_Mat(image, size);
                yolov8_1_models[i]->infer();
                yolov8_1_models[i]->postprocess(objs, score_thres, iou_thres, topk, num_labels);
                yolov8_1_models[i]->draw_objects(image, result_image, objs, CLASS_NAMES, COLORS, "11");
                analysis_successful = true;
            } catch (const std::exception& e) {
                std::cerr << "Error during person task processing: " << e.what() << std::endl;
                result->code = "2002"; // 设置错误代码
            }
            model_mutexes[i].unlock();
            break;
        }

    }

          if (!analysis_successful) {
            result->code = "2002"; // 如果分析未成功，设置错误代码
        } else {
            result->code = "2000"; // 确保分析成功时设置为正常状态码
        }

    return analysis_successful;
}

// 处理 "daozha" 任务的逻辑
bool process_daozha_task(const cv::Mat& image, cv::Mat& result_image, std::shared_ptr<Result>& result) {
    bool analysis_successful = false;

    for (size_t i = 0; i < daozha_1_models.size(); ++i) {
        if (model_mutexes[i].try_lock()) {
            try {
                auto ress = daozha_1_models[i]->infer(image);
                std::string result_desc = ress.first; // 获取预测结果
                float confidence = ress.second; // 获取置信度
                result->conf = confidence;
                result->desc = result_desc;
                cv::putText(image, result_desc, cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
                result_image = image;
                analysis_successful = true;
            } catch (const std::exception& e) {
                std::cerr << "Error during daozha task processing: " << e.what() << std::endl;
                result->code = "2002"; // 设置错误代码
            }
            model_mutexes[i].unlock();
            break;
        }
        
    }

     if (!analysis_successful) {
            result->code = "2002"; // 如果分析未成功，设置错误代码
        } else {
            result->code = "2000"; // 确保分析成功时设置为正常状态码
        }
    return analysis_successful;
}

// 处理 "defect_25nc" 任务的逻辑
bool process_defect_25nc_task(cv::Mat& image, cv::Mat& result_image, std::shared_ptr<Result>& result) {
    bool analysis_successful = false;
  // 创建一个 JSON 数组，用于存储 "areas" 信息
    Json::Value areas(Json::arrayValue);
    for (size_t i = 0; i < defect_25nc_1_models.size(); ++i) {
        if (model_mutexes[i].try_lock()) {
            try {
                defect_25nc_1_models[i]->blobFromImage(image); // 移除 const 限定符
                defect_25nc_1_models[i]->doInference();
                std::vector<Object> objects;
                std::vector<Object> results = defect_25nc_1_models[i]->decodeOutputs(objects);
               
                //result->pos = areas.toStyledString();;
                defect_25nc_1_models[i]->drawAndSaveResults(image, results,result->pos,result->conf,result->desc); // 移除 const 限定符
                
                std::cout << "Detected areas: " << result->pos << std::endl;
                 result_image = image;
                analysis_successful = true;

                
            } catch (const std::exception& e) {
                std::cerr << "检测过程中出现错误: " << e.what() << std::endl;
                result->code = "2002"; // 设置错误代码
            }
            model_mutexes[i].unlock();
            break;
        }

    }
   if (!analysis_successful) {
            result->code = "2002"; // 如果分析未成功，设置错误代码
        } else {
            result->code = "2000"; // 确保分析成功时设置为正常状态码
        }
    return analysis_successful;
}

// 处理 "defect_25nc" 任务的逻辑
bool process_air_switch_task(cv::Mat& image, cv::Mat& result_image, std::shared_ptr<Result>& result) {
    bool analysis_successful = false;
  // 创建一个 JSON 数组，用于存储 "areas" 信息
    Json::Value areas(Json::arrayValue);
    for (size_t i = 0; i <air_switch_models.size(); ++i) {
        if (model_mutexes[i].try_lock()) {
            try {
                air_switch_models[i]->blobFromImage(image); // 移除 const 限定符
                air_switch_models[i]->doInference();
                std::vector<Object> objects;
                std::vector<Object> results = air_switch_models[i]->decodeOutputs(objects);
               
                //result->pos = areas.toStyledString();;
                air_switch_models[i]->drawAndSaveResults(image, results,result->pos,result->conf,result->desc); // 移除 const 限定符
                
                std::cout << "Detected areas: " << result->pos << std::endl;
                 result_image = image;
                analysis_successful = true;

                
            } catch (const std::exception& e) {
                std::cerr << "检测过程中出现错误: " << e.what() << std::endl;
                result->code = "2002"; // 设置错误代码
            }
            model_mutexes[i].unlock();
            break;
        }

    }
   if (!analysis_successful) {
            result->code = "2002"; // 如果分析未成功，设置错误代码
        } else {
            result->code = "2000"; // 确保分析成功时设置为正常状态码
        }
    return analysis_successful;
}

// 处理 "number_class" 任务的逻辑
bool process_number_class_task(cv::Mat& image, cv::Mat& result_image, std::shared_ptr<Result>& result) {
    bool analysis_successful = false;
  
    for (size_t i = 0; i < number_class_models.size(); ++i) {
        if (model_mutexes[i].try_lock()) {
            try {
                //result->pos = areas.toStyledString();;
              
                
              
                
                number_class_models[i]->infer(image,  result->conf, result->desc);

                number_class_models[i]->drawResult(image, result->desc, result->conf);

                 result_image = image;
                analysis_successful = true;

                
            } catch (const std::exception& e) {
                std::cerr << "检测过程中出现错误: " << e.what() << std::endl;
                result->code = "2002"; // 设置错误代码
            }
            model_mutexes[i].unlock();
            break;
        }

    }
   if (!analysis_successful) {
            result->code = "2002"; // 如果分析未成功，设置错误代码
        } else {
            result->code = "2000"; // 确保分析成功时设置为正常状态码
        }
    return analysis_successful;
}

// 处理 "ele_number" 任务的逻辑
bool process_ele_number_task(cv::Mat& image, cv::Mat& result_image, std::shared_ptr<Result>& result) {
    bool analysis_successful = false;
 cv::Mat Image_cropped = image.clone();
  std::cerr << "裁剪模型个数: " << Elenumber_models_bp.size() << std::endl;
 
    for (size_t i = 0; i <  Elenumber_models_bp.size(); ++i) {
        if (model_mutexes[i].try_lock()) {
            try {
               
                 Elenumber_models_bp[i]->blobFromImage(image); 
                 Elenumber_models_bp[i]->doInference();
                std::vector<Object> objects;
                std::vector<Object> results =  Elenumber_models_bp[i]->decodeOutputs(objects);
                // Elenumber_models_bp[i]->drawAndSaveResults(image, results); // 移除 const 限定符
                 Elenumber_models_bp[i]->cropAndSaveResults(Image_cropped, results); 
                 result_image = Image_cropped;
                 analysis_successful = true;     
                 //analysis_successful = true;  
            } catch (const std::exception& e) {
                std::cerr << "检测过程中出现错误: " << e.what() << std::endl;
                result->code = "2002"; // 设置错误代码
            }
            model_mutexes[i].unlock();
            break;
        }
   
    }
      for (size_t i = 0; i < Elenumber_models_num.size(); ++i) {
            if (model_mutexes[i].try_lock()) {
                        try {
               Elenumber_models_num[i]->blobFromImage(Image_cropped);
                 Elenumber_models_num[i]->doInference();
                std::vector<Object> objects;
                std::vector<Object> results =  Elenumber_models_num[i]->decodeOutputs(objects);
                // Elenumber_models_bp[i]->drawAndSaveResults(image, results); // 移除 const 限定符
                 Elenumber_models_num[i]->drawAndSaveResults(Image_cropped, results); 
                 result_image = Image_cropped;
                 analysis_successful = true;     
            } catch (const std::exception& e) {
                std::cerr << "检测过程中出现错误: " << e.what() << std::endl;
                result->code = "2002"; // 设置错误代码
            }
            model_mutexes[i].unlock();
            break;
        }

     }
   if (!analysis_successful) {
            result->code = "2002"; // 如果分析未成功，设置错误代码
        } else {
            result->code = "2000"; // 确保分析成功时设置为正常状态码
        }
    return analysis_successful;
}

/*************************************************end*****************************************************/


// 合并同一图片任务结果逻辑函数
void merge_and_send_results(const Task& task, std::shared_ptr<Result>& result, ResultMessage& resultMessage) 
{
    std::unique_lock<std::mutex> lock(taskinfo_mutex);

    TaskInfo* taskInfo = findTaskByRequestIdAndImageUrl(task.requestId, task.url);

    if (taskInfo != nullptr) {
        taskInfo->resultMessage.resultList.results.push_back(result);

        if (taskInfo->type_count == 1) {
            send_post_request("http://127.0.0.1:8021/picAnalyseRetNotify", taskInfo->resultMessage);
            taskInfoBuffer.erase(std::remove_if(taskInfoBuffer.begin(), taskInfoBuffer.end(),
                                                [&](const TaskInfo& t) {
                                                    return t.requestId == task.requestId && t.imageUrl == task.url;
                                                }),
                                 taskInfoBuffer.end());
        } else {
            taskInfo->type_count--;
        }
    } else {
        ResultMessage newMessage;
        newMessage.requestid = task.requestId;
        newMessage.resultList.objectId = task.objectId;
        newMessage.resultList.results.push_back(result);

        // 注意这里 task.objectId 应该是可修改的非 const 引用
        addTaskToBuffer(task.requestId, task.url, newMessage, task.type_count, task.objectId); // 移除 const 限定符
    }
    lock.unlock();
}


// 处理图像的主函数
void process_image(std::atomic<int>& tasks_processed) {
    const int max_retry_count = 10;

    while (true) {
        Task task;
        ResultMessage resultMessage;
        auto result = std::make_shared<Result>();
        result->code = "2000";

        // 获取任务
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            queue_cond_var.wait(lock, [] { return !task_queue.empty() || stop; });

            if (!task_queue.empty()) {
                task = task_queue.front();
                resultMessage.requestid = task.requestId;
                resultMessage.resultList.objectId = task.objectId;
                result->type = task.task_type;
                result->value = "1";
                //result->resImagePath = task.url;
                result->resImagePath = "results/";
                task_queue.pop();
            }
        }

         if (task.url.empty()) {
        //imageLoads.clear();
         program_end_time = std::chrono::high_resolution_clock::now();
         if( !output_done&&tcp_start){
             
           std::chrono::duration<double> total_duration = program_end_time - program_start_time;
           
           std::cout << "本次任务数量: " << task_cout << std::endl;
           std::cout << "本次处理时间: " << total_duration.count() << "秒" << std::endl;
           std::cout << "剩余任务数量： " << task_queue.size() << std::endl;
           std::cout << "已处理任务数量： " << start_count << std::endl;
           std::cout << "imageLoads占用内存" <<imageLoads.capacity()<< std::endl;
           task_cout = 0;
        // std::cerr << "任务为空: " << task.image_id << std::endl;
           output_done = true;


         }
           continue; // 如果task为空，继续等待
                  
            
        }


        cv::Mat res, image;
        image = find_image_by_url(imageLoads, task.url);

        if (image.empty()) {
            std::cerr << "无法读取图像: " << task.url << std::endl;
            result->code = "2001";
            result->desc = "image load failed.";
            resultMessage.resultList.results.push_back(result);
            send_post_request("http://127.0.0.1:41577/picAnalyseRetNotify", resultMessage);
            continue;
        }

         start_count++;//执行任务记数
        auto start_time = std::chrono::high_resolution_clock::now();

        // 处理图像任务
        bool processed = false;
        int retry_count = 0;
        while (model_flag&&!processed && retry_count < max_retry_count) {
            if (task.task_type == "person") {
                processed = process_person_task(image, res, result);
            } else if (task.task_type == "daozha") {
                processed = process_daozha_task(image, res, result);
            } else if (task.task_type == "defect_25nc") {
                processed = process_defect_25nc_task(image, res, result);
            }else if (task.task_type == "ele_number") {
                processed = process_ele_number_task(image, res, result);
            }else if (task.task_type == "number_class") {
                processed = process_number_class_task(image, res, result);
            }else if (task.task_type == "air_switch") {
                processed = process_air_switch_task(image, res, result);
            }
        
            if (!processed) {
                retry_count++;
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }

        if (!processed) {
            std::cerr << "任务 " << task.image_id << " 超过最大重试次数，舍弃该任务" << std::endl;
            continue;
        }

     std::this_thread::sleep_for(std::chrono::milliseconds(10));

     ImageResult imgresults(task.image_id,resultMessage.requestid,res);//图像结果加入结果队列

     std::shared_ptr<ImageResult> imgResultPtr = std::make_shared<ImageResult>(imgresults.task_id,imgresults.request_id, imgresults.image);

    // 锁定互斥量以保证线程安全
    std::lock_guard<std::mutex> lock(mat_mutex);

    // 将智能指针添加到队列中
    result_queue.push(imgResultPtr);
   
        // 合并和发送结果
    merge_and_send_results(task, result, resultMessage);

        // 处理完成后的后续操作
    tasks_processed++;
      
    }
}



void monitor_and_upload() {
    while (true) {
        std::shared_ptr<ImageResult> result;
        {
            std::lock_guard<std::mutex> lock(mat_mutex);
            if (!result_queue.empty()) {
                result = result_queue.front();
                result_queue.pop();
            }
        }

        if (result) {
            try {
                upload_file(result->image, "results/" +result->request_id+"_"+result->task_id + ".jpg");
               std::cerr << "结果已上传： " + std::string("results/") +result->request_id+"_"+ result->task_id + ".jpg" << std::endl;

            } catch (const std::exception& e) {
                std::cerr << "Error processing image: " << e.what() << std::endl;
            }
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
}


int main(int argc, char* argv[]) {
      cv::setNumThreads(1);  // 限制 OpenCV 只使用一个线程
     //imageLoads.reserve(2000);
    std::thread server_thread(tcp_server);//开启TCP服务持续接收报文

    std::thread monitor_thread(monitor_and_upload);//自动检测并上传线程

    std::thread monitor_queue_thread(monitor_queue);// 启动任务队列监控线程
    monitor_queue_thread.detach(); // 将线程设为分离状态，独立运行

   // TcpSimulator tcp_simulator("request.json");//本地文件模拟TCP报文 本地测试用，现在没有使用

    // 获取libcurl版本信息
    curl_version_info_data *curl_version = curl_version_info(CURLVERSION_NOW);
    
    // 输出libcurl的版本号
    std::cout << "libcurl version: " << curl_version->version_num << std::endl;
    
    // 也可以输出版本字符串
    std::cout << "libcurl version string: " << curl_version->version << std::endl;
    
   
  
   /* if (!tcp_simulator.LoadTasks()) {
        return 1;
    }*/

    Task task;
    int total_tasks = 0; // 总任务数
  

    std::atomic<int> tasks_processed(0); // 已处理任务数

    

     cudaSetDevice(1);

    // 初始化模型
   // initialize_models();


    const int num_threads = 1; // 线程数量


    std::string ftp_url = "ftps://172.20.63.203:10012";  // FTPS URL
    std::string username = "dfftp";
    std::string password = "dfe2k_";
    initialize_ftps(ftp_url, username, password);//建立FTPS连接
     

   
    std::cout  <<"服务开启" << std::endl;
    std::cout << "任务数量： " << total_tasks << std::endl;
   // std::cout << "线程数: " << num_threads << " 模型数: " << num_models << std::endl;
   
    std::cout << "剩余任务数量： " << task_queue.size() << std::endl;
   
  

     
    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {//开启多线程并发执行任务
        threads.emplace_back(process_image, std::ref(tasks_processed)); // 每个线程循环执行 process_image，从队列中取出任务，并发执行
    }

    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop = true;
    }

    queue_cond_var.notify_all();

    for (auto& thread : threads) {
        thread.join();
    }

      server_thread.join();

      monitor_thread.join();

      monitor_queue_thread.join();
     
    // 打印每个线程的执行时间
   /* for (size_t i = 0; i < thread_durations.size(); ++i) {
        std::cout << "线程 " << i + 1 << " 执行时间: " << thread_durations[i] << " 秒" << std::endl;
    }*/

    return 0;
}
