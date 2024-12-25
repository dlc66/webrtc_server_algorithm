#ifndef TCPSIMULATOR_H
#define TCPSIMULATOR_H

#include <string>
#include <queue>
#include <opencv2/opencv.hpp>
#include <json.h>
// 任务结构体
struct Task {
    std::string requestId;//请求Id
    std::string image_id;
    std::string task_type;//任务类型
    std::string url;
    std::string objectId;//巡视发送来的点位编码
    int type_count;//类型数量
};
struct rect_name
{
    rect_name() {}
    cv::Rect rect;
    std::vector<cv::Point> rect_points;
    std::string name;
};
//json解析结构体
struct AlgJsonObject
{
    AlgJsonObject(){}
    double angle{0};
    std::string des;//单位
    Json::Value argu;//标定信息参数
    std::vector<cv::Rect> area_rect;//区域area
    std::vector<rect_name> area_rect_name;//区域名称
    int FHZSP_type = 0;
    int is_Integer = 0;
    std::string opertype;//标定或非标定
    std::string point_code;//点位编码
};

// 结果结构体
struct Result {
    std::string type;
    std::string value;
    std::string code;
    std::string resImagePath;
    std::string pos;
    float conf;
    std::string desc;

    // 默认构造函数
    Result() 
    : type(""), value(""), code(""), resImagePath(""), conf(0.0), desc("") {}

    // 拷贝构造函数
    Result(const Result& other)
    : type(other.type), value(other.value), code(other.code),
      resImagePath(other.resImagePath), pos(other.pos),
      conf(other.conf), desc(other.desc) {}
};



struct ResultList {
    std::string objectId;
    std::vector<std::shared_ptr<Result>> results;  // 使用智能指针
};

struct ResultMessage {
    std::string requestid;
    ResultList resultList;
};


class TcpSimulator {
public:
    TcpSimulator(const std::string& json_file);

    // 读取和解析JSON文件
    bool LoadTasks();

    // 从任务队列中获取任务
    bool GetNextTask(Task& task);

    // 模拟发送处理结果并保存到本地
    void SendResult(const std::string& result);

private:
    std::string json_file_;
    std::queue<Task> task_queue_;//创建任务队列
 
};

#endif // TCPSIMULATOR_H
