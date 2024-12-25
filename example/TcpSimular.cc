#include <TcpSimular.h>
#include <iostream>
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
//tcp模拟接收tcp.json中的报文
TcpSimulator::TcpSimulator(const std::string& json_file) : json_file_(json_file) {}

bool TcpSimulator::LoadTasks() {//结果接收
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
