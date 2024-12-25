#include <queue>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <iostream>
#include <cppconn/driver.h>
#include <cppconn/connection.h>
#include <cppconn/statement.h>
#include <cppconn/prepared_statement.h>
#include <cppconn/resultset.h>
#include <cppconn/metadata.h>
#include <cppconn/resultset_metadata.h>
#include <cppconn/exception.h>
#include <cppconn/warning.h>
#include <mysql_driver.h>
#include <mysql_connection.h>
#include <string>
class ConnectionPool {
public:
    ConnectionPool(const std::string& host, const std::string& user, const std::string& password, const std::string& database, size_t poolSize)
        : host_(host), user_(user), password_(password), database_(database), poolSize_(poolSize) {
        // 初始化连接池
        for (size_t i = 0; i < poolSize_; ++i) {
            connections_.emplace(createConnection());
        }
    }
    ~ConnectionPool(){

    }
    //获取连接
    std::unique_ptr<sql::Connection> getConnection() {
        std::unique_lock<std::mutex> lock(mutex_);
        while (connections_.empty()) {
            cond_.wait(lock);
        }
        auto conn = std::move(connections_.front());
        connections_.pop();
        return conn;
    }
    //释放连接
    void releaseConnection(std::unique_ptr<sql::Connection> conn) {
        std::unique_lock<std::mutex> lock(mutex_);
        connections_.emplace(std::move(conn));
        cond_.notify_one();//条件变量使用完毕通知
    }

private:
    //创建连接
    std::unique_ptr<sql::Connection> createConnection() {
        try {
            auto driver = sql::mysql::get_mysql_driver_instance();
            auto conn = driver->connect("tcp://" + host_ + ":13306", user_, password_);
            conn->setSchema(database_);
            return std::unique_ptr<sql::Connection>(conn);
        }catch (sql::SQLException& e) {
            std::cerr << "数据库连接失败！" << e.what() << std::endl;
            throw;
        }
    }

    std::string host_;
    std::string user_;
    std::string password_;
    std::string database_;
    size_t poolSize_;
    std::queue<std::unique_ptr<sql::Connection>> connections_;
    std::mutex mutex_;
    std::condition_variable cond_;
};
