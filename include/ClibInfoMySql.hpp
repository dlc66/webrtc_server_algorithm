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
#include <ConnectPool.hpp>

//数据库连接池
class DfCalibrateInfo {
public:
    DfCalibrateInfo(ConnectionPool& pool) : pool_(pool) {}

    std::string getCalibrateInfo(const std::string& method, const std::string& pointCode);
    std::string getCalibrateInfo(const std::string& pointCode);
    std::string getMethodInfo(const std::string& pointCode);
    std::string getDefectInfo(const std::string& pointCode);
    void updateOrInsertCalibrateInfo(const std::string& method, const std::string& pointCode, const std::string& calibrateJson);
    int getUnitIdByMethod(const std::string& method);

private:
    ConnectionPool& pool_;
};
std::string DfCalibrateInfo::getDefectInfo(const std::string& unitAnalyzeType)
{
   std::string result="";
    auto con = pool_.getConnection();
    try {
        // 查询 unit_custom_anaylze_type 字段
        auto pstmt = con->prepareStatement("SELECT unit_custom_anaylze_type FROM df_analyze_type WHERE unit_anaylze_type = ?");
        pstmt->setString(1, unitAnalyzeType);
        std::cout<<"unit_custom_anaylze_type ----- "<<unitAnalyzeType<<std::endl;
        auto res = pstmt->executeQuery();
        if (res->next()) {
            result = res->getString("unit_custom_anaylze_type"); // 获取查询结果
        } else {
            std::cerr << "未找到对应的 unit_anaylze_type: " << unitAnalyzeType << std::endl;
        }
    } catch (sql::SQLException& e) {
        std::cerr << "数据库查询失败: " << e.what() << "，错误代码: " << e.getErrorCode() << std::endl;
    }
    pool_.releaseConnection(std::move(con));

    if (result.empty()) {
       std::cerr <<"getDefectInfo 查询失败或数据不存在"<<std::endl;
    }
    return result;
}
std::string DfCalibrateInfo::getCalibrateInfo(const std::string& method, const std::string& pointCode) {
    std::string result="";
    auto con = pool_.getConnection();
    try {
        auto pstmt = con->prepareStatement("SELECT calibrate_json FROM df_calibrate_info WHERE method = ? AND point_code = ?");
        pstmt->setString(1, method);
        pstmt->setString(2, pointCode);

        auto res = pstmt->executeQuery();
        if (res->next()) {
            result = res->getString("calibrate_json");
        }
    } catch (sql::SQLException& e) {
        std::cerr << "数据库查询标定信息失败: " << e.what() << std::endl;
    }

    pool_.releaseConnection(std::move(con));
    return result;
}
std::string DfCalibrateInfo::getCalibrateInfo(const std::string& pointCode)
{
    std::string result="";
    auto con = pool_.getConnection();
    try {
        auto pstmt = con->prepareStatement("SELECT calibrate_json FROM df_calibrate_info WHERE point_code = ?");
        pstmt->setString(1, pointCode);
        auto res = pstmt->executeQuery();
        if (res->next()) {
            result = res->getString("calibrate_json");
        }
    } catch (sql::SQLException& e) {
        std::cerr << "数据库查询标定方法失败: " << e.what() << std::endl;
    }
    pool_.releaseConnection(std::move(con));
    return result;
}


std::string DfCalibrateInfo::getMethodInfo(const std::string& pointCode)
{
    std::string result="";
    auto con = pool_.getConnection();
    try {
        auto pstmt = con->prepareStatement("SELECT method FROM df_calibrate_info WHERE point_code = ?");
        pstmt->setString(1, pointCode);
        auto res = pstmt->executeQuery();
        if (res->next()) {
            result = res->getString("method");
        }
    } catch (sql::SQLException& e) {
        std::cerr << "数据库查询标定方法失败: " << e.what() << std::endl;
    }
    pool_.releaseConnection(std::move(con));
    return result;
}

void DfCalibrateInfo::updateOrInsertCalibrateInfo(const std::string& method, const std::string& pointCode, const std::string& calibrateJson) {
    auto con = pool_.getConnection();

    try {
        std::string checkQuery = "SELECT COUNT(*) FROM df_calibrate_info WHERE point_code = ?";
        std::string updateQuery = "UPDATE df_calibrate_info SET method = ?, calibrate_json = ? WHERE point_code = ?";
        std::string insertQuery = "INSERT INTO df_calibrate_info (method, point_code, calibrate_json) VALUES (?, ?, ?)";

        auto pstmtCheck = con->prepareStatement(checkQuery);
        pstmtCheck->setString(1, pointCode);
        auto resCheck = pstmtCheck->executeQuery();

        int count = 0;
        if (resCheck->next()) {
            count = resCheck->getInt(1);
        }

        if (count > 0) {
            std::cout << "标定数据已存在，进行更新操作" << std::endl;
            auto pstmtUpdate = con->prepareStatement(updateQuery);
            pstmtUpdate->setString(1, method);
            pstmtUpdate->setString(2, calibrateJson);
            pstmtUpdate->setString(3, pointCode);
            pstmtUpdate->executeUpdate();
        } else {
            std::cout << "标定数据不存在，进行插入操作" << std::endl;
            auto pstmtInsert = con->prepareStatement(insertQuery);
            pstmtInsert->setString(1, method);
            pstmtInsert->setString(2, pointCode);
            pstmtInsert->setString(3, calibrateJson);
            pstmtInsert->executeUpdate();
        }
    } catch (sql::SQLException& e) {
        std::cerr << "SQL 错误: " << e.what() << std::endl;
    }

    pool_.releaseConnection(std::move(con));
}

int DfCalibrateInfo::getUnitIdByMethod(const std::string& method) {
    int unitId = 0;
    auto con = pool_.getConnection();
    try {
        auto pstmt = con->prepareStatement(
            "SELECT unit_id FROM df_unit_server "
            "WHERE FIND_IN_SET(?, unit_capabilityset) AND unit_status = 1 "
            "GROUP BY unit_id ORDER BY unit_id ASC LIMIT 1"
        );
        pstmt->setString(1, method);

        auto res = pstmt->executeQuery();
        if (res->next()) {
            unitId = res->getInt("unit_id");
        }
    } catch (sql::SQLException& e) {
        std::cerr << "SQL error: " << e.what() << std::endl;
    }

    pool_.releaseConnection(std::move(con));
    return unitId;
}
