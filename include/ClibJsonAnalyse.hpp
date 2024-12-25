#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <json.h>
#include <global.h>
class ClibJsonAnalyse{
public:
    ClibJsonAnalyse();
    ~ClibJsonAnalyse();
    void ClibJsonParse(Json::Value& jsonValue,struct AlgJsonObject &obj);
};
ClibJsonAnalyse::ClibJsonAnalyse(){}
ClibJsonAnalyse::~ClibJsonAnalyse(){}
void ClibJsonAnalyse::ClibJsonParse(Json::Value& jsonValue, AlgJsonObject &obj) {
    if (jsonValue.isMember("method")) {
        obj.opertype = jsonValue["method"].asString();
    }
    if (jsonValue.isMember("angle")) {
        obj.angle = jsonValue["angle"].asDouble();
    }
    if (jsonValue.isMember("is_Integer")) {
        obj.is_Integer = jsonValue["is_Integer"].asInt(); 
    }
    if (jsonValue.isMember("FHZSP_type")) {
        obj.FHZSP_type = jsonValue["FHZSP_type"].asInt();
    }
    if (jsonValue.isMember("params")) {
        const Json::Value& params = jsonValue["params"];
        if (params.isMember("area")) {
            const Json::Value& area = params["area"];
            if (area.isObject()) {
                const Json::Value& point1_obj = area["point1"];
                const Json::Value& point2_obj = area["point2"];

                int x1 = point1_obj["x"].asInt();
                int y1 = point1_obj["y"].asInt();
                int x2 = point2_obj["x"].asInt();
                int y2 = point2_obj["y"].asInt();

                int w = x2 - x1;
                int h = y2 - y1;

                cv::Rect rect;
                if (w > 0) {
                    rect = cv::Rect(x1, y1, w, h);
                } else {
                    rect = cv::Rect(x2, y2, -w, -h);
                }
                obj.area_rect.push_back(rect);

                rect_name r_n;
                r_n.rect = rect;
                r_n.name = "";
                obj.area_rect_name.push_back(r_n);
            } else if (area.isArray()) {
                for (const auto& area_obj : area) {
                    rect_name r_n;
                    if (area_obj.isMember("point3")) {
                        r_n.name = area_obj["name"].asString();

                        // Handle rect points
                        for (int i = 1;; ++i) {
                            std::string point_key = "point" + std::to_string(i);
                            if (area_obj.isMember(point_key)) {
                                const Json::Value& point_obj = area_obj[point_key];
                                int x = point_obj["x"].asInt();
                                int y = point_obj["y"].asInt();
                                r_n.rect_points.emplace_back(cv::Point(x, y));
                            } else {
                                break;
                            }
                        }

                        obj.area_rect_name.push_back(r_n);
                    } else {
                        const Json::Value& point1_obj = area_obj["point1"];
                        const Json::Value& point2_obj = area_obj["point2"];

                        int x1 = point1_obj["x"].asInt();
                        int y1 = point1_obj["y"].asInt();
                        int x2 = point2_obj["x"].asInt();
                        int y2 = point2_obj["y"].asInt();

                        int w = x2 - x1;
                        int h = y2 - y1;

                        cv::Rect rect;
                        if (w > 0) {
                            rect = cv::Rect(x1, y1, w, h);
                        } else {
                            rect = cv::Rect(x2, y2, -w, -h);
                        }
                        obj.area_rect.push_back(rect);
                        r_n.rect = rect;
                        r_n.name = area_obj["name"].asString();
                        obj.area_rect_name.push_back(r_n);
                    }
                }
            }
        }
    }
    if (jsonValue.isMember("point_code")) {
        obj.point_code = jsonValue["point_code"].asString();
    }
}
