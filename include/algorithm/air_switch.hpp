#include "NvInferPlugin.h"
#include "common.hpp"
#include <fstream>
#include <opencv2/opencv.hpp>
#include "cuda_runtime.h"
#include <iostream>
#include <fstream>
#include <cassert>
#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include <vector>
#include <algorithm>
#include <numeric>  // Include this for std::accumulate

class air_switch {
public:
    air_switch(const std::string& enginePath, cv::Size inputSize);
    std::vector<std::string> classNames = {
        "kqkg_off", "kqkg_on"
    };

    ~air_switch();

    void initialize();
    void preprocess(cv::Mat& src, cv::Mat& dst);
    void blobFromImage(cv::Mat& img);
    int doInference();
    void generate_proposals(std::vector<Object>& objects, float confThresh);
    std::pair<int, float> argmax(std::vector<float>& vSingleProbs);
    void qsort_descent_inplace(std::vector<Object>& objects);
    void nms_sorted_bboxes(const std::vector<Object>& vObjects, std::vector<int>& picked, float nms_threshold);
    std::vector<Object> decodeOutputs(std::vector<Object>& objects);
    void drawAndSaveResults(cv::Mat& img, const std::vector<Object>& objects, std::string& pos, float& conf, std::string& desc);

private:
    void qsort_descent_inplace(std::vector<Object>& objects, int left, int right);
    float intersection_area(const Object& a, const Object& b);

    std::string mEnginePath;
    nvinfer1::IRuntime* mRuntime{nullptr};
    nvinfer1::ICudaEngine* mEngine{nullptr};
    nvinfer1::IExecutionContext* mContext{nullptr};
    Logger mGLogger;  // Use the custom Logger class
    
    int mInputIndex;
    int mOutputIndex;
    float* mBlob{nullptr};
    float* mProb{nullptr};
    void* mBuffers[2];
    cudaStream_t mStream;

    size_t mInputSize;
    size_t mOutputSize;
    cv::Size mCvInputSize;
    cv::Size mCvOriginSize;

    // Utility function to check CUDA status
    void checkStatus(cudaError_t status) {
        if (status != cudaSuccess) {
            std::cerr << "CUDA Error: " << cudaGetErrorString(status) << std::endl;
            exit(EXIT_FAILURE);
        }
    }
};

air_switch::air_switch(const std::string& enginePath, cv::Size inputSize)
    : mEnginePath(enginePath), mCvInputSize(inputSize) {
    mInputSize = inputSize.width * inputSize.height * 3; // Assuming 3 channels (RGB)
    mBlob = new float[mInputSize];
}

air_switch::~air_switch() {
    delete[] mBlob;
    delete[] mProb;
    cudaFree(mBuffers[mInputIndex]);
    cudaFree(mBuffers[mOutputIndex]);
    cudaStreamDestroy(mStream);
    mContext->destroy();
    mEngine->destroy();
    mRuntime->destroy();
}
//模型与加载模型
void air_switch::initialize() {
    //cudaSetDevice(0);
    char *trtModelStream{nullptr};
    size_t size{0};

    std::ifstream file(mEnginePath, std::ios::binary);
    std::cout << "[I] Detection model creating...\n";
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }

    mRuntime = nvinfer1::createInferRuntime(mGLogger);
    assert(mRuntime != nullptr);

    std::cout << "[I] Detection engine creating...\n";
    mEngine = mRuntime->deserializeCudaEngine(trtModelStream, size);
    assert(mEngine != nullptr);
    mContext = mEngine->createExecutionContext();
    assert(mContext != nullptr);
    delete[] trtModelStream;

    auto out_dims = mEngine->getBindingDimensions(1);

    mOutputSize = std::accumulate(out_dims.d, out_dims.d + out_dims.nbDims, 1, std::multiplies<int>());
    mProb = new float[mOutputSize];

    assert(mEngine->getNbBindings() == 2);
    std::cout << "[I] Cuda buffer creating...\n";

    mInputIndex = mEngine->getBindingIndex("images");
    assert(mEngine->getBindingDataType(mInputIndex) == nvinfer1::DataType::kFLOAT);
    mOutputIndex = mEngine->getBindingIndex("output");
    assert(mEngine->getBindingDataType(mOutputIndex) == nvinfer1::DataType::kFLOAT);

    checkStatus(cudaMalloc(&mBuffers[mInputIndex], mInputSize * sizeof(float)));
    checkStatus(cudaMalloc(&mBuffers[mOutputIndex], mOutputSize * sizeof(float)));

    std::cout << "[I] Cuda stream creating...\n";
    checkStatus(cudaStreamCreate(&mStream));

    std::cout << "[I] Detection engine created!\n";
}

void air_switch::preprocess(cv::Mat& src, cv::Mat& dst) {
    mCvOriginSize = src.size();  // 保存原始尺寸
    dst = src.clone();
    cv::cvtColor(dst, dst, cv::COLOR_BGR2RGB);  // 转换为RGB
    cv::resize(dst, dst, mCvInputSize);  // 调整大小到模型输入尺寸
    dst.convertTo(dst, CV_32FC3);  // 转换为32位浮点数
    dst = dst / 255.0f;  // 归一化到[0, 1]
}

void air_switch::blobFromImage(cv::Mat& img) {
    preprocess(img, img);
    int channels = img.channels();
    int cols = img.cols;
    int rows = img.rows;

    for (int c = 0; c < channels; c++) {
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                mBlob[c * rows * cols + row * cols + col] = img.at<cv::Vec3f>(row, col)[c]; 
            }
        }
    }
}

int air_switch::doInference() {
    checkStatus(cudaMemcpyAsync(mBuffers[mInputIndex], mBlob, mInputSize * sizeof(float), cudaMemcpyHostToDevice, mStream));
    mContext->enqueueV2(mBuffers, mStream, nullptr);
    checkStatus(cudaMemcpyAsync(mProb, mBuffers[mOutputIndex], mOutputSize * sizeof(float), cudaMemcpyDeviceToHost, mStream));
    cudaStreamSynchronize(mStream);

    return 0;
}

// Other methods like generate_proposals, argmax, qsort_descent_inplace, and drawAndSaveResults remain unchanged
std::pair<int, float> air_switch::argmax(std::vector<float>& vSingleProbs) {
    std::pair<int, float> result;
    auto iter = std::max_element(vSingleProbs.begin(), vSingleProbs.end());
    result.first = static_cast<int>(iter - vSingleProbs.begin());
    result.second = *iter;

    return result;
}

void air_switch::generate_proposals(std::vector<Object>& objects, float confThresh) {
    int nc = 2;  // 类别数量
    for (int i = 0; i < 25200; i++) {
        float prob = mProb[i * (nc + 5) + 4];  // 获取置信度
        if (prob > confThresh) {
           
            Object obj;
            float cx = mProb[i * (nc + 5)];
            float cy = mProb[i * (nc + 5) + 1];
            float w  = mProb[i * (nc + 5) + 2];
            float h  = mProb[i * (nc + 5) + 3];
            obj.rect.x = static_cast<int>(cx - w * 0.5f);
            obj.rect.y = static_cast<int>(cy - h * 0.5f);
            obj.rect.width = static_cast<int>(w);
            obj.rect.height = static_cast<int>(h);

            std::vector<float> vSingleProbs(nc);
            for (int j = 0; j < vSingleProbs.size(); j++) {
                vSingleProbs[j]= mProb[i * (nc+5) + 5 + j];
            }

            auto max = argmax(vSingleProbs);
            obj.label = max.first;  // 设置类别索引
            obj.prob = prob;

   

            objects.push_back(obj);
        }
    }
}

void air_switch::qsort_descent_inplace(std::vector<Object>& objects, int left, int right) {
    int i = left;
    int j = right;
    float p = objects[(left + right) / 2].prob;

    while (i <= j) {
        while (objects[i].prob > p)
            i++;

        while (objects[j].prob < p)
            j--;

        if (i <= j) {
            std::swap(objects[i], objects[j]);
            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(objects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(objects, i, right);
        }
    }
}

void air_switch::qsort_descent_inplace(std::vector<Object>& objects) {
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

float air_switch::intersection_area(const Object& a, const Object& b) {
    cv::Rect intersection = a.rect & b.rect;
    return intersection.area();
}

void air_switch::nms_sorted_bboxes(const std::vector<Object>& vObjects, std::vector<int>& picked, float nms_threshold) {
    picked.clear();

    const int n = vObjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) {
        areas[i] = vObjects[i].rect.area();
    }

    for (int i = 0; i < n; i++) {
        const Object& a = vObjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++) {
            const Object& b = vObjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

#include <map>

std::vector<Object> air_switch::decodeOutputs(std::vector<Object>& objects) {
    generate_proposals(objects, 0.001f);  // 阈值可以根据需求调整
    qsort_descent_inplace(objects);

    std::vector<int> picked;
    nms_sorted_bboxes(objects, picked, 0.45f);  // 阈值可以根据需求调整

    int count = picked.size();

    int img_w = mCvOriginSize.width;
    int img_h = mCvOriginSize.height;
    float scaleH = static_cast<float>(mCvInputSize.height) / static_cast<float>(img_h);
    float scaleW = static_cast<float>(mCvInputSize.width) / static_cast<float>(img_w);

    Object bestObject;
    float maxProbability = 0.0f;

    for (int i = 0; i < count; i++) {
        Object obj = objects[picked[i]];

        // 恢复到原始图像尺寸的偏移
        float x0 = static_cast<float>(obj.rect.x) / scaleW;
        float y0 = static_cast<float>(obj.rect.y) / scaleH;
        float x1 = static_cast<float>(obj.rect.x + obj.rect.width) / scaleW;
        float y1 = static_cast<float>(obj.rect.y + obj.rect.height) / scaleH;

        // 裁剪
        x0 = std::max(std::min(x0, static_cast<float>(img_w - 1)), 0.0f);
        y0 = std::max(std::min(y0, static_cast<float>(img_h - 1)), 0.0f);
        x1 = std::max(std::min(x1, static_cast<float>(img_w - 1)), 0.0f);
        y1 = std::max(std::min(y1, static_cast<float>(img_h - 1)), 0.0f);

        obj.rect.x = static_cast<int>(x0);
        obj.rect.y = static_cast<int>(y0);
        obj.rect.width = static_cast<int>(x1 - x0);
        obj.rect.height = static_cast<int>(y1 - y0);

        // 检查是否为概率最高的检测框
        if (obj.prob > maxProbability) {
            maxProbability = obj.prob;
            bestObject = obj;
        }
    }

    // 只返回概率最高的检测结果
    std::vector<Object> results;
    if (maxProbability > 0.0f) {
        results.push_back(bestObject);
    }

    return results;
}

void air_switch::drawAndSaveResults(cv::Mat& img, const std::vector<Object>& objects,std::string& pos,float& conf,std::string& desc) {
    // 将图像数据恢复到 8 位整数格式并缩放回 [0, 255]
    img.convertTo(img, CV_8UC3, 255.0);

    // 恢复 img 为原始输入尺寸
    cv::resize(img, img, mCvOriginSize);
    // Json::Value areas(Json::arrayValue); // 用于存储坐标的 JSON 数组
    Json::Value posWrapper(Json::arrayValue);  // 用于存储坐标的 JSON 包装数组
    for (const auto& obj : objects) {
        // 获取类别名称
        std::string className = classNames[obj.label];
         Json::Value areas(Json::arrayValue);  // 单独存储坐标的 JSON 数组
        // 打印类别和概率到控制台
        std::cout << "Detected object: " << className << ", Probability: " << obj.prob << std::endl;
        
        // 画边框
        cv::rectangle(img, obj.rect, cv::Scalar(0, 255, 0), 2);

        // 画标签和概率
        std::string label = className + ": " + std::to_string(obj.prob);
        int baseLine = 0;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        cv::rectangle(img, cv::Rect(cv::Point(obj.rect.x, obj.rect.y - labelSize.height),
                      cv::Size(labelSize.width, labelSize.height + baseLine)),
                      cv::Scalar(255, 255, 255), cv::FILLED);
        cv::putText(img, label, cv::Point(obj.rect.x, obj.rect.y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
        // 计算边框的四个点坐标
        int x_min = obj.rect.x;
        int y_min = obj.rect.y;
        int x_max = obj.rect.x + obj.rect.width;
        int y_max = obj.rect.y + obj.rect.height;

        // 将四个点存储到 JSON
        Json::Value top_left, top_right, bottom_right, bottom_left;
        top_left["x"] = x_min;
        top_left["y"] = y_min;
        top_right["x"] = x_max;
        top_right["y"] = y_min;
        bottom_right["x"] = x_max;
        bottom_right["y"] = y_max;
        bottom_left["x"] = x_min;
        bottom_left["y"] = y_max;

        // 将每个点依次添加到 JSON 数组//只保留左上和右下
        areas.append(top_left);
      //  areas.append(top_right);
        areas.append(bottom_right);
        //areas.append(bottom_left);
        conf = obj.prob;
        desc.append(className);

          // 将这些坐标包裹在 "areas" 下
        Json::Value areaObject;
       
      
        areaObject["areas"] = areas;
            // 将 areas 加入 posWrapper 中
        posWrapper.append(areaObject);
    }

    // 打印边框坐标 JSON
   //std::cout << "Detected areas: " << areas.toStyledString() << std::endl;
  //  pos = areas.toStyledString() ;

// 生成并传输紧凑的JSON字符串
Json::StreamWriterBuilder writerBuilder;
writerBuilder["indentation"] = "";  // 移除多余的空格和换行符
pos = Json::writeString(writerBuilder, posWrapper);
std::cout << "Detected areas: " << pos << std::endl;

    // 将图像从 RGB 转回 BGR
    cv::cvtColor(img, img, cv::COLOR_RGB2BGR);

    // 保存图像
  
}