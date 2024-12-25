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
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <numeric>  // Include this for std::accumulate



class EleNumber_num {
public:
    EleNumber_num(const std::string& enginePath, cv::Size inputSize);
    std::vector<std::string> classNames = {
       "-", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0.", "1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9."
    };

    ~EleNumber_num();

    void initialize();
    void preprocess(cv::Mat& src, cv::Mat& dst);
    void ReOriImageSize(cv::Mat& img,int cols,int rows);
    void blobFromImage(cv::Mat& img);
    int doInference();
    void generate_proposals(std::vector<Object>& objects, float confThresh);
    std::pair<int, float> argmax(std::vector<float>& vSingleProbs);
    void qsort_descent_inplace(std::vector<Object>& objects);
    void nms_sorted_bboxes(const std::vector<Object>& vObjects, std::vector<int>& picked, float nms_threshold);
    std::vector<Object> decodeOutputs(std::vector<Object>& objects);
    //void drawAndSaveResults(cv::Mat& img, const std::vector<Object>& objects);
    void drawAndSaveResults(cv::Mat& img, const std::vector<Object>& objects_num,const std::vector<Object>& objects_bp);
    void drawAndSaveResults(cv::Mat& img, const std::vector<Object>& objects_num);
    //void cropAndSaveResults(const cv::Mat& img, const std::vector<Object>& objects);

    
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
    cv::Size mCvOriginSize;//模型推理图大小
    

    // Utility function to check CUDA status
    void checkStatus(cudaError_t status) {
        if (status != cudaSuccess) {
            std::cerr << "CUDA Error: " << cudaGetErrorString(status) << std::endl;
            exit(EXIT_FAILURE);
        }
    }
};

EleNumber_num::EleNumber_num(const std::string& enginePath, cv::Size inputSize)
    : mEnginePath(enginePath), mCvInputSize(inputSize) {
    mInputSize = inputSize.width * inputSize.height * 3; // Assuming 3 channels (RGB)
    mBlob = new float[mInputSize];
}

EleNumber_num::~EleNumber_num() {
    delete[] mBlob;
    delete[] mProb;
    cudaFree(mBuffers[mInputIndex]);
    cudaFree(mBuffers[mOutputIndex]);
    cudaStreamDestroy(mStream);
    mContext->destroy();
    mEngine->destroy();
    mRuntime->destroy();
}

void EleNumber_num::initialize() {
    cudaSetDevice(0);
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
void EleNumber_num::ReOriImageSize(cv::Mat& img,int cols,int rows){
     if (img.empty()) {
        std::cerr << "恢复原大小图像为空！" << std::endl;
        return;
    }
    // 恢复图像大小
    cv::resize(img, img, cv::Size(cols, rows));
    std::cout << "图像恢复原大小: " << cols << " x " << rows << std::endl;
}
void EleNumber_num::preprocess(cv::Mat& src, cv::Mat& dst) {
    mCvOriginSize = src.size();  // 保存原始尺寸
    dst = src.clone();
    cv::cvtColor(dst, dst, cv::COLOR_BGR2RGB);  // 转换为RGB
    cv::resize(dst, dst, mCvInputSize);  // 调整大小到模型输入尺寸
    dst.convertTo(dst, CV_32FC3);  // 转换为32位浮点数
    dst = dst / 255.0f;  // 归一化到[0, 1]
}

void EleNumber_num::blobFromImage(cv::Mat& img) {
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

int EleNumber_num::doInference() {
    checkStatus(cudaMemcpyAsync(mBuffers[mInputIndex], mBlob, mInputSize * sizeof(float), cudaMemcpyHostToDevice, mStream));
    mContext->enqueueV2(mBuffers, mStream, nullptr);
    checkStatus(cudaMemcpyAsync(mProb, mBuffers[mOutputIndex], mOutputSize * sizeof(float), cudaMemcpyDeviceToHost, mStream));
    cudaStreamSynchronize(mStream);

    return 0;
}

std::pair<int, float> EleNumber_num::argmax(std::vector<float>& vSingleProbs) {
    std::pair<int, float> result;
    auto iter = std::max_element(vSingleProbs.begin(), vSingleProbs.end());
    result.first = static_cast<int>(iter - vSingleProbs.begin());
    result.second = *iter;

    return result;
}

void EleNumber_num::generate_proposals(std::vector<Object>& objects, float confThresh) {
    int nc = 21;  // 类别数量
    for (int i = 0; i < 25200; i++) {
        float conf = mProb[i * (nc + 5) + 4];  // 获取置信度
        if (conf > confThresh) {
           
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
                vSingleProbs[j] = mProb[i * (nc+5) + 5 + j];
            }

            auto max = argmax(vSingleProbs);
            obj.label = max.first;  // 设置类别索引
            obj.conf = conf;

            objects.push_back(obj);
        }
    }
}

void EleNumber_num::qsort_descent_inplace(std::vector<Object>& objects, int left, int right) {
    int i = left;
    int j = right;
    float p = objects[(left + right) / 2].conf;

    while (i <= j) {
        while (objects[i].conf > p)
            i++;

        while (objects[j].conf < p)
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

void EleNumber_num::qsort_descent_inplace(std::vector<Object>& objects) {
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

float EleNumber_num::intersection_area(const Object& a, const Object& b) {
    cv::Rect intersection = a.rect & b.rect;
    return intersection.area();
}

void EleNumber_num::nms_sorted_bboxes(const std::vector<Object>& vObjects, std::vector<int>& picked, float nms_threshold) {
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

std::vector<Object> EleNumber_num::decodeOutputs(std::vector<Object>& objects) {
    generate_proposals(objects, 0.5f);  // 阈值可以根据需求调整
    qsort_descent_inplace(objects);

    std::vector<int> picked;
    nms_sorted_bboxes(objects, picked, 0.45f);  // NMS 阈值可以根据需求调整

    int count = picked.size();  // 获取通过 NMS 的对象数量

    int img_w = mCvOriginSize.width;
    int img_h = mCvOriginSize.height;
    float scaleH = static_cast<float>(mCvInputSize.height) / static_cast<float>(img_h);
    float scaleW = static_cast<float>(mCvInputSize.width) / static_cast<float>(img_w);

    std::vector<Object> results;

    for (int i = 0; i < count; i++) {
        Object obj = objects[picked[i]];

        float x0 = static_cast<float>(obj.rect.x) / scaleW;
        float y0 = static_cast<float>(obj.rect.y) / scaleH;
        float x1 = static_cast<float>(obj.rect.x + obj.rect.width) / scaleW;
        float y1 = static_cast<float>(obj.rect.y + obj.rect.height) / scaleH;

        x0 = std::max(std::min(x0, static_cast<float>(img_w - 1)), 0.0f);
        y0 = std::max(std::min(y0, static_cast<float>(img_h - 1)), 0.0f);
        x1 = std::max(std::min(x1, static_cast<float>(img_w - 1)), 0.0f);
        y1 = std::max(std::min(y1, static_cast<float>(img_h - 1)), 0.0f);

        obj.rect.x = static_cast<int>(x0);
        obj.rect.y = static_cast<int>(y0);
        obj.rect.width = static_cast<int>(x1 - x0);
        obj.rect.height = static_cast<int>(y1 - y0);

        results.push_back(obj);
    }

    return results;
}

void EleNumber_num::drawAndSaveResults(cv::Mat& img, const std::vector<Object>& objects_num,const std::vector<Object>& objects_bp) {
    //img.convertTo(img, CV_8UC3, 255.0);
    //cv::resize(img, img, mCvOriginSize);

    int x_min = img.cols, y_min = img.rows;
    int x_max = 0, y_max = 0;
    float total_confidence = 0.0f;
    int num_objects = 0;
    std::string detected_numbers;

    // 创建一个临时的对象向量并按照 `rect.x` 进行排序
    std::vector<Object> sorted_objects = objects_num;
    std::sort(sorted_objects.begin(), sorted_objects.end(), [](const Object& a, const Object& b) {
        return a.rect.x < b.rect.x; // 按照 x 坐标从小到大排序
    });

    // 遍历排序后的对象
    for (const auto& obj : sorted_objects) {
        std::string className = classNames[obj.label];

        x_min = std::min(x_min, static_cast<int>(obj.rect.x));
        y_min = std::min(y_min, static_cast<int>(obj.rect.y));

        x_max = std::max(x_max, static_cast<int>(obj.rect.x + obj.rect.width));
        y_max = std::max(y_max, static_cast<int>(obj.rect.y + obj.rect.height));

        total_confidence += obj.conf;
        num_objects++;

        if (isdigit(className[0]) && className[0] != '0') {
            detected_numbers += className; // 按照 x 坐标顺序拼接数字
        }

        std::cout << "Detected object: " << className << ", Confidence: " << obj.conf << std::endl;
    }
    float avg_confidence = (num_objects > 0) ? (total_confidence / num_objects) : 0.0f;

    if (detected_numbers.empty()) {
        detected_numbers = "-";
    }
     // 绘制表盘对象的检测框和数字
    std::cout<<"对象长度："<<objects_bp.size()<<std::endl;
    std::string result_label =  detected_numbers;
    for (const auto& bp_obj : objects_bp) {
        std::cout<<bp_obj.rect.x<<","<<bp_obj.rect.y<<std::endl;
        std::string bp_className = classNames[bp_obj.label];
        // 绘制表盘的绿色矩形框
         cv::rectangle(img, bp_obj.rect, cv::Scalar(0, 255, 0), 4); // 绿色框
         cv::Point text_position(bp_obj.rect.x, bp_obj.rect.y - 6); // 在左上角稍微向上偏移
         cv::putText(img, result_label, text_position, cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 0, 255), 2);
    }
  

    std::cout << "Detected numbers: " << detected_numbers << ", Avg Confidence: " << avg_confidence << std::endl;
}


void EleNumber_num::drawAndSaveResults(cv::Mat& img, const std::vector<Object>& objects_num) {
    img.convertTo(img, CV_8UC3, 255.0);
    cv::resize(img, img, mCvOriginSize);

    int x_min = img.cols, y_min = img.rows;
    int x_max = 0, y_max = 0;
    float total_confidence = 0.0f;
    int num_objects = 0;
    std::string detected_numbers;

    // 创建一个临时的对象向量并按照 `rect.x` 进行排序
    std::vector<Object> sorted_objects = objects_num;
    std::sort(sorted_objects.begin(), sorted_objects.end(), [](const Object& a, const Object& b) {
        return a.rect.x < b.rect.x; // 按照 x 坐标从小到大排序
    });

    // 遍历排序后的对象
    for (const auto& obj : sorted_objects) {
        std::string className = classNames[obj.label];

        x_min = std::min(x_min, static_cast<int>(obj.rect.x));
        y_min = std::min(y_min, static_cast<int>(obj.rect.y));

        x_max = std::max(x_max, static_cast<int>(obj.rect.x + obj.rect.width));
        y_max = std::max(y_max, static_cast<int>(obj.rect.y + obj.rect.height));

        total_confidence += obj.conf;
        num_objects++;

        if (isdigit(className[0]) && className[0] != '0') {
            detected_numbers += className; // 按照 x 坐标顺序拼接数字
        }

        std::cout << "Detected object: " << className << ", Confidence: " << obj.conf << std::endl;

        //cv::rectangle(img, obj.rect, cv::Scalar(0, 255, 0), 2);

        std::string label = className + ": " + std::to_string(obj.conf);
        int baseLine = 0;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        // cv::rectangle(img, cv::Rect(cv::Point(obj.rect.x, obj.rect.y - labelSize.height),
        //               cv::Size(labelSize.width, labelSize.height + baseLine)),
        //               cv::Scalar(255, 255, 255), cv::FILLED);
    }

    float avg_confidence = (num_objects > 0) ? (total_confidence / num_objects) : 0.0f;

   // cv::rectangle(img, cv::Rect(cv::Point(x_min, y_min), cv::Point(x_max, y_max)), cv::Scalar(255, 0, 0), 2);
    
    if (detected_numbers.empty()) {
        detected_numbers = "-";
    }
    
    //std::string result_label = "Detected: " + detected_numbers + " Avg Conf: " + std::to_string(avg_confidence);
    std::string result_label =  detected_numbers;
    cv::putText(img, result_label, cv::Point(x_min, y_min - 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);

    cv::cvtColor(img, img, cv::COLOR_RGB2BGR);

    std::cout << "Detected numbers: " << detected_numbers << ", Avg Confidence: " << avg_confidence << std::endl;
}
