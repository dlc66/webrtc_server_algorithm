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



class EleNumber_bp {
public:
    EleNumber_bp(const std::string& enginePath, cv::Size inputSize);
    std::vector<std::string> classNames = {
        "szyb","zzyb_r","szyb_c"
    };
    ~EleNumber_bp();

    void initialize();
    void preprocess(cv::Mat& src, cv::Mat& dst);
    void blobFromImage(cv::Mat& img);
    int  doInference();
    void generate_proposals(std::vector<Object>& objects, float confThresh);
    std::pair<int, float> argmax(std::vector<float>& vSingleProbs);
    void qsort_descent_inplace(std::vector<Object>& objects);
    void nms_sorted_bboxes(const std::vector<Object>& vObjects, std::vector<int>& picked, float nms_threshold);
    std::vector<Object> decodeOutputs(std::vector<Object>& objects);
    //void drawAndSaveResults(cv::Mat& img, const std::vector<Object>& objects);
    void cropAndSaveResults(cv::Mat& img, const std::vector<Object>& objects);
    
    

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

EleNumber_bp::EleNumber_bp(const std::string& enginePath, cv::Size inputSize)
    : mEnginePath(enginePath), mCvInputSize(inputSize) {
    mInputSize = inputSize.width * inputSize.height * 3; // Assuming 3 channels (RGB)
    mBlob = new float[mInputSize];
}

EleNumber_bp::~EleNumber_bp() {
    delete[] mBlob;
    delete[] mProb;
    cudaFree(mBuffers[mInputIndex]);
    cudaFree(mBuffers[mOutputIndex]);
    cudaStreamDestroy(mStream);
    mContext->destroy();
    mEngine->destroy();
    mRuntime->destroy();
}

void EleNumber_bp::initialize() {
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

void EleNumber_bp::preprocess(cv::Mat& src, cv::Mat& dst) {
    mCvOriginSize = src.size();  // 保存原始尺寸
    dst = src.clone();
    cv::cvtColor(dst, dst, cv::COLOR_BGR2RGB);  // 转换为RGB
    cv::resize(dst, dst, mCvInputSize);  // 调整大小到模型输入尺寸
    dst.convertTo(dst, CV_32FC3);  // 转换为32位浮点数
    dst = dst / 255.0f;  // 归一化到[0, 1]
}

void EleNumber_bp::blobFromImage(cv::Mat& img) {
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

int EleNumber_bp::doInference() {
    checkStatus(cudaMemcpyAsync(mBuffers[mInputIndex], mBlob, mInputSize * sizeof(float), cudaMemcpyHostToDevice, mStream));
    mContext->enqueueV2(mBuffers, mStream, nullptr);
    checkStatus(cudaMemcpyAsync(mProb, mBuffers[mOutputIndex], mOutputSize * sizeof(float), cudaMemcpyDeviceToHost, mStream));
    cudaStreamSynchronize(mStream);

    return 0;
}

std::pair<int, float> EleNumber_bp::argmax(std::vector<float>& vSingleProbs) {
    std::pair<int, float> result;
    auto iter = std::max_element(vSingleProbs.begin(), vSingleProbs.end());
    result.first = static_cast<int>(iter - vSingleProbs.begin());
    result.second = *iter;

    return result;
}

void EleNumber_bp::generate_proposals(std::vector<Object>& objects, float confThresh) {
    int nc = 3;  // 类别数量
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
}//生成检测框

void EleNumber_bp::qsort_descent_inplace(std::vector<Object>& objects, int left, int right) {
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
}//检测框按照置信度排列

void EleNumber_bp::qsort_descent_inplace(std::vector<Object>& objects) {
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

float EleNumber_bp::intersection_area(const Object& a, const Object& b) {
    cv::Rect intersection = a.rect & b.rect;
    return intersection.area();
}

void EleNumber_bp::nms_sorted_bboxes(const std::vector<Object>& vObjects, std::vector<int>& picked, float nms_threshold) {
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

            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}//非极大值抑制

std::vector<Object> EleNumber_bp::decodeOutputs(std::vector<Object>& objects) {
    generate_proposals(objects, 0.001f);  // 阈值可以根据需求调整 生成候选框
    qsort_descent_inplace(objects);//置信度排序

    std::vector<int> picked;
    nms_sorted_bboxes(objects, picked, 0.45f);  //非极大值抑制,取出重叠框 阈值可以根据需求调整

    int count = picked.size();//抑制后候选框数组
    // 4. 计算原图和处理图的尺寸比例，用于还原坐标
    int img_w = mCvOriginSize.width;
    int img_h = mCvOriginSize.height;//原图尺寸
    float scaleH = static_cast<float>(mCvInputSize.height) / static_cast<float>(img_h);
    float scaleW = static_cast<float>(mCvInputSize.width) / static_cast<float>(img_w);

    Object bestObject;
    float maxConfidence = 0.0f;
    //寻找最大置信度框
    for (int i = 0; i < count; i++) {
        Object obj = objects[picked[i]];
        // 还原候选框在原图中的坐标
        float x0 = static_cast<float>(obj.rect.x) / scaleW;
        float y0 = static_cast<float>(obj.rect.y) / scaleH;
        float x1 = static_cast<float>(obj.rect.x + obj.rect.width) / scaleW;
        float y1 = static_cast<float>(obj.rect.y + obj.rect.height) / scaleH;

        // x0 = std::max(std::min(x0, static_cast<float>(img_w - 1)), 0.0f);
        // y0 = std::max(std::min(y0, static_cast<float>(img_h - 1)), 0.0f);
        // x1 = std::max(std::min(x1, static_cast<float>(img_w - 1)), 0.0f);
        // y1 = std::max(std::min(y1, static_cast<float>(img_h - 1)), 0.0f);
        //确保坐标在边界
        x0 = std::clamp(x0, 0.0f, static_cast<float>(img_w - 1));
        y0 = std::clamp(y0, 0.0f, static_cast<float>(img_h - 1));
        x1 = std::clamp(x1, 0.0f, static_cast<float>(img_w - 1));
        y1 = std::clamp(y1, 0.0f, static_cast<float>(img_h - 1));
        
        // 更新候选框的坐标和尺寸
        obj.rect.x = static_cast<int>(x0);
        obj.rect.y = static_cast<int>(y0);
        obj.rect.width = static_cast<int>(x1 - x0);
        obj.rect.height = static_cast<int>(y1 - y0);
        //取最值
        if (obj.conf > maxConfidence) {
            maxConfidence = obj.conf;
            bestObject = obj;
        }
     }
 
   
    std::vector<Object> results;
    if (maxConfidence > 0.0f) {
        results.push_back(bestObject);
    }

    return results;
}

// void EleNumber_bp::drawAndSaveResults(cv::Mat& img, const std::vector<Object>& objects) {
//     img.convertTo(img, CV_8UC3, 255.0);
//     cv::resize(img, img, mCvOriginSize);

//     for (const auto& obj : objects) {
//         std::string className = classNames[obj.label];
        
//         std::cout << "Detected object: " << className << ", Confidence: " << obj.conf << std::endl;
        
//         cv::rectangle(img, obj.rect, cv::Scalar(0, 255, 0), 2);

//         std::string label = className + ": " + std::to_string(obj.conf);
//         int baseLine = 0;
//         cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
//         cv::rectangle(img, cv::Rect(cv::Point(obj.rect.x, obj.rect.y - labelSize.height),
//                       cv::Size(labelSize.width, labelSize.height + baseLine)),
//                       cv::Scalar(255, 255, 255), cv::FILLED);
//         cv::putText(img, label, cv::Point(obj.rect.x, obj.rect.y),
//                     cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
//     }

//     cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
   
// }

void EleNumber_bp::cropAndSaveResults( cv::Mat& img, const std::vector<Object>& objects) {
    for (size_t i = 0; i < objects.size(); ++i) {
        const auto& obj = objects[i];
        std::string className = classNames[obj.label];
        std::cout << "Detected object: " << className << ", Confidence: " << obj.conf << std::endl;
        std::cout<< "obj.rect.x: "<<obj.rect.x<<"obj.rect.y: "<<obj.rect.y<<"obj.rect.width: "<<obj.rect.width<<"obj.rect.height: "<<obj.rect.height<<std::endl;
        // 检查候选框是否在图像的有效范围内
        if (obj.rect.x >= 0 && obj.rect.y >= 0 &&
            obj.rect.x + obj.rect.width <= img.cols &&
            obj.rect.y + obj.rect.height <= img.rows) {
              // 如果候选框有效，则从图像中裁剪出对应区域
            cv::Mat cropped_image = img(obj.rect);
            img = cropped_image;

        } else {
            std::cerr << "Invalid bounding box for cropping!" << std::endl;
        }
    }
}
