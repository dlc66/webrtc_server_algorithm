#include <fstream>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "../common.hpp"
#include "NvInfer.h"
#include "cuda_runtime.h"



class number_class {
public:
    number_class(const std::string& enginePath);
    ~number_class();
    
    void preprocess(const cv::Mat& image, float* inputBuffer);
    int infer(const cv::Mat& inputImage, float& confidence, std::string& label);
    void drawResult(cv::Mat& image, const std::string& label, float confidence);
    void drawResult(cv::Mat& image, const std::string& label, float confidence,cv::Rect &bound_box);//重载

private:
    std::string mEnginePath;
    nvinfer1::IRuntime* mRuntime{nullptr};
    nvinfer1::ICudaEngine* mEngine{nullptr};
    nvinfer1::IExecutionContext* mContext{nullptr};
    

    void* mBuffers[2]; // For input and output buffers
    cudaStream_t mStream;
    int mInputIndex, mOutputIndex;
    size_t mInputSize, mOutputSize;

    std::vector<std::string> labels_name = {"-", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};
};

number_class::number_class(const std::string& enginePath) : mEnginePath(enginePath) {
    std::ifstream file(mEnginePath, std::ios::binary);
    assert(file.good());

    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);

    std::vector<char> modelData(size);
    file.read(modelData.data(), size);
    file.close();

    mRuntime = nvinfer1::createInferRuntime(gLogger);
    assert(mRuntime != nullptr);

    mEngine = mRuntime->deserializeCudaEngine(modelData.data(), size);
    assert(mEngine != nullptr);

    mContext = mEngine->createExecutionContext();
    assert(mContext != nullptr);

    mInputIndex = mEngine->getBindingIndex("image");
    mOutputIndex = mEngine->getBindingIndex("prob");

    auto inputDims = mEngine->getBindingDimensions(mInputIndex);
    auto outputDims = mEngine->getBindingDimensions(mOutputIndex);

    mInputSize = 1 * 3 * 128 * 128;
    mOutputSize = 1 * 21;

    cudaMalloc(&mBuffers[mInputIndex], mInputSize * sizeof(float));
    cudaMalloc(&mBuffers[mOutputIndex], mOutputSize * sizeof(float));
    cudaStreamCreate(&mStream);
}

number_class::~number_class() {
    cudaFree(mBuffers[mInputIndex]);
    cudaFree(mBuffers[mOutputIndex]);
    cudaStreamDestroy(mStream);
    mContext->destroy();
    mEngine->destroy();
    mRuntime->destroy();
}

void number_class::preprocess(const cv::Mat& image, float* inputBuffer) {
    cv::Mat resizedImage, floatImage;

    cv::resize(image, resizedImage, cv::Size(128, 128));
    cv::cvtColor(resizedImage, resizedImage, cv::COLOR_BGR2RGB);
    resizedImage.convertTo(floatImage, CV_32FC3, 1.0f / 255.0f);

    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < 128; ++h) {
            for (int w = 0; w < 128; ++w) {
                inputBuffer[c * 128 * 128 + h * 128 + w] = floatImage.at<cv::Vec3f>(h, w)[c];
            }
        }
    }
}

int number_class::infer(const cv::Mat& inputImage, float& confidence, std::string& label) {
    float* inputBuffer = new float[mInputSize];
    preprocess(inputImage, inputBuffer);

    cudaMemcpyAsync(mBuffers[mInputIndex], inputBuffer, mInputSize * sizeof(float), cudaMemcpyHostToDevice, mStream);
    mContext->enqueueV2(mBuffers, mStream, nullptr);

    float outputBuffer[21];
    cudaMemcpyAsync(outputBuffer, mBuffers[mOutputIndex], mOutputSize * sizeof(float), cudaMemcpyDeviceToHost, mStream);
    cudaStreamSynchronize(mStream);

    int predictedClass = std::distance(outputBuffer, std::max_element(outputBuffer, outputBuffer + 21));
    confidence = outputBuffer[predictedClass];
    label = labels_name[predictedClass];

    delete[] inputBuffer;
    return predictedClass;
}

void number_class::drawResult(cv::Mat& image, const std::string& label, float confidence) {
    // 构建要显示的文本 (结果 + 置信度)
    std::string text = label + ": " + std::to_string(confidence);

    // 获取文本的尺寸，用于计算绘制的背景框
    int baseLine = 0;
    cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 1, 2, &baseLine);

    // 设置文本在图像上的起点坐标
    cv::Point textOrigin(10, textSize.height + 10);

    // 绘制一个白色背景框以确保文字可见
    cv::rectangle(image, textOrigin + cv::Point(0, baseLine), textOrigin + cv::Point(textSize.width, -textSize.height), cv::Scalar(255, 255, 255), cv::FILLED);

    // 在背景框上绘制黑色文字
    cv::putText(image, text, textOrigin, cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0), 2);

    // 保存图像
    //cv::imwrite("output_with_prediction.jpg", image);
}

void number_class::drawResult(cv::Mat& image, const std::string& label, float confidence, cv::Rect &bound_box) {
    // 构建要显示的文本 (结果 + 置信度)
    std::string text = label + ": " + std::to_string(confidence);

    // 获取文本的尺寸，用于计算绘制的背景框
    int baseLine = 0;
    cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 1, 2, &baseLine);

    // 设置文本在矩形框左上角的起点坐标
    cv::Point textOrigin(bound_box.x, bound_box.y - textSize.height - 10); // 在矩形上方显示


    // 在背景框上绘制红色文字
    cv::putText(image, text, textOrigin, cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2); // 红色文字

    // 绘制红色矩形框
    cv::rectangle(image, bound_box, cv::Scalar(0, 0, 255), 2); // 红色矩形框

}
