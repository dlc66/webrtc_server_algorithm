#include <iostream>
#include <fstream>
#include <NvInfer.h>
#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>
#include <NvInferRuntime.h>
#include <numeric>
#include <cassert>

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity == Severity::kINFO) {
            return;
        }
        std::cerr << msg << std::endl;
    }
};

Logger gLogger;

class number_class {
public:
    number_class(const std::string& enginePath);
    ~number_class();
    
    void preprocess(const cv::Mat& image, float* inputBuffer);
    int infer(const cv::Mat& inputImage, float& confidence, std::string& label);
    void drawResult(cv::Mat& image, const std::string& label, float confidence);

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
    cv::imwrite("output_with_prediction.jpg", image);
}


int main() {
    std::string modelPath = "models/number_class/classifier_digital_resnet18_128_20230530.trtmodel";
    std::string testImagePath = "test/digital_20200202_0159n3.jpg";

    cv::Mat inputImage = cv::imread(testImagePath);
    if (inputImage.empty()) {
        std::cerr << "Error: Could not open image: " << testImagePath << std::endl;
        return -1;
    }

    number_class classifier(modelPath);

    float confidence;
    std::string label;
    classifier.infer(inputImage, confidence, label);

    std::cout << "Detected digit: " << label << " with confidence: " << confidence << std::endl;

    classifier.drawResult(inputImage, label, confidence);

    return 0;
}




//std::string bp_enginePath = "models/ele_number/szyb_bp_6.0l_nc3_20240906.trtmodel"; // 假设你的 TensorRT engine 文件路径
//std::string engine_file = "defect_25nc/defect_25labels_v5x_640_20240530_nc25.trtmodel";
// cv::Mat image = cv::imread("test/chongqing_bolin_20240408_2aDb_23c5226c-90a5-4c3c-b3b7-1c91f54467e0.jpg");  // 替换为你的测试图像路径

   // cv::Mat image = cv::imread("chongqing_bolin_20240408_2aDb_23c5226c-90a5-4c3c-b3b7-1c91f54467e0.jpg");