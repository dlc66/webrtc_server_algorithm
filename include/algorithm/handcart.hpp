#include <fstream>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "../common.hpp"
#include "NvInfer.h"
#include "cuda_runtime.h"

Logger gLogger;

class HandCartClassifier {
public:
    HandCartClassifier(const std::string& engine_file_path);
    ~HandCartClassifier();
    std::pair<std::string, float> infer(const cv::Mat& input_image); // 修改返回类型
private:
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IRuntime* runtime;
    nvinfer1::IExecutionContext* context;
    cudaStream_t stream;
    void* input_buffer;
    void* output_buffer;
    int input_size;
    int output_size;

    const std::vector<std::string> labels = {};
    cv::Mat preprocess(const cv::Mat& image);
};

HandCartClassifier::HandCartClassifier(const std::string& engine_file_path)
    : engine(nullptr), runtime(nullptr), context(nullptr), stream(0), input_buffer(nullptr), output_buffer(nullptr) {
    
    // 加载 TensorRT 引擎
    std::ifstream engine_file(engine_file_path, std::ios::binary);
    if (!engine_file) {
        throw std::runtime_error("Failed to open engine file.");
    }

    engine_file.seekg(0, std::ios::end);
    size_t file_size = engine_file.tellg();//获取文件大小
    engine_file.seekg(0, std::ios::beg);

    std::vector<char> engine_data(file_size);
    engine_file.read(engine_data.data(), file_size);//读取文件数据

    runtime = nvinfer1::createInferRuntime(gLogger);
    engine = runtime->deserializeCudaEngine(engine_data.data(), file_size, nullptr);
    context = engine->createExecutionContext();

    // 创建 CUDA 流
    cudaStreamCreate(&stream);

    // 假设模型的输入输出大小固定
    input_size = 1 * 3 * 640 * 640;  // 例如 224x224 图像，3 个通道
    output_size = 2;  // 因为有两个类别 'fenzha' 和 'hezha'

    // 分配 GPU 缓存
    cudaMalloc(&input_buffer, input_size * sizeof(float));
    cudaMalloc(&output_buffer, output_size * sizeof(float));
}

HandCartClassifier::~HandCartClassifier() {
    cudaStreamDestroy(stream);
    cudaFree(input_buffer);
    cudaFree(output_buffer);
    context->destroy();
    engine->destroy();
    runtime->destroy();
}

cv::Mat HandCartClassifier::preprocess(const cv::Mat& image) {
    cv::Mat resized, float_image, normalized_image;
    cv::resize(image, resized, cv::Size(640, 640));  // 调整大小为640x640
    resized.convertTo(float_image, CV_32F);  // 转换为浮点型
    normalized_image = float_image / 255.0f;  // 归一化到0-1范围
    return normalized_image;//返回归一化
}

std::pair<std::string, float> HandCartClassifier::infer(const cv::Mat& input_image) {
    cv::Mat preprocessed_image = preprocess(input_image);

    // 将图像转换为 NCHW 格式并复制到 GPU
    std::vector<float> input_data(1 * 3 * 640 * 640);
    std::memcpy(input_data.data(), preprocessed_image.data, input_size * sizeof(float));

    cudaMemcpyAsync(input_buffer, input_data.data(), input_size * sizeof(float), cudaMemcpyHostToDevice, stream);

    // 推理
    void* buffers[] = {input_buffer, output_buffer};
    context->enqueueV2(buffers, stream, nullptr);

    // 将输出数据从设备传输到主机
    std::vector<float> output_data(output_size);
    cudaMemcpyAsync(output_data.data(), output_buffer, output_size * sizeof(float), cudaMemcpyDeviceToHost, stream);

    // 同步 CUDA 流
    cudaStreamSynchronize(stream);

    // 找到最高分数的类别及其置信度
    int max_index = std::distance(output_data.begin(), std::max_element(output_data.begin(), output_data.end()));
    float confidence = output_data[max_index]; // 最高分数即为置信度
    return {labels[max_index], confidence};
}