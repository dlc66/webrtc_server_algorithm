#include <fstream>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "cuda_runtime.h"

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

class DaoZhaClassifier {
public:
    DaoZhaClassifier(const std::string& engine_file_path);
    ~DaoZhaClassifier();
    std::string infer(const cv::Mat& input_image);

private:
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IRuntime* runtime;
    nvinfer1::IExecutionContext* context;
    cudaStream_t stream;
    void* input_buffer;
    void* output_buffer;
    int input_size;
    int output_size;

    const std::vector<std::string> labels = {"fenzha", "hezha"};
    cv::Mat preprocess(const cv::Mat& image);
};

DaoZhaClassifier::DaoZhaClassifier(const std::string& engine_file_path)
    : engine(nullptr), runtime(nullptr), context(nullptr), stream(0), input_buffer(nullptr), output_buffer(nullptr) {
    
    // 加载 TensorRT 引擎
    std::ifstream engine_file(engine_file_path, std::ios::binary);
    if (!engine_file) {
        throw std::runtime_error("Failed to open engine file.");
    }

    engine_file.seekg(0, std::ios::end);
    size_t file_size = engine_file.tellg();
    engine_file.seekg(0, std::ios::beg);

    std::vector<char> engine_data(file_size);
    engine_file.read(engine_data.data(), file_size);

    runtime = nvinfer1::createInferRuntime(gLogger);
    engine = runtime->deserializeCudaEngine(engine_data.data(), file_size, nullptr);
    context = engine->createExecutionContext();

    // 创建 CUDA 流
    cudaStreamCreate(&stream);

    // 假设模型的输入输出大小固定
    input_size = 1 * 3 * 224 * 224;  // 例如 224x224 图像，3 个通道
    output_size = 2;  // 因为有两个类别 'fenzha' 和 'hezha'

    // 分配 GPU 缓存
    cudaMalloc(&input_buffer, input_size * sizeof(float));
    cudaMalloc(&output_buffer, output_size * sizeof(float));
}

DaoZhaClassifier::~DaoZhaClassifier() {
    cudaStreamDestroy(stream);
    cudaFree(input_buffer);
    cudaFree(output_buffer);
    context->destroy();
    engine->destroy();
    runtime->destroy();
}

cv::Mat DaoZhaClassifier::preprocess(const cv::Mat& image) {
    cv::Mat resized, float_image, normalized_image;
    cv::resize(image, resized, cv::Size(224, 224));  // 调整大小为224x224
    resized.convertTo(float_image, CV_32F);  // 转换为浮点型 图像转换
    normalized_image = float_image / 255.0f;  // 归一化到0-1范围
    return normalized_image;
}

std::string DaoZhaClassifier::infer(const cv::Mat& input_image) {
    cv::Mat preprocessed_image = preprocess(input_image);

    // 将图像转换为 NCHW 格式并复制到 GPU
    std::vector<float> input_data(1 * 3 * 224 * 224);
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

// 打印输出数据以检查
for (size_t i = 0; i < output_data.size(); ++i) {
    std::cout << "Output[" << i << "]: " << output_data[i] << std::endl;
}

int max_index = std::distance(output_data.begin(), std::max_element(output_data.begin(), output_data.end()));
return labels[max_index];

}

int main() {
    try {
        DaoZhaClassifier classifier("daozha_class/daozha_resnet101_bs224_nc2_20240621_1.trtmodel");


        // 读取输入图像
        cv::Mat input_image = cv::imread("daozha_class/2020daozhah3n1.jpg");
        if (input_image.empty()) {
            std::cerr << "Failed to load input image." << std::endl;
            return -1;
        }

        // 进行推理
        std::string result = classifier.infer(input_image);

        // 输出分类结果
        std::cout << "Classification Result: " << result << std::endl;

        // 在图像上绘制结果并保存
        cv::putText(input_image, result, cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        cv::imwrite("output_image.jpg", input_image);
        std::cout << "Output image saved to path/to/output_image.jpg" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }

    return 0;
}


