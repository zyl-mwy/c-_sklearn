// 最小化示例
// 功能：
// 1) 定位 ONNX 模型（优先使用命令行参数；否则在 ../../python_train_module 中寻找第一个 .onnx）
// 2) 创建 ONNX Runtime Session
// 3) 构造 1x4 的 float 输入并执行一次推理
// 4) 将输出张量按常见类型（float/int64）打印到标准输出

// onnxruntime C++ API 头文件
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>

// 在给定目录中查找第一个以 .onnx 结尾的文件。若未找到返回空串。
static std::string FindFirstOnnx(const std::filesystem::path& dir) {
	for (const auto& e : std::filesystem::directory_iterator(dir)) {
		if (e.is_regular_file() && e.path().extension() == ".onnx") return e.path().string();
	}
	return {};
}

int main(int argc, char* argv[]) {
	// 解析模型路径：优先使用 argv[1]，否则回退到默认目录扫描
	std::string model_path;
	if (argc > 1) model_path = argv[1];
	if (model_path.empty()) {
		std::filesystem::path def = "../../python_train_module";
		if (!std::filesystem::exists(def)) { std::cerr << "默认目录不存在" << std::endl; return -1; }
		model_path = FindFirstOnnx(def);
		if (model_path.empty()) { std::cerr << "未找到 .onnx" << std::endl; return -1; }
	}

	try {
		// 创建 ORT 环境与推理 Session（默认使用 CPU EP）
		Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "app");
		Ort::SessionOptions opts;
		opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
		Ort::Session session(env, model_path.c_str(), opts);

		// 读取模型 I/O 名称
		auto input_names_s = session.GetInputNames();
		auto output_names_s = session.GetOutputNames();
		if (input_names_s.empty() || output_names_s.empty()) { std::cerr << "模型输入/输出为空" << std::endl; return -1; }
		std::vector<const char*> input_names{ input_names_s[0].c_str() };
		std::vector<const char*> output_names;
		output_names.reserve(output_names_s.size());
		for (auto& s : output_names_s) output_names.push_back(s.c_str());

		// 构造示例输入（1x4 的 float 向量）
		std::vector<int64_t> shape{1, 4};
		std::vector<float> data{5.1f, 3.5f, 1.4f, 0.2f};
		Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
		Ort::Value input = Ort::Value::CreateTensor<float>(mem, data.data(), data.size(), shape.data(), shape.size());

		// 执行推理并将输出按类型打印
		auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input, 1, output_names.data(), output_names.size());
        
        for (size_t idx = 0; idx < outputs.size(); ++idx) {
			auto& o = outputs[idx];
			if (!o.IsTensor()) { std::cout << "Output[" << idx << "]: (非张量，已跳过)\n"; continue; }
			auto info = o.GetTensorTypeAndShapeInfo();
			size_t n = info.GetElementCount();
			if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
				float* ptr = o.GetTensorMutableData<float>();
				for (size_t i = 0; i < n; ++i) std::cout << ptr[i] << (i+1<n? ",":"\n");
			} else if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
				int64_t* ptr = o.GetTensorMutableData<int64_t>();
				for (size_t i = 0; i < n; ++i) std::cout << ptr[i] << (i+1<n? ",":"\n");
			} else {
				std::cout << "Output[" << idx << "]: (不支持的张量类型)\n";
			}
		}
	} catch (const std::exception& e) {
		std::cerr << e.what() << std::endl; return -1;
	}
	return 0;
}
