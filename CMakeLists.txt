cmake_minimum_required(VERSION 3.13)
project(onnx_cpp_infer)

set(CMAKE_CXX_STANDARD 17)

# 说明：
# - ORT_INCLUDE_DIR 指向 onnxruntime 的 include 根目录
# - ORT_LIB_DIR 指向包含 libonnxruntime.so 的目录
# 可在配置时覆盖：
#   cmake -S . -B build -DORT_INCLUDE_DIR=/path/include -DORT_LIB_DIR=/path/lib
set(ORT_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/pretrain_onnxruntime/onnxruntime/include" CACHE PATH "onnxruntime include dir")
set(ORT_LIB_DIR     "${CMAKE_SOURCE_DIR}/pretrain_onnxruntime/onnxruntime/build/Linux/Release" CACHE PATH "onnxruntime lib dir")

message(STATUS "ORT include: ${ORT_INCLUDE_DIR}")
message(STATUS "ORT lib    : ${ORT_LIB_DIR}")

add_executable(onnx_cpp_infer main.cpp)

# 头文件搜索路径（包含 onnxruntime_cxx_api.h 所在位置）
target_include_directories(onnx_cpp_infer PRIVATE ${ORT_INCLUDE_DIR})
target_include_directories(onnx_cpp_infer PRIVATE ${ORT_INCLUDE_DIR}/onnxruntime/core/session)

# 链接目录与库
target_link_directories(onnx_cpp_infer PRIVATE ${ORT_LIB_DIR})
target_link_libraries(onnx_cpp_infer PRIVATE onnxruntime)

# 设置 RPATH，运行时可直接找到 libonnxruntime.so
set_target_properties(onnx_cpp_infer PROPERTIES
    BUILD_RPATH "${ORT_LIB_DIR}"
    INSTALL_RPATH "${ORT_LIB_DIR}"
)
