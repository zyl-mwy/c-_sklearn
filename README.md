# ONNX Runtime C++ inference example

This small project shows how to run the same ONNX inference from C++ as the Python snippet:

Python example:

```python
inputs = np.array([5.1, 3.5, 1.4, 0.2], dtype=np.float32).reshape(1, 4)
pred_onx = sess.run(None, {input_name: inputs})[0]
```

C++ example is in `src/main.cpp`. It loads `gbdt_iris.onnx` (current directory by default) and runs a single inference.

Build (Linux):

1. Install ONNX Runtime C++ package. Either:
   - Install system packages or build ONNX Runtime from source and install the CMake config, or
   - Download the prebuilt Linux package and point CMake to it using `-Donnxruntime_DIR=/path/to/onnxruntime/lib/cmake/onnxruntime`

2. Build with CMake:

```bash
mkdir -p build && cd build
cmake ..
make
```

3. Run (model file expected in the current directory or pass path as first arg):

```bash
./onnx_cpp_infer
```

Notes:
- If CMake cannot find ONNX Runtime, set `ONNXRUNTIME_INCLUDE_DIR` and `ONNXRUNTIME_LIB_DIR` when invoking CMake.
- The example uses the C++ API (onnxruntime_cxx_api.h).

# How to Prepare
1. Run "main.py" to get "gbdt_iris.onnx"
```bash
pip install scikit-learn
pip install skl2onnx
pip install numpy
pip install onnxruntime
python main.py
```

2. Install "onnxruntime" environment for "c++"
```bash
sudo apt install build-essential cmake git libpython3-dev python3-pip

git clone --recursive https://github.com/Microsoft/onnxruntime
cd onnxruntime

# For CPU
./build.sh --skip_tests --config Release --build_shared_lib --parallel

# For GPU
./build.sh --skip_tests --use_cuda --config Release --build_shared_lib --parallel --cuda_home /usr/local/cuda --cudnn_home /usr/local/cuda

# For TensorRT
./build.sh --skip_tests --use_tensorrt --config Release --build_shared_lib --parallel --tensorrt_home /path/to/TensorRT
```
After compilation, the library files will be generated in the onnxruntime/build/Linux/Release/ directory. In the CMakeLists.txt, set ONNXRUNTIME_ROOT_PATH to this path.

3. How to solve the problem about anaconda when building onnxruntime with ./build.sh

Disable the anaconda environment by editing the "~./bashrc" and removing code about ananconda path. And then reboot your Ubuntu System.
