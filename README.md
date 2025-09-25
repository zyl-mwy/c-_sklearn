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
cmake .. -Donnxruntime_DIR=/path/to/onnxruntime/lib/cmake/onnxruntime
cmake --build . -j
```

3. Run (model file expected in the current directory or pass path as first arg):

```bash
./onnx_cpp_infer /path/to/gbdt_iris.onnx
```

Notes:
- If CMake cannot find ONNX Runtime, set `ONNXRUNTIME_INCLUDE_DIR` and `ONNXRUNTIME_LIB_DIR` when invoking CMake.
- The example uses the C++ API (onnxruntime_cxx_api.h).
