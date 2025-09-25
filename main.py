# 模型训练有关头文件
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingClassifier

# 模型转换有关头文件
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# onnx模型推理有关头文件
import numpy as np
import onnxruntime as rt

if __name__ == '__main__':
    # 训练模型
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    gbdt = GradientBoostingClassifier()
    gbdt.fit(X_train, y_train)
    gbdt_val = gbdt.predict(X_test)
    print(mean_absolute_error(y_test, gbdt_val))

    # 模型转换
    initial_type = [('float_input', FloatTensorType([1, 4]))]
    print(FloatTensorType([1, 4]))
    onx = convert_sklearn(gbdt, initial_types=initial_type)
    with open("gbdt_iris.onnx", "wb") as f:
        f.write(onx.SerializeToString())

    # python实现onnx模型推理
    onnx_model = r"gbdt_iris.onnx"
    sess = rt.InferenceSession(
        onnx_model, providers=rt.get_available_providers())
    
    inputs = np.array([5.1, 3.5, 1.4, 0.2], dtype=np.float32).reshape(1, 4)
    input_name = sess.get_inputs()[0].name
    pred_onx = sess.run(None, {input_name: inputs})[0]
    print(pred_onx)