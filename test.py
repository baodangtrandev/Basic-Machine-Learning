from ML_library import BasicLinearRegression
import numpy as np 

# Tạo dữ liệu mẫu
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1.2, 1.9, 3.1, 4.0, 5.1])

# Sử dụng lớp BasicLinearRegression
regressor = BasicLinearRegression()
regressor.train(X, y)
predictions = regressor.predict(X)

print("Predictions:", predictions)
