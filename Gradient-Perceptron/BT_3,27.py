import numpy as np

w = np.array([1, 2, -10])
x = np.array([3, 4, 1])
y_true = -1

# 1. Tính w^T * x
score = np.dot(w, x)

# 2. Dự đoán nhãn bằng hàm sgn
y_pred = 1 if score >= 0 else -1

# 3. Kiểm tra phân lớp
is_misclassified = (y_pred != y_true)

print(f"1. w^T * x = {score}")
print(f"2. Nhãn dự đoán (y_pred): {y_pred}")
print(f"3. Bị phân lớp sai? -> {is_misclassified}")