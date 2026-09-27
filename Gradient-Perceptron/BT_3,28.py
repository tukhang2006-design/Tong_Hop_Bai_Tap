import numpy as np

w = np.array([-2, 1, 0])
x = np.array([2, 3, 1])
y = 1

# 1. Kiểm tra phân lớp
score = np.dot(w, x)
y_pred = 1 if score >= 0 else -1
print(f"1. w^T * x = {score} -> Dự đoán: {y_pred} (Thực tế: {y})")
print(f"   Bị phân lớp sai: {y_pred != y}")

# 2. Cập nhật w (với eta = 1)
if y_pred != y:
    w_new = w + y * x
    print(f"2. w mới sau cập nhật: {w_new}")

# 3. Tính lại w^T * x
score_new = np.dot(w_new, x)
print(f"3. w_new^T * x = {score_new} -> Dự đoán mới: {1 if score_new >= 0 else -1}")