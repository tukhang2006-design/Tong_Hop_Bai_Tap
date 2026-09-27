import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# 1. Tải dữ liệu nhị phân (30 đặc trưng, nhãn: 0 - Ác tính, 1 - Lành tính)
data = load_breast_cancer()
X, y = data.data, data.target

# 2. Chia tập Train/Test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Chuẩn hóa dữ liệu (Z-score normalization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Khởi tạo và huấn luyện Perceptron
# eta0: learning rate, max_iter: số epoch tối đa, tol: điều kiện dừng
clf = Perceptron(eta0=0.1, max_iter=1000, tol=1e-3, random_state=42)
clf.fit(X_train_scaled, y_train)

# 5. Dự báo trên tập kiểm tra
y_pred = clf.predict(X_test_scaled)

# 6. Tính toán các độ đo đánh giá
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("=== KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH PERCEPTRON ===")
print(f"Accuracy (Độ chính xác tổng thể) : {acc:.4f} ({acc*100:.2f}%)")
print(f"Precision (Độ chuẩn xác)        : {prec:.4f}")
print(f"Recall (Độ nhạy / Thu hồi)       : {rec:.4f}")
print(f"F1-score (Trung bình điều hòa)   : {f1:.4f}")

print("\n--- Ma trận nhầm lẫn (Confusion Matrix) ---")
print(confusion_matrix(y_test, y_pred))

print("\n--- Báo cáo chi tiết ---")
print(classification_report(y_test, y_pred, target_names=data.target_names))