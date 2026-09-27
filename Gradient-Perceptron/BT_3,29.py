import numpy as np

class Perceptron:
    def __init__(self, eta=1.0, max_iter=1000):
        self.eta = eta              # Learning rate (tốc độ học)
        self.max_iter = max_iter    # Số vòng lặp tối đa
        self.w = None               # Vector trọng số (bao gồm cả bias)

    def _add_bias(self, X):
        """Thêm cột 1 vào đầu ma trận X để gộp bias w0"""
        ones = np.ones((X.shape[0], 1))
        return np.hstack((ones, X))

    def fit(self, X, y):
        """
        Huấn luyện Perceptron:
        X: ma trận đặc trưng (N mẫu, d chiều)
        y: nhãn thực tế (-1 hoặc 1)
        """
        Xbar = self._add_bias(X)
        d = Xbar.shape[1]
        self.w = np.zeros(d)        # Khởi tạo w toàn 0
        
        for epoch in range(self.max_iter):
            # Dự đoán: w^T * x >= 0 -> 1, ngược lại -1
            y_pred = np.where(np.dot(Xbar, self.w) >= 0, 1, -1)
            
            # Tìm danh sách chỉ số các điểm bị phân lớp sai
            misclassified = np.where(y_pred != y)[0]
            
            # Nếu không còn điểm nào sai -> dữ liệu đã phân tách hoàn toàn
            if len(misclassified) == 0:
                print(f"Hội tụ sau {epoch} vòng lặp.")
                return self
            
            # Chọn 1 điểm sai ngẫu nhiên để cập nhật: w = w + eta * y_i * x_i
            random_id = np.random.choice(misclassified)
            self.w += self.eta * y[random_id] * Xbar[random_id]
            
        print(f"Dừng lại sau {self.max_iter} vòng lặp (có thể dữ liệu không linearly separable).")
        return self

    def predict(self, X):
        """Dự báo nhãn cho dữ liệu mới (-1 hoặc 1)"""
        Xbar = self._add_bias(X)
        return np.where(np.dot(Xbar, self.w) >= 0, 1, -1)

# Tạo 2 cụm dữ liệu phân tách tuyến tính
X_train = np.array([
    [1, 2], [2, 3], [3, 1],   # Cụm 1: nhãn -1
    [6, 5], [7, 7], [8, 6]    # Cụm 2: nhãn 1
])
y_train = np.array([-1, -1, -1, 1, 1, 1])

# Khởi tạo và huấn luyện
model = Perceptron(eta=1.0)
model.fit(X_train, y_train)

# In vector trọng số tìm được: [w0 (bias), w1, w2]
print("Vector trọng số w:", model.w)

# Dự báo nhãn cho 2 điểm mới
X_new = np.array([[2, 2], [7, 6]])
print("Dự báo điểm mới:", model.predict(X_new))