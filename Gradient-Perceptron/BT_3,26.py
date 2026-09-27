import numpy as np

# f(x) = x^2 - 4x + 5 -> đạo hàm: f'(x) = 2x - 4
def grad(x):
    return 2 * x - 4

def my_gd(x0, eta):
    x = [x0]
    for it in range(100):
        x_new = x[-1] - eta * grad(x[-1])
        if abs(grad(x_new)) < 1e-4:  # Đạo hàm xấp xỉ 0 thì dừng
            break
        x.append(x_new)
    return x, it

# Chạy đề bài: x0 = 5, eta = 0.2
x_history, it = my_gd(x0=5.0, eta=0.2)

print(f"Hội tụ tại x = {x_history[-1]:.4f} sau {it} vòng lặp")
print(f"5 bước đầu: {[round(val, 4) for val in x_history[:5]]}")