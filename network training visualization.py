import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.ticker as ticker

# 設定隨機種子
np.random.seed(42)


# 生成訓練數據
def generate_data(n_samples):
    X = np.random.rand(n_samples, 2)  # 隨機生成 n_samples 個 (x, y)，這邊的 x 是文件中的 x1，y 是 x2
    y = 5 * X[:, 0] + 8 * X[:, 1]  # 計算 f(x, y) = 5x + 8y，這邊的 x 是文件中的 x1，y 是 x2
    return X, y

# 定義activation function和它的導數
def linear_activation(x):
    return x

def linear_activation_derivative(x):
    return 1

# 定義神經網路結構和前向傳播
class SimpleNN:
    def __init__(self):
        self.w1 = np.random.randn()  # 初始化權重 w1
        self.w2 = np.random.randn()  # 初始化權重 w2
        self.b = np.random.randn()   # 初始化偏置項 b

    def forward(self, x):
        self.z = self.w1 * x[:, 0] + self.w2 * x[:, 1] + self.b
        self.a = linear_activation(self.z)
        return self.a

    def compute_loss(self, y_true, y_pred):
        return 0.5 * np.mean((y_true - y_pred) ** 2)

    def backward(self, x, y_true, y_pred, learning_rate):
        m = x.shape[0]
        dz = (y_pred - y_true) * linear_activation_derivative(self.z)
        self.w1 -= learning_rate * np.sum(dz * x[:, 0]) / m
        self.w2 -= learning_rate * np.sum(dz * x[:, 1]) / m
        self.b -= learning_rate * np.sum(dz) / m

# 訓練
def train_nn(nn, X, y, epochs, learning_rate):
    loss_history = []
    w1_history = []
    w2_history = []

    for epoch in range(epochs):
        y_pred = nn.forward(X)
        loss = nn.compute_loss(y, y_pred)
        nn.backward(X, y, y_pred, learning_rate)
        
        loss_history.append(loss)
        w1_history.append(nn.w1)
        w2_history.append(nn.w2)

    return loss_history, w1_history, w2_history

# Main function
X, y = generate_data(100)
nn = SimpleNN()
epochs = 1000
learning_rate = 1

loss_history, w1_history, w2_history = train_nn(nn, X, y, epochs, learning_rate)

# 視覺化訓練過程（有 2D 和 3D 動態）
fig = plt.figure(figsize=(14, 7))

# 2D 視覺化
ax1 = fig.add_subplot(121)
ax1.set_xlim(0, epochs)
ax1.set_ylim(min(min(w1_history), min(w2_history)), max(max(w1_history), max(w2_history))+1)
ax1.yaxis.set_major_locator(ticker.MultipleLocator(1))
line1, = ax1.plot([], [], label='w1')
line2, = ax1.plot([], [], label='w2')
ax1.axvline(x=0, color='red', linestyle='--', label='w1=5, w2=8')  # 初始化紅線
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Weights')
ax1.set_title('Weights w1 and w2 over Epochs')
ax1.legend()
ax1.grid(True)

# 3D 視覺化
ax2 = fig.add_subplot(122, projection='3d')
ax2.set_xlim(min(w1_history) - 2, max(w1_history) + 2)  # 增加軸的大小
ax2.set_ylim(min(w2_history) - 2, max(w2_history) + 2)  # 增加軸的大小
ax2.set_zlim(min(loss_history), max(loss_history))

line3, = ax2.plot([], [], [], label='Training Process')
point, = ax2.plot([], [], [], 'ro')  # 設置更新點為紅色
ax2.set_xlabel('w1')
ax2.set_ylabel('w2')
ax2.set_zlabel('Loss')
ax2.set_title('Cost Function during Training')
ax2.legend()

# 顯示初始和最終權重
initial_w1, initial_w2, initial_loss = w1_history[0], w2_history[0], loss_history[0]
final_w1, final_w2, final_loss = w1_history[-1], w2_history[-1], loss_history[-1]
ax2.text(initial_w1, initial_w2, initial_loss, f'Start\n({initial_w1:.2f}, {initial_w2:.2f})', color='blue')
ax2.text(final_w1, final_w2, final_loss, f'End\n({final_w1:.2f}, {final_w2:.2f})', color='green')

# 用於存儲垂直線段
lines = []

def init():
    line1.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    line3.set_3d_properties([])
    point.set_data([], [])
    point.set_3d_properties([])
    return line1, line2, line3, point

def update(num):
    line1.set_data(range(num), w1_history[:num])
    line2.set_data(range(num), w2_history[:num])
    line3.set_data(w1_history[:num], w2_history[:num])
    line3.set_3d_properties(loss_history[:num])
    point.set_data(w1_history[num-1:num], w2_history[num-1:num])
    point.set_3d_properties(loss_history[num-1:num])
    
    # 更新紅線的位置
    if any(abs(w1 - 5) < 0.1 and abs(w2 - 8) < 0.1 for w1, w2 in zip(w1_history, w2_history)):
        index = next(i for i, (w1, w2) in enumerate(zip(w1_history, w2_history)) if abs(w1 - 5) < 0.1 and abs(w2 - 8) < 0.1)
        ax1.axvline(x=index, color='red', linestyle='--', label='w1=5, w2=8')

    # 新增對應cost的垂直線段
    for line in lines:
        line.remove()
    lines.clear()
    for i in range(num):
        line, = ax2.plot([w1_history[i], w1_history[i]], [w2_history[i], w2_history[i]], [0, loss_history[i]], 'grey')
        lines.append(line)
    
    return line1, line2, line3, point, *lines

# 增加動畫速度，設置interval為10（默認為200），同時設置步長step為10
ani = FuncAnimation(fig, update, frames=range(0, epochs, 10), init_func=init, blit=True, interval=100)

plt.show()

