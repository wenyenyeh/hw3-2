import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import streamlit as st

# 生成2D資料集
np.random.seed(0)
num_points = 600
mean = 0
variance = 10
x1 = np.random.normal(mean, np.sqrt(variance), num_points)
x2 = np.random.normal(mean, np.sqrt(variance), num_points)

# 計算距離來標記類別
distances = np.sqrt(x1**2 + x2**2)
Y = np.where(distances < 10, 0, 1)
X = np.column_stack((x1, x2))

# 使用 SVM 進行分類
svm = SVC(kernel='rbf', C=1, gamma=0.1, probability=True)
svm.fit(X, Y)

# Streamlit 標題
st.title("2D SVM with Circular Distribution")

# 畫出散點圖
st.write("### Original Data Distribution")
fig, ax = plt.subplots()
ax.scatter(x1[Y==0], x2[Y==0], color='blue', marker='o', label='Y=0')
ax.scatter(x1[Y==1], x2[Y==1], color='red', marker='s', label='Y=1')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.legend()
st.pyplot(fig)

# 生成 3D 決策邊界
st.write("### 3D Decision Boundary of SVM")

x1_range = np.linspace(x1.min() - 1, x1.max() + 1, 100)
x2_range = np.linspace(x2.min() - 1, x2.max() + 1, 100)
x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
grid = np.c_[x1_grid.ravel(), x2_grid.ravel()]
z = svm.decision_function(grid).reshape(x1_grid.shape)

# 繪製 3D 圖形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x1_grid, x2_grid, z, cmap='coolwarm', alpha=0.6, edgecolor='none')
ax.scatter(x1[Y==0], x2[Y==0], np.zeros_like(x1[Y==0]), color='blue', marker='o', label='Y=0', zorder=5)
ax.scatter(x1[Y==1], x2[Y==1], np.zeros_like(x1[Y==1]), color='red', marker='s', label='Y=1', zorder=5)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('Decision Function')
ax.set_title('3D Decision Surface')
ax.legend()
st.pyplot(fig)
