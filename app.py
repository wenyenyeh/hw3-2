import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
num_points = 600
mean = 0
variance = 10
x1 = np.random.normal(mean, np.sqrt(variance), num_points)
x2 = np.random.normal(mean, np.sqrt(variance), num_points)

distances = np.sqrt(x1**2 + x2**2)

Y = np.where(distances < 10, 0, 1)

plt.figure(figsize=(8, 6))
plt.scatter(x1[Y==0], x2[Y==0], color='blue', marker='o', label='Y=0')
plt.scatter(x1[Y==1], x2[Y==1], color='red', marker='s', label='Y=1')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Scatter Plot with Two Classes (Y=0 and Y=1)')
plt.legend()
plt.grid()
plt.show()
