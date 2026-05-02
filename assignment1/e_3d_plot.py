import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

df = pd.read_csv("results.csv")

models = df['Model']
accuracy = df['Accuracy']

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x_pos = np.arange(len(models))
y_pos = np.zeros(len(models))
z_pos = np.zeros(len(models))

dx = np.ones(len(models))
dy = np.ones(len(models))
dz = accuracy

ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz)

ax.set_xticks(x_pos)
ax.set_xticklabels(models, rotation=20)
ax.set_title("3D Accuracy Comparison")

plt.show()