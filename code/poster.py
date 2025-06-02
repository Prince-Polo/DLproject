import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

# 1. 读取灰度图
img = cv2.imread('/root/autodl-tmp/DDColor/test2.png', cv2.IMREAD_GRAYSCALE)

# 2. 提取轮廓
edges = cv2.Canny(img, 100, 200)

# 3. 查找轮廓
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# 4. 创建白底画布
canvas = np.ones_like(img) * 255

# 5. Matplotlib 可视化
plt.figure(figsize=(8, 8))
plt.imshow(canvas, cmap='gray', vmin=0, vmax=255)

all_sampled_points = []

for contour in contours:
    # 6. 画虚线轮廓
    for i in range(0, len(contour)-1, 10):
        pt1 = contour[i][0]
        pt2 = contour[i+1][0]
        plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color='black', linestyle='dashed', linewidth=1)

    # 7. 提取红点
    sampled_points = contour[::20]
    for pt in sampled_points:
        x, y = pt[0]
        plt.plot(x, y, 'ro', markersize=8)
        all_sampled_points.append([x, y])

# 8. 三角剖分
if len(all_sampled_points) >= 3:
    points_array = np.array(all_sampled_points)
    tri = Delaunay(points_array)
    for triangle in tri.simplices:
        pts = points_array[triangle]
        plt.plot([pts[0][0], pts[1][0]], [pts[0][1], pts[1][1]], 'g-', linewidth=1)
        plt.plot([pts[1][0], pts[2][0]], [pts[1][1], pts[2][1]], 'g-', linewidth=1)
        plt.plot([pts[2][0], pts[0][0]], [pts[2][1], pts[0][1]], 'g-', linewidth=1)

plt.axis('off')
plt.tight_layout()
plt.savefig('contour_visualization.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
