from typing import Iterable, List, Tuple, Union
import cv2
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import xml.etree.ElementTree as ET
import re
import tempfile
import os
from PIL import Image
import cairosvg

SVG_NS = 'http://www.w3.org/2000/svg'
ET.register_namespace('', SVG_NS)


# Convert svg to jpg(white background). If you want to convert to png(transparent background), use cairosvg.svg2png directly.
def svg_to_jpg(input_svg_path: str, output_jpg_path: str, scale: float = 1.0):
    temp_png = output_jpg_path.replace('.jpg', '_temp.png')

    cairosvg.svg2png(url=input_svg_path, write_to=temp_png, scale=scale)

    with Image.open(temp_png) as img:
        if img.mode in ('RGBA', 'LA'):
            # 创建白色背景图像
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1])  # 使用 alpha 作为 mask
            background.save(output_jpg_path, 'JPEG')
        else:
            img.convert('RGB').save(output_jpg_path, 'JPEG')

    os.remove(temp_png)

def length_within_points(a : Iterable, empty_value : Union[int, float] = 0) -> int:
    """
        a simple instance:
            array : [empty_value, empty_value, empty_value, 1, empty_value, 0, 1, 2, empty_value]
            Then length_within_points(array) will return index diff between 1 and 2, which is 5
    """
    a = list(a)
    l_pivot, r_pivot = -1, -2
    for index, (l_val, r_val) in enumerate(zip(a[::1], a[::-1])):
        if l_val != empty_value and l_pivot == -1:
            l_pivot = index 
        if r_val != empty_value and r_pivot == -2:
            r_pivot = len(a) - index

    return r_pivot - l_pivot + 1

def extract_m_l_points(d):
    tokens = re.findall(r'[MLZmlz]|-?\d+(?:\.\d+)?', d)
    points = []
    i = 0
    while i < len(tokens):
        if tokens[i] in ['M', 'L']:
            x = float(tokens[i+1])
            y = float(tokens[i+2])
            points.append((x, y))
            i += 3
        elif tokens[i] in ['Z', 'z']:
            if points:
                points.append(points[0])
            i += 1
        else:
            i += 1
    return points

def perpendicular_distance(pt, line_start, line_end):
    pt = np.array(pt)
    start = np.array(line_start)
    end = np.array(line_end)
    if np.allclose(start, end):
        return np.linalg.norm(pt - start)
    return np.linalg.norm(np.cross(end - start, start - pt)) / np.linalg.norm(end - start)

def douglas_peucker(points, epsilon):
    if len(points) < 3:
        return points
    start, end = points[0], points[-1]
    max_dist = 0
    index = 0
    for i in range(1, len(points) - 1):
        dist = perpendicular_distance(points[i], start, end)
        if dist > max_dist:
            index = i
            max_dist = dist
    if max_dist > epsilon:
        left = douglas_peucker(points[:index+1], epsilon)
        right = douglas_peucker(points[index:], epsilon)
        return left[:-1] + right
    else:
        return [start, end]

def points_to_bezier(points):
    if not points:
        return ''
    result = ['M', str(points[0][0]), str(points[0][1])]
    for i in range(1, len(points)):
        x0, y0 = points[i - 1]
        x3, y3 = points[i]
        x1 = x0 + (x3 - x0) / 3
        y1 = y0 + (y3 - y0) / 3
        x2 = x0 + 2 * (x3 - x0) / 3
        y2 = y0 + 2 * (y3 - y0) / 3
        result.extend(['C', str(x1), str(y1), str(x2), str(y2), str(x3), str(y3)])
    if np.allclose(points[0], points[-1]):
        result.append('Z')
    return ' '.join(result)

def remove_element_safely(element, root):
    for parent in root.iter():
        if element in list(parent):
            parent.remove(element)
            break


# Convert image to svg. Works for both png and jpg. Note: it doesn't work for transparent background.
# epsilon is the rate for control points. Larger epsilon means less control points.
def bitmap_to_final_svg(input_bitmap_path: str, output_svg_path: str, epsilon: float = 0.4):
    # Step 1: 轮廓提取并绘制
    img = cv2.imread(input_bitmap_path)
    img = cv2.copyMakeBorder(
        img,
        top=5, bottom=5, left=5, right=5,
        borderType=cv2.BORDER_CONSTANT,
        value=(255, 255, 255)  # 白色边框
    )

    blur = cv2.GaussianBlur(img, (3, 3), 0)
    if len(blur.shape) == 3:
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    else:
        gray = blur  # 已经是灰度图了

    # gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    rings = [np.array(c).reshape([-1, 2]) for c in contours]

    # get ratio between width and height to adjust the final output
    valid_width = length_within_points(bw.sum(axis=0))
    valid_height = length_within_points(bw.sum(axis=1))
    true_ratio = valid_width / valid_height

    max_y = max([ring.max(axis=0)[1] for ring in rings])
    plt.figure(figsize=(8 * true_ratio, 8))
    for i, ring in enumerate(rings):
        close_ring = np.vstack((ring, ring[0]))
        xx = close_ring[..., 0]
        yy = max_y - close_ring[..., 1]
        plt.plot(xx, yy, color="k", linewidth=2)

    plt.axis("off")

    # Step 2: 输出到内存 SVG
    with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tmp_svg:
        temp_svg_path = tmp_svg.name
    plt.savefig(temp_svg_path)
    plt.close()

    # Step 3: 解析 SVG，删除 line2d_1
    tree = ET.parse(temp_svg_path)
    root = tree.getroot()
    for g in list(root.iter(f'{{{SVG_NS}}}g')):
        gid = g.attrib.get('id', '')
        if gid == 'line2d_1' or gid == 'patch_1':
            remove_element_safely(g, root)

    # Step 4: 提取并简化 path 数据
    all_d = []
    for path in list(root.iter(f'{{{SVG_NS}}}path')):
        d_attr = path.attrib.get('d', '')
        points = extract_m_l_points(d_attr)
        simplified = douglas_peucker(points, epsilon=epsilon)
        new_d = points_to_bezier(simplified)
        all_d.append(new_d)

    # 清空原路径
    for g in list(root.iter(f'{{{SVG_NS}}}g')):
        remove_element_safely(g, root)

    # 添加合并路径
    combined_d = ' '.join(all_d)
    new_path = ET.Element(f'{{{SVG_NS}}}path')
    new_path.set('d', combined_d)
    new_path.set('fill', 'black')
    new_path.set('stroke', 'none')
    root.append(new_path)

    tree.write(output_svg_path, encoding='utf-8', xml_declaration=True)

    os.remove(temp_svg_path)  # 删除临时文件

if __name__ == '__main__':
    bitmap_to_final_svg("./111.jpg", "./result.svg", epsilon=1.0)
