import numpy as np
from collections import deque

THRESHOLD = 0.4


def search_center(matrix):
    h, w = matrix.shape
    q = deque()
    q.append((0, 0, h, w))
    while len(q) > 0:
        x1, y1, x2, y2 = q.popleft()
        if x1 == x2 - 1 or y1 == y2 - 1: continue

        md_x = (x1 + x2) // 2
        md_y = (y1 + y2) // 2
        if not matrix[md_x, md_y]: return md_x, md_y

        q.extend([
            (x1, y1, md_x, md_y),
            (md_x, y1, x2, md_y),
            (md_x, md_y, x2, y2),
            (x1, md_y, md_x, y2),
        ])
    raise Exception("Can not find region")


def binary_search(l, r, check):
    while l != r:
        md = (l + r) // 2
        if check(md):
            l = md + 1
        else:
            r = md
    return l


def find_bounds(d, threshold=THRESHOLD):
    h, w = d.shape
    matrix = d > threshold

    center_x, center_y = search_center(matrix)

    top = binary_search(0, center_x, lambda x: np.all(matrix[x - 1:x, :]))
    bottom = binary_search(center_x, h, lambda x: not np.all(matrix[x:x + 1, :])) - 1
    left = binary_search(0, center_y, lambda x: np.all(matrix[top:bottom + 1, x - 1:x]))
    right = binary_search(center_y, w, lambda x: not np.all(matrix[top:bottom + 1, x:x + 1])) - 1
    return top, bottom, left, right
