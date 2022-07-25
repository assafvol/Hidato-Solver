import cv2
import numpy as np


def preProcess(img):
    """Receives colored image and returns binary image"""

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, 1, 1, 11, 2)
    return imgThreshold


def biggestContour(contours):
    """Receives a list of contours a returns the contour of biggest area of biggest area considering only contour of
       area > 50 and of polygon approximation of more than or equal to 20 corners (smallest hidato of side with 3
       hexagons has 30 corners). Also returns the points of the polygon approximation the perimeter
       of the biggest contour  for potential further fitting of polygon"""

    biggest_polygon = np.array([])
    biggest_contour = np.array([])
    biggest_contour_perimeter = None
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 50:
            perimeter = cv2.arcLength(contour, True)
            polygon = cv2.approxPolyDP(contour, 0.003 * perimeter, True)
            if area > max_area and len(polygon) >= 20:
                biggest_contour = contour
                biggest_polygon = polygon
                biggest_contour_perimeter = perimeter
                max_area = area
    return biggest_contour, biggest_polygon, biggest_contour_perimeter


def fit_contour_to_a_possible_hidato_size(contour, polygon, perimeter):
    """In case number of points of the polygon is not valid for a hidato (30,42,54,66... or in general
    6 + 12i for non negative integer i), refit new polygons by until getting to the nearest valid number of points
    and return the newly fitted polygon. If the number of points is valid for a hidato, simply return the input
     polygon"""
    n = len(polygon)
    if n in [30, 42, 54, 66, 78, 90]:
        return polygon
    else:
        alpha = 0.003
        delta = 0.0001
        goal_n = min([30, 42, 54, 66, 78, 90], key=lambda x: abs(x - n))
        sign = np.sign(n - goal_n)

        curr_polygon = polygon
        curr_n = n
        while np.sign(curr_n - goal_n) == sign:
            alpha += sign * delta
            curr_polygon = cv2.approxPolyDP(contour, alpha * perimeter, True)
            curr_n = len(curr_polygon)

        if curr_n != goal_n:
            raise Exception("Failed to fit valid polygon for hidato to contour")
        else:
            return curr_polygon


def apply_perspective_transform(v, mat):
    """Returns a vector which is the result of apply mat to v
    Arguments:
    v - numpy vector
    mat - numpy matrix"""
    v_t = []
    for p in v:
        px = (mat[0][0] * p[0] + mat[0][1] * p[1] + mat[0][2]) / (mat[2][0] * p[0] + mat[2][1] * p[1] + mat[2][2])
        py = (mat[1][0] * p[0] + mat[1][1] * p[1] + mat[1][2]) / (mat[2][0] * p[0] + mat[2][1] * p[1] + mat[2][2])
        p_after = (int(px), int(py))
        v_t.append(p_after)
    return np.array(v_t)


# def find_corners(contour):
#     shape = contour.shape
#     contour = contour.reshape((shape[0], shape[2]))
#     corner_idxs = np.sum((contour - center) ** 2, axis=1).argsort()[-4:]
#     return contour[corner_idxs]


def reorder(points):
    new_points = np.zeros_like(points)
    new_points[0] = min(points, key=lambda el: sum(el))
    new_points[1] = min(points, key=lambda el: el[0] - el[1])
    new_points[2] = max(points, key=lambda el: el[0] - el[1])
    new_points[3] = max(points, key=lambda el: sum(el))
    return new_points


def distance(v1, v2):
    """distance between two 1d vectors represented bynumpy arrays"""

    return np.sqrt(np.sum((v1 - v2) ** 2))


def get_corner_indices(labels):
    corner_idxs = set()
    n = len(labels)
    for i in range(n + 4):
        if len({labels[(i - 1) % n], labels[i % n], labels[(i + 1) % n]}) == 3:
            corner_idxs.add(i % n)
    return sorted(list(corner_idxs))


def find_rectangle_corners(corner_idxs, contour):
    n = len(contour)
    top_left_idx = min(corner_idxs, key=lambda c_idx: distance((0, 0), contour[c_idx]))
    bottom_left_idx = (corner_idxs[(corner_idxs.index(top_left_idx) + 2) % 6] + 1) % n
    bottom_right_idx = corner_idxs[(corner_idxs.index(top_left_idx) + 3) % 6]
    top_right_idx = (corner_idxs[(corner_idxs.index(top_left_idx) + 5) % 6] + 1) % n

    return np.array([top_left_idx, bottom_left_idx, top_right_idx, bottom_right_idx])


def find_hexagons(circum, corners_warped):
    """strategy : find middle y point of every row of hexagons, find unique x points excluding first and last.
    then the center are: for odd_rows even xs """
    all_x = sorted(circum[:, 0])
    all_y = sorted(circum[:, 1])
    n_rows = n_cols = len(circum) // 6
    W = max(all_x) - min(all_x)
    H = max(all_y) - min(all_y)
    h = H / n_rows
    w = W / n_cols
    x = get_different_values(all_x, accuracy=w / 5)[1:-1]
    #     y = get_different_values(all_y,accuracy=h/5)
    top_left_w, bottom_left_w, top_right_w, bottom_right_w = corners_warped
    y_r = np.sort(circum[circum[:, 0] >= min(top_right_w[0], bottom_right_w[0])][:, 1])
    y_l = np.sort(circum[circum[:, 0] <= max(top_left_w[0], bottom_left_w[0])][:, 1])
    y = np.mean(np.stack([y_l, y_r], axis=1), axis=1)
    y = [np.mean([y[i], y[i + 1]]) for i in range(1, len(y) - 1, 2)]

    rows = [[] for i in range(n_rows)]
    x_e = x[0::2]
    x_o = x[1::2]
    n_e, n_o = len(x_e), len(x_o)
    mid = n_rows // 2
    rows[mid] = [[x_e[i], y[mid]] for i in range(0, n_cols)]
    for j in range(1, (n_rows - 1) // 2 + 1):
        if j % 2 == 1:
            rows[mid + j] = [[x_o[k], y[mid + j]] for k in range(j // 2, n_o - j // 2)]
            rows[mid - j] = [[x_o[k], y[mid - j]] for k in range(j // 2, n_o - j // 2)]
        else:
            rows[mid + j] = [[x_e[k], y[mid + j]] for k in range(j // 2, n_e - j // 2)]
            rows[mid - j] = [[x_e[k], y[mid - j]] for k in range(j // 2, n_e - j // 2)]
    return rows


def get_different_values(lst, accuracy):
    groups = []
    curr_group = [lst[0]]
    for i in range(1, len(lst)):
        if lst[i] - lst[i - 1] < accuracy:
            curr_group.append(lst[i])
        else:
            groups.append(curr_group)
            curr_group = [lst[i]]
    if tuple(groups[-1]) != tuple(curr_group):
        groups.append(curr_group)
    return [np.mean(group) for group in groups]
