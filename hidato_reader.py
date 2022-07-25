import cv2
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

from hidato_reader_utils import *

# params
show_plots = True
heightImg = 500
widthImg = 500

# resize to square and convert to binary image
img = cv2.imread("hidato_img5_rotated.png")
img = cv2.resize(img, (widthImg, heightImg))
imgThreshold = preProcess(img)

# find contours
imgContours = img.copy()
contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 3)

# find biggest contour
imgBigContours = img.copy()
biggest_contour, biggest_polygon, perimeter = biggestContour(contours)
biggest_polygon = fit_contour_to_a_possible_hidato_size(biggest_contour, biggest_polygon, perimeter)
cv2.drawContours(imgBigContours, biggest_polygon, -1, (0, 255, 0), 5)

# find the four corners of the hidato polygon
polygon = np.squeeze(biggest_polygon)
n = len(polygon)
vectors = np.roll(polygon, -1, axis=0) - polygon
k_means_model = KMeans(n_clusters=6)
vectors_labels = k_means_model.fit_predict(vectors)
corner_indices = get_corner_indices(vectors_labels)
four_corners = polygon[find_rectangle_corners(corner_indices, polygon)]

# Calculate coordinates for perspective transformation
n_points = len(polygon)
# 2 * n_s - 1 = n_points/6 where n_s is the number of hexagons along one of the sides of the hidato and n_points
# is the number of points in the hidato polygon.
n_s = (n_points / 6 + 1) / 2
x_1 = ((n_s / 2) / (2 * n_s - 1)) * widthImg
x_2 = ((n_s / 2 + n_s - 1) / (2 * n_s - 1)) * widthImg

# perspective transformation of image, hidato polygon, and polygon four corners
pts1 = np.float32(four_corners)
pts2 = np.float32([[x_1, 0], [x_1, heightImg], [x_2, 0], [x_2, heightImg]])
matrix = cv2.getPerspectiveTransform(pts1, pts2)
imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
polygon_warped = apply_perspective_transform(polygon, matrix)
corners_warped = apply_perspective_transform(four_corners, matrix)

# find the lattice of hexagons (an array with the centers of all the hexagonal cells in the hidato
circum = np.array(polygon_warped).copy()
lattice = find_hexagons(circum, corners_warped)

if show_plots:
    plt.imshow(imgWarpColored)
    for row in lattice:
        for p in row:
            plt.scatter(x=p[0], y=p[1], c='blue', s=10)
    plt.xlim(0, 500)
    plt.ylim(500, 0)
    plt.show()
