from hidato_reader_utils import *
from custom_solver import *
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow import keras

# params
show_plots = True
heightImg = 500
widthImg = 500

# resize to square and convert to binary image
img = cv2.imread("images/hidato_img.png")
img = cv2.resize(img, (widthImg, heightImg))
imgThreshold = pre_process(img)

# find contours
imgContours = img.copy()
contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 3)

# find biggest contour
imgBigContours = img.copy()
biggest_contour, biggest_polygon, perimeter = find_biggest_contour(contours)
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
lattice, h, w = find_hexagons(circum, corners_warped)

# convert the warped img to binary
imgWarpBinary = pre_process(imgWarpColored)
imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)

# remove the hexagonal grid and
nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(imgWarpBinary, connectivity=4)
sizes = stats[:, -1]
largest_components_label = np.argmax(sizes[1:]) + 1
imgWarpBinary[output == largest_components_label] = 0
imgWarpGray[output == largest_components_label] = 255
imgWarpGray[output == 0] = 255

# remove all components whose area is small than 3% of the size of a cell (w*h)
for i, size in enumerate(sizes):
    if i != 0 and size < 0.03 * w * h:
        imgWarpBinary[output == i] = 0
        imgWarpGray[output == i] = 255
# imgWarpGray[output == largest_components_label] = 255


boxes = get_boxes(lattice, h, w, imgWarpBinary)
boxes_gray = get_boxes(lattice, h, w, imgWarpGray)
# plot_boxes(boxes)


# Find digits inside boxes.
digits = [[[] for col in row] for row in boxes]
for i, row in enumerate(boxes):
    for j, box in enumerate(row):
        nb_components, _, stats, centroids = cv2.connectedComponentsWithStats(box, connectivity=4)
        for x, y, width, height, _ in sorted(stats[1:], key= lambda lst: lst[0]):
            digits[i][j].append(cv2.resize(boxes_gray[i][j][y - height//7 : y + height + height//7, x - width//7: x + width + width//7 ], (28, 28)))


digit_images = []
digit_indices = []
for i, row in enumerate(digits):
    for j, digit_list in enumerate(row):
        if digit_list:
            for digit_img in digit_list:
                digit_images.append(digit_img)
                digit_indices.append([i,j])

print(len(digit_images),len(digit_indices))
digit_images = np.expand_dims(np.array(digit_images), axis=3) / 255

plt.imshow(imgWarpGray,cmap='Greys')
plt.show()
model = keras.models.load_model("my_model2")
preds = model.predict(digit_images)
digit_preds = np.argmax(preds,axis=1)

final_lattice = [[[] for _ in row] for row in lattice]
for cell_indices, digit_pred in zip(digit_indices, digit_preds):
    i,j = cell_indices
    final_lattice[i][j].append(digit_pred)

final_lattice = [[int(''.join(map(str, cell))) if cell else 0 for cell in row] for row in final_lattice]
hidato = Hidato(final_lattice)
hidato.plot(initial_cells_only=True)
solution = solve(hidato)
solution.plot()
# final_preds = np.argmax(preds, axis=1)
# final_preds_proba = np.max(preds, axis=1)
# for image, pred, pred_proba in zip(digit_images, final_preds, final_preds_proba):
#     if pred_proba <= 1:
#         print(f"this is a {pred}, i am {pred_proba*100:.2f}% sure")
#         plt.imshow(image, cmap='Greys')
#         plt.show()
