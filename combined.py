import numpy as np
import cv2
import tqdm
import pyvisgraph as vg


# https://stackoverflow.com/questions/55948254/scale-contours-up-grow-outward
def find_contours(img, to_gray=None):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    morphed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    contours = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours[0]


def mask_from_contours(ref_img, contours):
    mask = np.zeros(ref_img.shape, np.uint8)
    mask = cv2.drawContours(mask, contours, -1, (20, 20, 20), -1)
    return cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)


def dilate_mask(mask, kernel_size=10):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=1)
    return dilated


startPoint = 0
endPoint = 0

imageFrame = cv2.imread('screen10.png', cv2.IMREAD_COLOR)
hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

red_lower = np.array([0, 50, 50])
red_upper = np.array([10, 255, 255])
red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)

# Set range for green color and
# define mask
green_lower = np.array([25, 52, 72], np.uint8)
green_upper = np.array([102, 255, 255], np.uint8)
green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)

# Set range for blue color and
# define mask
blue_lower = np.array([94, 80, 2], np.uint8)
blue_upper = np.array([120, 255, 255], np.uint8)
blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)

sensitivity = 20
lower_white = np.array([0, 0, 0])
upper_white = np.array([0, 0, sensitivity])

black_mask = cv2.inRange(hsvFrame, lower_white, upper_white)

# Morphological Transform, Dilation
# for each color and bitwise_and operator
# between imageFrame and mask determines
# to detect only that particular color
kernal = np.ones((5, 5), "uint8")

# For red color
red_mask = cv2.dilate(red_mask, kernal)
res_red = cv2.bitwise_and(imageFrame, imageFrame,
                          mask=red_mask)

# For green color
green_mask = cv2.dilate(green_mask, kernal)
res_green = cv2.bitwise_and(imageFrame, imageFrame,
                            mask=green_mask)

# For blue color
blue_mask = cv2.dilate(blue_mask, kernal)
res_blue = cv2.bitwise_and(imageFrame, imageFrame,
                           mask=blue_mask)


black_mask = cv2.dilate(black_mask, kernal)
res = cv2.bitwise_and(imageFrame, imageFrame, mask=black_mask)


contours, hierarchy = cv2.findContours(black_mask,
                                       cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[0]
(x, y), radius = cv2.minEnclosingCircle(cnt)
centerThymio = (int(x), int(y))
safetyMargin = 1.5  # 10%
radiusThymio = int(radius * safetyMargin)
img = cv2.circle(imageFrame, centerThymio, radiusThymio, (0, 255, 0), 2)
area = cv2.contourArea(cnt)

if (area > 300):
    approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
    startPoint = vg.Point(centerThymio[0], centerThymio[1])

# determine goal (green)
contours, hierarchy = cv2.findContours(green_mask,
                                       cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)

cnt = contours[0]
area = cv2.contourArea(cnt)
if (area > 300):
    approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)

    M = cv2.moments(cnt)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    string = str(cX) + " " + str(cY)

    cv2.circle(imageFrame, (cX, cY), 7, (255, 255, 255), -1)
    cv2.putText(imageFrame, string, (cX, cY),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

    endPoint = vg.Point(cX, cY)

# Creating contour to track red color
contours, hierarchy = cv2.findContours(red_mask,
                                       cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)

orig_mask = mask_from_contours(imageFrame, contours)

dilated_mask = dilate_mask(orig_mask, radiusThymio)
dilated_contours = find_contours(dilated_mask)

total_poly = []
for cnt in dilated_contours:

    area = cv2.contourArea(cnt)
    if (area > 300):
        approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)

        single_contour_poly = []
        for point in approx:
            x_coordinate = point[0][0]
            y_coordiante = point[0][1]
            poly = vg.Point(x_coordinate, y_coordiante)
            single_contour_poly.append(poly)

        total_poly.append(single_contour_poly)

        # draws boundary of contours.
        cv2.drawContours(imageFrame, [approx], 0, (255, 0, 0), 5)

        # Used to flatted the array containing
        # the co-ordinates of the vertices.
        n = approx.ravel()
        i = 0

        for j in n:
            if (i % 2 == 0):
                x = n[i]
                y = n[i + 1]

                # String containing the co-ordinates.
                string = str(x) + " " + str(y)

                # text on remaining co-ordinates.
                cv2.putText(imageFrame, string, (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
            i = i + 1

        # Creating contour to track green color

contours, hierarchy = cv2.findContours(blue_mask,
                                       cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    area = cv2.contourArea(cnt)
    if (area > 300):
        approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
        # draws boundary of contours.
        cv2.drawContours(imageFrame, [approx], 0, (0, 0, 255), 5)
        M = cv2.moments(cnt)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        string = str(cX) + " " + str(cY)

        cv2.circle(imageFrame, (cX, cY), 7, (255, 255, 255), -1)
        cv2.putText(imageFrame, string, (cX, cY),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

g = vg.VisGraph()
g.build(total_poly)
shortest = g.shortest_path(startPoint, endPoint)
print(shortest)
pathtodraw = []

for coordiates in shortest:
    pathtodraw.append([coordiates.x, coordiates.y])

array = np.asarray(pathtodraw)

isClosed = False

# Blue color in BGR
color = (255, 255, 0)

# Line thickness of 2 px
thickness = 2

# Using cv2.polylines() method
# Draw a Blue polygon with
# thickness of 1 px
img2 = cv2.polylines(imageFrame, np.int32([array]), isClosed, color, thickness)

imS = cv2.resize(imageFrame, (960, 540))  # Resize image
cv2.imshow("Multiple Color Detection in Real-TIme", imS)

# Exiting the window if 'q' is pressed on the keyboard.
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
