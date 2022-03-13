# Python code to find the co-ordinates of 
# the contours detected in an image. 
import numpy as np
import cv2
import tqdm
import pyvisgraph as vg

#Source: https://www.geeksforgeeks.org/find-co-ordinates-of-contours-using-opencv-python/
#https://www.pyimagesearch.com/2016/02/15/determining-object-color-with-opencv/
#https://www.geeksforgeeks.org/multiple-color-detection-in-real-time-using-python-opencv/
# Reading image 
font = cv2.FONT_HERSHEY_COMPLEX
img2 = cv2.imread('screen1.png', cv2.IMREAD_COLOR)

# Reading same image in another  
# variable and converting to gray scale. 
img = cv2.imread('screen1.png', cv2.IMREAD_GRAYSCALE)

# Converting image to a binary image 
# ( black and white only image). 
_, threshold = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY)

# Detecting contours in image. 
contours, _ = cv2.findContours(threshold, cv2.RETR_TREE,
                               cv2.CHAIN_APPROX_SIMPLE)

# Going through every contours found in the image.

total_poly = []

for cnt in contours:

    approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)

    single_contour_poly = []
    for point in approx:
        x_coordinate = point[0][0]
        y_coordiante = point[0][1]
        poly = vg.Point(x_coordinate, y_coordiante)
        single_contour_poly.append(poly)

    total_poly.append(single_contour_poly)

    # draws boundary of contours. 
    cv2.drawContours(img2, [approx], 0, (0, 0, 255), 5)

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
            cv2.putText(img2, string, (x, y),
                        font, 0.5, (0, 255, 0))
        i = i + 1

# Showing the final image.

g = vg.VisGraph()
g.build(total_poly)
shortest = g.shortest_path(vg.Point(20,500), vg.Point(600, 200))
print(shortest)
pathtodraw = []

for coordiates in shortest:
    pathtodraw.append([coordiates.x,coordiates.y])

array = np.asarray(pathtodraw)

isClosed = False

# Blue color in BGR
color = (255, 0, 0)

# Line thickness of 2 px
thickness = 2

# Using cv2.polylines() method
# Draw a Blue polygon with
# thickness of 1 px
img2 = cv2.polylines(img2, np.int32([array]), isClosed, color, thickness)

cv2.imshow('image2', img2)


# Exiting the window if 'q' is pressed on the keyboard.
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
