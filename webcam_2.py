import numpy as np
import cv2
import tqdm
import pyvisgraph as vg


# blabla


def detect_thymio_contour(_frame):
    black_lower = np.array([0, 0, 0])
    black_upper = np.array([0, 0, 20])
    black_mask = cv2.inRange(_frame, black_lower, black_upper)

    black_mask = cv2.dilate(black_mask, np.ones((5, 5), "uint8"))

    # take 2 black contours
    black_contours = cv2.findContours(black_mask,
                                      cv2.RETR_TREE,
                                      cv2.CHAIN_APPROX_SIMPLE)[0]

    thymio_cont = []
    area_prev = 0
    for cnt in black_contours:
        area = cv2.contourArea(cnt)
        if (area > 300):
            # Making sure to have bigger circle (Thymio center) as first entry
            if (area > area_prev):
                thymio_cont.insert(0, cnt)
            else:
                thymio_cont.append(cnt)
            # Quit if 2 circles detected
            if (len(thymio_cont) == 2):
                return thymio_cont
            area_prev = area

    return 0


def detect_goal_contour(_frame):
    # Set range for green color and
    # define mask
    green_lower = np.array([25, 52, 72], np.uint8)
    green_upper = np.array([102, 255, 255], np.uint8)
    green_mask = cv2.inRange(_frame, green_lower, green_upper)

    green_mask = cv2.dilate(green_mask, np.ones((5, 5), "uint8"))

    return cv2.findContours(green_mask,
                            cv2.RETR_TREE,
                            cv2.CHAIN_APPROX_SIMPLE)[0][0]


def detect_obstacles_contours(_frame):
    red_lower = np.array([0, 50, 50])
    red_upper = np.array([10, 255, 255])
    red_mask = cv2.inRange(_frame, red_lower, red_upper)

    red_mask = cv2.dilate(red_mask, np.ones((thymio.get_radius(), thymio.get_radius()), "uint8"))

    return cv2.findContours(red_mask,
                            cv2.RETR_TREE,
                            cv2.CHAIN_APPROX_SIMPLE)[0]


def detect_corner_contours(_frame):
    # Set range for blue color and
    # define mask
    blue_lower = np.array([94, 80, 2], np.uint8)
    blue_upper = np.array([120, 255, 255], np.uint8)
    blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)

    blue_mask = cv2.dilate(blue_mask, np.ones((5, 5), "uint8"))

    return cv2.findContours(blue_mask,
                            cv2.RETR_TREE,
                            cv2.CHAIN_APPROX_SIMPLE)[0]


class Thymio:

    def __init__(self, x, y, enc_radius):
        self._radius = None
        self._startX = None
        self._startY = None
        self._x = x
        self._y = y
        self.set_radius(enc_radius)

    def get_position(self):
        return self._x, self._y

    def set_position(self, x, y):
        self._x = x
        self._y = y

    def get_start_point(self):
        return self._startX, self._startY

    def set_start_point(self, start_x, start_y):
        self._startX = start_x
        self._startY = start_y

    def get_radius(self):
        return self._radius

    def set_radius(self, radius):
        safety_margin = 2
        self._radius = int(radius * safety_margin)


class Goal:
    def __init__(self, x, y):
        self._x = x
        self._y = y

    def get_position(self):
        return self._x, self._y

    def set_position(self, x, y):
        self._x = x
        self._y = y


class Corner:
    def __init__(self, x, y):
        self._x = x
        self._y = y

    def get_position(self):
        return self._x, self._y

    def set_position(self, x, y):
        self._x = x
        self._y = y


class Path:
    def __init__(self):
        self._start_point = None
        self._end_point = None
        self._obstacles = []

    def set_start_point(self, start_point):
        self._start_point = start_point

    def set_end_point(self, end_point):
        self._end_point = end_point

    def get_start_point(self):
        return self._start_point

    def get_end_point(self):
        return self._end_point

    def set_obstacle(self, obstacle):
        self._obstacles.append(obstacle)

    def get_obstacles(self):
        return self._obstacles


imageFrame = cv2.imread('Yaw3.png', cv2.IMREAD_COLOR)

# Convert the imageFrame in
# BGR(RGB color space) to
# HSV(hue-saturation-value)
# color space

imageFrame = cv2.GaussianBlur(imageFrame, (5, 5), 0)
hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

# get Thymios contour
thymio_contour = detect_thymio_contour(hsvFrame)

# get position of center and radius of circle enclosing thymio
(thymio_center_x, thymio_center_y), enc_radius = cv2.minEnclosingCircle(thymio_contour[0])
M = cv2.moments(thymio_contour[1])
thymio_front_x = int(M["m10"] / M["m00"])
thymio_front_y = int(M["m01"] / M["m00"])

orientation = np.array([thymio_front_x - thymio_center_x, thymio_front_y - thymio_center_y])
orientation_norm = orientation / np.linalg.norm(orientation)
v_x = np.array([1, 0])  # unit vector
yaw = np.arccos(np.clip(np.dot(orientation_norm, v_x), -1.0, 1.0))  # in radians

# create a Thymio object
thymio = Thymio(*(int(thymio_center_x), int(thymio_center_y)), enc_radius)
# draw a circle around thymio
cv2.circle(imageFrame, thymio.get_position(), thymio.get_radius(), (0, 255, 0), 2)

# get Goal contour
goal_contour = detect_goal_contour(hsvFrame)

M = cv2.moments(goal_contour)
cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])

goal = Goal(*(cX, cY))

string = str(cX) + " " + str(cY)

cv2.circle(imageFrame, goal.get_position(), 7, (255, 255, 255), -1)
cv2.putText(imageFrame, string, goal.get_position(),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

obstacle_contours = detect_obstacles_contours(hsvFrame)

# specify the initial position of thymio as start point for the path algorithm

optimalPath = Path()
optimalPath.set_start_point(vg.Point(thymio.get_position()[0], thymio.get_position()[1]))
optimalPath.set_end_point(vg.Point(goal.get_position()[0], goal.get_position()[1]))

for obstacle_contour in obstacle_contours:

    approx = cv2.approxPolyDP(obstacle_contour, 0.009 * cv2.arcLength(obstacle_contour, True), True)

    single_contour_poly = []
    for point in approx:
        x_coordinate = point[0][0]
        y_coordiante = point[0][1]
        single_contour_poly.append(vg.Point(point[0][0], point[0][1]))

    optimalPath.set_obstacle(single_contour_poly)

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

corner_contours = detect_corner_contours(hsvFrame)
playground_corners = []
for cnt in corner_contours:
    approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
    # draws boundary of contours.
    cv2.drawContours(imageFrame, [approx], 0, (0, 0, 255), 5)
    M = cv2.moments(cnt)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    corner = Corner(*(cX, cY))

    string = str(cX) + " " + str(cY)

    cv2.circle(imageFrame, corner.get_position(), 7, (255, 255, 255), -1)
    cv2.putText(imageFrame, string, corner.get_position(),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
    playground_corners.append(corner)

g = vg.VisGraph()
g.build(optimalPath.get_obstacles())

shortest_path = g.shortest_path(optimalPath.get_start_point(), optimalPath.get_end_point())
print(shortest_path)
pathtodraw = []

for coordiates in shortest_path:
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
