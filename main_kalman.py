import cv2
import pyvisgraph as vg

import os
import sys
import time
import math
import numpy as np
import random

# Connect and use Thymio
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
from Thymio import Thymio

th = Thymio.serial(port="COM14", refreshing_rate=0.1)


def angle_to_vector(angle):
    return math.cos(angle), math.sin(angle)


def detect_thymio_contour(_frame):
    black_lower = np.array([0, 0, 0])
    black_upper = np.array([50, 50, 100])
    black_mask = cv2.inRange(_frame, black_lower, black_upper)

    black_mask = cv2.dilate(black_mask, np.ones((5, 5), "uint8"))

    # take 2 black contours
    black_contours = cv2.findContours(black_mask,
                                      cv2.RETR_TREE,
                                      cv2.CHAIN_APPROX_SIMPLE)[0]

    thymio_contours = []
    area_prev = 0
    for cnt in black_contours:
        area = cv2.contourArea(cnt)
        if area > 300:
            # Making sure to have bigger circle (Thymio center) as first entry
            if area > area_prev:
                thymio_contours.insert(0, cnt)
            else:
                thymio_contours.append(cnt)
            # Quit if 2 circles detected
            if len(thymio_contours) == 2:
                return thymio_contours
            area_prev = area

    return 0


def detect_goal_contour(_frame):
    # Set range for green color and
    # define mask
    green_lower = np.array([25, 52, 72], np.uint8)
    green_upper = np.array([102, 255, 255], np.uint8)
    green_mask = cv2.inRange(_frame, green_lower, green_upper)

    green_mask = cv2.dilate(green_mask, np.ones((5, 5), "uint8"))

    green_contours = cv2.findContours(green_mask,
                                      cv2.RETR_TREE,
                                      cv2.CHAIN_APPROX_SIMPLE)[0]

    for cnt in green_contours:
        area = cv2.contourArea(cnt)
        if area > 300:
            return cnt
    return 0


def detect_obstacles_contours(_frame, _thymio):
    red_lower = np.array([0, 50, 80])
    red_upper = np.array([10, 255, 255])
    red_mask = cv2.inRange(_frame, red_lower, red_upper)

    red_mask = cv2.dilate(red_mask, np.ones((_thymio.get_radius(), _thymio.get_radius()), "uint8"))

    return cv2.findContours(red_mask,
                            cv2.RETR_TREE,
                            cv2.CHAIN_APPROX_SIMPLE)[0]


def detect_corner_contours(_frame):
    # Set range for blue color and
    # define mask
    blue_lower = np.array([94, 80, 2], np.uint8)
    blue_upper = np.array([120, 255, 255], np.uint8)
    blue_mask = cv2.inRange(_frame, blue_lower, blue_upper)

    blue_mask = cv2.dilate(blue_mask, np.ones((5, 5), "uint8"))

    return cv2.findContours(blue_mask,
                            cv2.RETR_TREE,
                            cv2.CHAIN_APPROX_SIMPLE)[0]


class Thymio_Info:

    def __init__(self, x, y, enc_radius, yaw):
        self._radius = None
        self._startX = None
        self._startY = None
        self._yaw = yaw
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

    def get_yaw(self):
        return self._yaw

    def set_yaw(self, yaw):
        self._yaw = yaw


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


class PidVar:
    Kp_dev = 90
    Ki_dev = 0
    Kd_dev = 6000

    Kp_ori = 40
    Ki_ori = 0
    Kd_ori = 0

    def __init__(self):
        self._error_sum = 0
        self._previous_error = 0
        self._error_sum2 = 0
        self._previous_error2 = 0
        self._pid_selector = "orientation_pid"
        return

    def get_error_sum(self):
        return self._error_sum

    def set_error_sum(self, error_sum):
        self._error_sum = error_sum

    def get_previous_error(self):
        return self._previous_error

    def set_previous_error(self, previous_error):
        self._previous_error = previous_error

    def get_pid_selector(self):
        return self._pid_selector

    def set_pid_selector(self, pid_selector):
        self._pid_selector = pid_selector

    def get_error_sum2(self):
        return self._error_sum2

    def set_error_sum2(self, error_sum2):
        self._error_sum2 = error_sum2

    def get_previous_error2(self):
        return self._previous_error2

    def set_previous_error2(self, previous_error2):
        self._previous_error2 = previous_error2


# PID CONTROL FUNCTIONS
def is_next_goal_reached(next_goal_point, _thymio_info):
    tolerance = 20
    distance = dist_2_points(next_goal_point, _thymio_info.get_position())

    if distance < tolerance:
        return True

    return False


def dist_2_points(point1, point2):  # compute distance between two points
    vect = np.subtract(point1, point2)
    vector = np.array(vect)
    magnitude = np.linalg.norm(vector)
    return magnitude


def move(l_speed, r_speed):
    # Changing negative values to the expected ones with the bitwise complement
    l_speed = l_speed if l_speed >= 0 else 2 ** 16 + l_speed
    r_speed = r_speed if r_speed >= 0 else 2 ** 16 + r_speed

    # Setting the motor speeds
    th.set_var("motor.left.target", l_speed)
    th.set_var("motor.right.target", r_speed)


# alpha_r: angle between robot_position and next_goal
# alpha_g: angle between last_goal and next_goal
def get_deviation_error(robot_pos, robot_orientation, last_goal, next_goal):
    alpha_r = math.atan2(next_goal[1] - robot_pos[1], next_goal[0] - robot_pos[0])
    alpha_g = math.atan2(next_goal[1] - last_goal[1], next_goal[0] - last_goal[0])

    # print(alpha_r*180/math.pi, alpha_g*180/math.pi, (alpha_g-alpha_r)*180/math.pi) # put in degrees for understandability
    deviation_angle = alpha_g - alpha_r
    distance_to_path = abs(math.sin(deviation_angle) * dist_2_points(robot_pos, next_goal))
    deviation_distance = 1 / (-10 + distance_to_path) + 1 / (10 + distance_to_path)
    print(distance_to_path)
    return deviation_angle


def pid_deviation_regulator(last_goal, next_goal, _pid, _thymio):
    error_sum = _pid.get_error_sum()
    previous_error = _pid.get_previous_error()

    error_deviation = get_deviation_error(_thymio.get_position(), _thymio.get_yaw(), last_goal, next_goal)
    error_sum = error_sum + error_deviation
    error_diff = error_deviation - previous_error
    previous_error = error_deviation
    deviation_correction = _pid.Kp_dev * error_deviation + _pid.Ki_dev * error_sum + _pid.Kd_dev * error_diff

    _pid.set_error_sum(error_sum)
    _pid.set_previous_error(previous_error)
    return deviation_correction


def get_orientation_error(robot_orientation, robot_pos, last_goal, next_goal):
    trajectory_orientation = math.atan2(next_goal[1] - last_goal[1], next_goal[0] - last_goal[0])
    alpha_r = math.atan2(next_goal[1] - robot_pos[1], next_goal[0] - robot_pos[0])
    # print(robot_orientation * 180 / math.pi, trajectory_orientation * 180 / math.pi)
    return robot_orientation - trajectory_orientation + (trajectory_orientation - alpha_r)


def pid_orientation_regulator(last_goal, next_goal, _pid, _thymio):
    error_sum = _pid.get_error_sum2()
    previous_error = _pid.get_previous_error2()

    error_orientation = get_orientation_error(_thymio.get_yaw(), _thymio.get_position(), last_goal, next_goal)
    error_sum = error_sum + error_orientation
    error_diff = error_orientation - previous_error
    previous_error = error_orientation
    orientation_correction = _pid.Kp_ori * error_orientation + _pid.Ki_ori * error_sum + _pid.Kd_ori * error_diff

    _pid.set_error_sum2(error_sum)
    _pid.set_previous_error2(previous_error)
    return orientation_correction


def motion_control(straight_speed, last_goal, next_goal, _pid, _thymioinfo):
    if _pid.get_pid_selector() == "orientation_pid":
        orientation_correction = pid_orientation_regulator(last_goal, next_goal, _pid, _thymioinfo)
        straight_speed = 0
        deviation_correction = 0
        if abs(orientation_correction) < 1:
            _pid.__init__()
            _pid.set_pid_selector("deviation_pid")
            move(0, 0)
            pid_deviation_regulator(last_goal, next_goal, _pid, _thymioinfo)

    else:
        # deviation_angle = get_deviation_error(_thymioinfo.get_position(), last_goal, next_goal)
        # distance_to_path = abs(math.sin(deviation_angle) * dist_2_points(_thymioinfo.get_position(), next_goal))

        distance_to_goal = dist_2_points(_thymioinfo.get_position(), next_goal)
        if distance_to_goal < 50:
            straight_speed = int(straight_speed / 2)

        deviation_correction = pid_deviation_regulator(last_goal, next_goal, _pid, _thymioinfo)
        orientation_correction = 0

    left_speed = straight_speed - int(orientation_correction) - int(deviation_correction)
    right_speed = straight_speed + int(orientation_correction) + int(deviation_correction)
    move(left_speed, right_speed)
    #motor_speeds = [left_speed, right_speed]

    #return motor_speeds


# Video Capture functions
def get_camera_frame(capture):
    ret, frame = capture.read()
    hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    return frame, hsvFrame


def find_draw_thymio_state(frame, hsvFrame):
    # get Thymios contour
    thymio_contour = detect_thymio_contour(hsvFrame)

    # get position of center and radius of circle enclosing thymio
    (thymio_center_x, thymio_center_y), enc_radius = cv2.minEnclosingCircle(thymio_contour[0])
    M = cv2.moments(thymio_contour[1])
    thymio_front_x = int(M["m10"] / M["m00"])
    thymio_front_y = int(M["m01"] / M["m00"])

    yaw = math.atan2(thymio_front_y - thymio_center_y, thymio_front_x - thymio_center_x)

    # draw a circle around thymio
    cv2.circle(frame, tuple([int(thymio_center_x), int(thymio_center_y)]), int(enc_radius), (0, 255, 0), 2)
    cv2.line(frame, (int(thymio_center_x), int(thymio_center_y)), (int(thymio_front_x), int(thymio_front_y)),
             (300, 100, 80), 3, 8)

    state = np.array([thymio_center_x, thymio_center_y, yaw])

    return state, enc_radius


def create_goal_object(frame, hsvFrame):
    goal_contour = detect_goal_contour(hsvFrame)

    M = cv2.moments(goal_contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    goal = Goal(*(cX, cY))

    string = str(cX) + " " + str(cY)

    cv2.circle(frame, goal.get_position(), 7, (255, 255, 255), -1)
    cv2.putText(frame, string, goal.get_position(),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

    return goal


def find_shortest_path(thymio, goal, hsvFrame, firstFrame):
    # Create path object
    optimalPath = Path()
    optimalPath.set_start_point(vg.Point(thymio.get_position()[0], thymio.get_position()[1]))
    optimalPath.set_end_point(vg.Point(goal.get_position()[0], goal.get_position()[1]))

    # Find obstacles and dilate shape by thymio contour
    obstacle_contours = detect_obstacles_contours(hsvFrame, thymio)
    for obstacle_contour in obstacle_contours:

        area = cv2.contourArea(obstacle_contour)
        if (area > 5200):  # explain area size!
            approx = cv2.approxPolyDP(obstacle_contour, 0.009 * cv2.arcLength(obstacle_contour, True), True)

            single_contour_poly = []
            for point in approx:
                single_contour_poly.append(vg.Point(point[0][0], point[0][1]))

            # Add dilated contour corners to path points
            optimalPath.set_obstacle(single_contour_poly)

            # draws boundary of contours.
            cv2.drawContours(firstFrame, [approx], 0, (255, 0, 0), 5)

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
                    cv2.putText(firstFrame, string, (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
                i = i + 1

    g = vg.VisGraph()
    g.build(optimalPath.get_obstacles())

    shortest_path = g.shortest_path(optimalPath.get_start_point(), optimalPath.get_end_point())

    # Convert shortest path to np array
    pathtodraw = []
    for coordiates in shortest_path:
        pathtodraw.append([coordiates.x, coordiates.y])
    array = np.asarray(pathtodraw)

    # Draw path
    isClosed = False
    color = (255, 255, 0)
    thickness = 2
    img2 = cv2.polylines(firstFrame, np.int32([array]), isClosed, color, thickness)

    return array


# Kalman filter functions
def calc_input(motor_speeds):
    R = 2.23
    L = 9.4

    v = R/2*(motor_speeds[0] + motor_speeds[1]) # [cm/s]
    w = R/L*(motor_speeds[0] - motor_speeds[1]) # [rad/s]

    return [v, w]


# calculate next state
def motion_model(x, u, DT):
    A = np.array([[1.0, 0, 0],
                  [0, 1.0, 0],
                  [0, 0, 1.0]])
    # Need to update B to work with left and right motor inputs

    # EITHER TRANSFORM MOTOR SPEEDS TO NEW STATES IN FUNCTION ABOVE OR CHANGE HERE DIRECTLY
    B = np.array([[DT * math.cos(x[2]), 0],
                  [DT * math.sin(x[2]), 0],
                  [0.0, DT]])

    x = A @ x + B @ u
    return x


def main():
    # Start video capture through webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    # Get Frame
    firstFrame, hsvFrame = get_camera_frame(cap)

    # Initial state and draw circle around on firstFrame
    state, enc_radius = find_draw_thymio_state(firstFrame, hsvFrame)
    # Create Thymio object
    thymio_info = Thymio_Info(*(int(state[0]), int(state[1])), enc_radius, state[2])

    # Find goal and draw onto firstFrame
    goal = create_goal_object(firstFrame, hsvFrame)

    # Find shortest path around obstacle contours and draw onto frame
    array = find_shortest_path(thymio_info, goal, hsvFrame, firstFrame)

    for point in array:
        cv2.circle(firstFrame, (int(point[0]), int(point[1])), 1, (0, 0, 255), 2)

    # Debug
    print("Yaw: " + str(math.degrees(thymio_info.get_yaw())) + " Position: " + str(thymio_info.get_position()))

    # Used for PD control
    last_goal_index = 0
    next_goal_index = 1  # show which is the next goal in the variable"array" and also count the nb of goals already reached

    pid = PidVar()
    # pid.__init__()

    # Covariance for KF -> NEED TO TUNE THESE
    Q = np.diag([  # corresponding to stochastic perturbation from input
        0.1,  # variance of location on x-axis
        0.1,  # variance of location on y-axis
        0.1  # variance of yaw angle
    ])

    R = np.diag([  # corresponding to measurement error from camera (< Q)
        0.01,
        0.01,
        0.05
    ])  # Observation x,y position and yaw

    # initial sigma is Q
    C = np.identity(3)
    Sigma = Q
    S = Sigma + R

    # initial estimated next state (no input)
    mu = state
    u = [0, 0]
    t_start = time.time()
    t_end = 0

    while True:

        # Get Frame
        frame, hsvFrame = get_camera_frame(cap)

        # New Thymio state measurement
        state, enc_radius = find_draw_thymio_state(frame, hsvFrame)
        thymio_info.set_position(int(state[0]), int(state[1]))
        thymio_info.set_yaw(state[2])

        '''        
        # simulate measurement of movement towards x
        pos_error = random.randint(0,2)/10
        yaw_error = np.deg2rad(random.randint(-2,2))
        state += [1+pos_error, pos_error, yaw_error]
        '''

        '''
        # simulate input
        error = random.randint(0, 2) / 100
        u = [0.1 + error, 0.1 + error]
        '''

        # time keeping for dt
        t_end = time.time()
        dt = t_end - t_start

        mu_est = motion_model(mu, u, dt)
        print("dt = " + str(dt))

        t_start = time.time()

        # Update matrices
        Sigma_est = Sigma + Q
        S = Sigma_est + R

        innovation = state - mu_est

        K = Sigma_est.dot(C.dot(np.linalg.inv(S)))
        mu = mu_est + K.dot(innovation)  # or mu a posteriori
        Sigma = (np.identity(3) - K.dot(C)).dot(Sigma_est)

        # What does this do?
        cv2.circle(frame, (int(array[next_goal_index][0]), int(array[next_goal_index][1])), 5, (304, 68, 66), 2)

        next_goal_reached = is_next_goal_reached(array[next_goal_index], thymio_info)
        if next_goal_reached:
            print("go next point now")
            last_goal_index = next_goal_index
            next_goal_index = next_goal_index + 1
            next_goal_reached = False
            pid.__init__()
            if len(array) == next_goal_index:  # if the final goal is reached -> robot stops
                move(0, 0)
                break
            # align_to_next_goal(array[next_goal_index], thymio_info)

        # save input given to compute estimated next state
        motion_control(50, array[last_goal_index], array[next_goal_index], pid, thymio_info)
        #move(motor_speeds[0], motor_speeds[1])

        # u = calc_input(motor_speeds, dt)
        u = motor_speeds # this is probably wrong!

        print(state, mu)
        print()  # we should be able to work with state or mu directly -> filtered state of Thymio

        cv2.imshow("Multiple Color Detection in Real-TIme", frame)
        c = cv2.waitKey(1)
        if c == 27:
            break

    cv2.destroyAllWindows()
    cap.release()


main()