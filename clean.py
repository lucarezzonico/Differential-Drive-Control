import cv2
import pyvisgraph as vg

import os
import sys
import time
import math
import numpy as np


def angle_to_vector(angle):
    return math.cos(angle), math.sin(angle)


def detect_thymio_contour(_frame):
    blue_lower = np.array([94, 80, 2], np.uint8)
    blue_upper = np.array([120, 255, 255], np.uint8)

    blue_mask = cv2.inRange(_frame, blue_lower, blue_upper)

    blue_mask = cv2.dilate(blue_mask, np.ones((5, 5), "uint8"))

    # take 2 black contours
    blue_contours = cv2.findContours(blue_mask,
                                     cv2.RETR_TREE,
                                     cv2.CHAIN_APPROX_SIMPLE)[0]

    thymio_contours = []
    area_prev = 0
    for cnt in blue_contours:
        area = cv2.contourArea(cnt)
        if area > 200:
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
    ## Gen lower mask (0-5) and upper mask (175-180) of RED
    mask1 = cv2.inRange(_frame, (0, 50, 20), (5, 255, 255))
    mask2 = cv2.inRange(_frame, (175, 50, 20), (180, 255, 255))

    ## Merge the mask and crop the red regions
    mask = cv2.bitwise_or(mask1, mask2)

    red_mask = cv2.dilate(mask,
                          np.ones((int(_thymio.get_radius() * 1.5), int(_thymio.get_radius() * 1.5)), "uint8"))

    return cv2.findContours(red_mask,
                            cv2.RETR_TREE,
                            cv2.CHAIN_APPROX_SIMPLE)[0]


class Thymio_Info:

    def __init__(self, x, y, enc_radius, yaw, front_x, front_y):
        self._radius = 0
        self._front_x = front_x
        self._front_y = front_y
        self._yaw = yaw
        self._x = x
        self._y = y
        self.set_radius(enc_radius)

    def get_position(self):
        return self._x, self._y

    def set_position(self, x, y):
        self._x = x
        self._y = y

    def get_front_position(self):
        return self._front_x, self._front_y

    def set_front_position(self, front_x, front_y):
        self._front_x = front_x
        self._front_y = front_y

    def get_radius(self):
        return self._radius

    def set_radius(self, radius):
        safety_margin = 2
        self._radius = int(radius * safety_margin)

    def get_yaw(self):
        return self._yaw

    def set_yaw(self, yaw):
        self._yaw = yaw


class Kalman_State:

    def __init__(self, x, y, yaw):
        self._yaw = yaw
        self._x = x
        self._y = y

    def get_position(self):
        return self._x, self._y

    def set_position(self, x, y):
        self._x = x
        self._y = y

    def get_yaw(self):
        return self._yaw

    def set_yaw(self, yaw):
        self._yaw = yaw


class Mile_Stone:
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
    Kp_dev = 45
    Ki_dev = 0
    Kd_dev = 8000

    Kp_ori = 40  # perfect
    Ki_ori = 0
    Kd_ori = 0

    def __init__(self):
        self._error_sum = 0
        self._previous_error = 0
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


def is_next_goal_reached(tolerance, next_goal_point, _thymio_info):
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
def get_deviation_error(robot_pos, last_goal, next_goal):
    alpha_r = math.atan2(next_goal[1] - robot_pos[1], next_goal[0] - robot_pos[0])
    alpha_g = math.atan2(next_goal[1] - last_goal[1], next_goal[0] - last_goal[0])
    deviation_angle = alpha_g - alpha_r

    # modulo 2pi
    if deviation_angle > math.pi:
        deviation_angle = deviation_angle - 2 * math.pi
    elif deviation_angle < -math.pi:
        deviation_angle = deviation_angle + 2 * math.pi

    return deviation_angle


def pid_deviation_regulator(last_goal, next_goal, _pid, _thymio):
    error_sum = _pid.get_error_sum()
    previous_error = _pid.get_previous_error()

    error_deviation = get_deviation_error(_thymio.get_position(), last_goal, next_goal)
    error_sum = error_sum + error_deviation
    error_diff = error_deviation - previous_error
    previous_error = error_deviation
    deviation_correction = _pid.Kp_dev * error_deviation + _pid.Ki_dev * error_sum + _pid.Kd_dev * error_diff

    _pid.set_error_sum(error_sum)
    _pid.set_previous_error(previous_error)
    return deviation_correction


def get_orientation_error(robot_orientation, robot_pos, next_goal):
    alpha_r = math.atan2(next_goal[1] - robot_pos[1], next_goal[0] - robot_pos[0])
    orientation_angle = robot_orientation - alpha_r

    # modulo 2pi
    if orientation_angle > math.pi:
        orientation_angle = orientation_angle - 2 * math.pi
    elif orientation_angle < -math.pi:
        orientation_angle = orientation_angle + 2 * math.pi

    return orientation_angle


def pid_orientation_regulator(next_goal, _pid, _thymio):
    error_sum = _pid.get_error_sum()
    previous_error = _pid.get_previous_error()

    error_orientation = get_orientation_error(_thymio.get_yaw(), _thymio.get_position(), next_goal)
    error_sum = error_sum + error_orientation
    error_diff = error_orientation - previous_error
    previous_error = error_orientation
    orientation_correction = _pid.Kp_ori * error_orientation + _pid.Ki_ori * error_sum + _pid.Kd_ori * error_diff

    _pid.set_error_sum(error_sum)
    _pid.set_previous_error(previous_error)
    return orientation_correction


# slow the robot to a max speed
# sets the highest value of wheel speed to the max_speed, and proportionnaly changes the other wheel speed
def slow_robot(max_speed, straight_speed, deviation_correction):
    left_speed = straight_speed - int(deviation_correction)
    right_speed = straight_speed + int(deviation_correction)
    left_sign = np.sign(left_speed)
    right_sign = np.sign(right_speed)
    left_speed = abs(left_speed)
    right_speed = abs(right_speed)
    if left_speed > right_speed:
        right_speed = (right_speed * max_speed) / left_speed
        left_speed = max_speed
    else:
        left_speed = (left_speed * max_speed) / right_speed
        right_speed = max_speed

    left_speed = left_sign * int(left_speed)
    right_speed = right_sign * int(right_speed)
    move(left_speed, right_speed)
    return left_speed, right_speed


def motion_control(straight_speed, last_goal, next_goal, _pid, _thymioinfo):
    if _pid.get_pid_selector() == "orientation_pid":
        orientation_correction = pid_orientation_regulator(next_goal, _pid, _thymioinfo)
        deviation_correction = 0
        straight_speed = 0
        if abs(orientation_correction) < 10:
            _pid.set_error_sum(0)
            _pid.set_previous_error(0)
            _pid.set_pid_selector("deviation_pid")
            move(0, 0)
            pid_deviation_regulator(last_goal, next_goal, _pid, _thymioinfo)
    else:  # in pid_selector = "deviation_pid"
        deviation_correction = pid_deviation_regulator(last_goal, next_goal, _pid, _thymioinfo)
        orientation_correction = 0

        # slows the robot when close to a goal
        # sets the max speed to 50
        distance_to_goal = dist_2_points(_thymioinfo.get_position(), next_goal)
        if distance_to_goal < 60:
            speed = slow_robot(50, straight_speed, deviation_correction)
            return [speed[0], speed[1]]

    left_speed = straight_speed - int(orientation_correction) - int(deviation_correction)
    right_speed = straight_speed + int(orientation_correction) + int(deviation_correction)
    move(left_speed, right_speed)

    return [left_speed, right_speed]


# Kalman filter functions
def calc_input(motor_speeds):  # motor_speeds has an input speed in [tymspeed]
    R = 2.23 * cm_to_pixel  # [pixel]
    L = 9.4 * cm_to_pixel  # [pixel]

    motor_speeds = [motor_speeds[0] * tymspeed_to_pixelpersec, motor_speeds[1] * tymspeed_to_pixelpersec]
    v = (1 / 2) * (motor_speeds[0] + motor_speeds[1])  # [pixel/s]
    w = (2 / L) * (motor_speeds[1] - motor_speeds[0])  # [rad/s]

    return [v, w]


# calculate next state
def motion_model(_kalman_state, u, DT):
    A = np.array([[1.0, 0, 0],
                  [0, 1.0, 0],
                  [0, 0, 1.0]])
    # Need to update B to work with left and right motor inputs

    # EITHER TRANSFORM MOTOR SPEEDS TO NEW STATES IN FUNCTION ABOVE OR CHANGE HERE DIRECTLY
    B = np.array([[DT * math.cos(_kalman_state.get_yaw()), 0],
                  [DT * math.sin(_kalman_state.get_yaw()), 0],
                  [0.0, DT]])

    return A.dot([*_kalman_state.get_position(), _kalman_state.get_yaw()]) + B.dot(u)


def jacobian_motion(_kalman_state, u, dt):
    # Jacobian of Motion Model
    v = u[0]
    G = np.array([
        [1.0, 0.0, -dt * v * math.sin(_kalman_state.get_yaw())],
        [0.0, 1.0, dt * v * math.cos(_kalman_state.get_yaw())],
        [0.0, 0.0, 1.0]
    ])

    return G


def local_avoidance():
    # Weights of neuron inputs
    w_l = np.array([40, 20, -20, -20, -40, 0, 0])
    w_r = np.array([-40, -20, -20, 20, 40, 0, 0])

    # Scale factors for sensors and constant factor
    sensor_scale = 3000
    # constant_scale = 20

    x = np.zeros(shape=(7,))
    y = np.zeros(shape=(2,))

    # Get and scale inputs
    x = np.array(th["prox.horizontal"]) / sensor_scale

    # Compute outputs of neurons and set motor powers
    y[0] = np.sum(x * w_l)
    y[1] = np.sum(x * w_r)

    move(50 + int(y[0]), 50 + int(y[1]))


def is_obstacle_detected():
    sensor_scale = 800
    obstacle_prox = np.array(th["prox.horizontal"]) / sensor_scale

    if max(obstacle_prox) > 2:
        return True
    else:
        return False


def calculate_next_kalman_state(is_thymio_visible, _kalman_state, _thymio_info, u, dt, _sigma):
    H = np.identity(3)

    if (is_thymio_visible):
        Q = np.diag([0.5, 0.5, 0.5])
        R = np.diag([0.01, 0.01, 0.03])

    else:
        Q = np.diag([0.5, 0.5, 0.5])
        R = np.diag([100000, 100000, 100000])

    mu_est = motion_model(_kalman_state, u, dt)
    G = jacobian_motion(_kalman_state, u, dt)

    innovation = np.subtract([*_thymio_info.get_position(), _thymio_info.get_yaw()], H.dot(mu_est))

    sigma_est = G.dot(_sigma.dot(np.transpose(G))) + Q
    S = H.dot(sigma_est.dot(np.transpose(H))) + R

    K = sigma_est.dot(np.transpose(H).dot(np.linalg.inv(S)))
    _mu = mu_est + K.dot(innovation)  # or mu a posteriori
    _sigma = (np.identity(3) - K.dot(H)).dot(sigma_est)

    return Kalman_State(_mu[0], _mu[1], _mu[2]), _sigma


def calculate_thymio_info(thymio_contour):
    (thymio_center_x, thymio_center_y), enc_radius = cv2.minEnclosingCircle(thymio_contour[0])
    M = cv2.moments(thymio_contour[1])
    thymio_front_x = int(M["m10"] / M["m00"])
    thymio_front_y = int(M["m01"] / M["m00"])
    yaw = math.atan2(thymio_front_y - thymio_center_y, thymio_front_x - thymio_center_x)

    return Thymio_Info(int(thymio_center_x), int(thymio_center_y), enc_radius, yaw, int(thymio_front_x),
                       int(thymio_front_y))


def calculate_goal(goal_contour):
    M = cv2.moments(goal_contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    return Mile_Stone(*(cX, cY))


def add_obstacles_to_path(_obstacle_contours, _optimal_path):
    for obstacle_contour in _obstacle_contours:

        area = cv2.contourArea(obstacle_contour)
        if (area > 5000):
            approx = cv2.approxPolyDP(obstacle_contour, 0.009 * cv2.arcLength(obstacle_contour, True), True)

            single_contour_poly = []
            for point in approx:
                single_contour_poly.append(vg.Point(point[0][0], point[0][1]))

            _optimal_path.set_obstacle(single_contour_poly)


sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

from Thymio import Thymio

th = Thymio.serial(port="COM15", refreshing_rate=0.1)
time.sleep(3)

# each 1 cm is 4.27617 pixels
# so a precision of 1.39mm among an axis and a precision of 1.96 in diagonal
cm_to_pixel = 4.27617
# each 1 tymspeed is 0.128285 pixel/sec
tymspeed_to_pixelpersec = 0.128285


def main():
    g = vg.VisGraph()
    optimalPath = Path()
    pid = PidVar()

    Sigma = np.diag([0.5, 0.5, 0.5])
    u = [0, 0]

    last_goal_index = 0
    next_goal_index = 1

    mile_stones = []

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    ret, first_frame = cap.read()
    hsv_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2HSV)

    thymio_contour = detect_thymio_contour(hsv_frame)
    thymio_info = calculate_thymio_info(thymio_contour)

    goal_contour = detect_goal_contour(hsv_frame)
    goal = calculate_goal(goal_contour)

    obstacle_contours = detect_obstacles_contours(hsv_frame, thymio_info)

    optimalPath.set_start_point(vg.Point(thymio_info.get_position()[0], thymio_info.get_position()[1]))
    optimalPath.set_end_point(vg.Point(goal.get_position()[0], goal.get_position()[1]))
    add_obstacles_to_path(obstacle_contours, optimalPath)

    g.build(optimalPath.get_obstacles())
    shortest_path = g.shortest_path(optimalPath.get_start_point(), optimalPath.get_end_point())

    for point_on_path in shortest_path:
        mile_stones.append([point_on_path.x, point_on_path.y])

    mile_stones_as_array = np.asarray(mile_stones)

    kalman_state = Kalman_State(*thymio_info.get_position(), thymio_info.get_yaw())

    while True:

        if is_obstacle_detected():
            local_avoidance()
        else:

            t_start = time.perf_counter()

            ret, image_frame = cap.read()

            hsv_frame = cv2.cvtColor(image_frame, cv2.COLOR_BGR2HSV)

            thymio_contour = detect_thymio_contour(hsv_frame)

            if not (thymio_contour == 0):

                thymio_info = calculate_thymio_info(thymio_contour)

                t_end = time.perf_counter()
                dt = t_end - t_start

                kalman_state, Sigma = calculate_next_kalman_state(True, kalman_state, thymio_info, u, dt, Sigma)

                cv2.line(image_frame, thymio_info.get_position(),
                         thymio_info.get_front_position(),
                         (300, 100, 80), 3, 8)

                # draw a circle around thymio and point at next goal
                cv2.circle(image_frame, thymio_info.get_position(), thymio_info.get_radius(), (0, 255, 0), 2)
                for mile_stone in mile_stones_as_array:
                    cv2.circle(image_frame, (int(mile_stone[0]), int(mile_stone[1])), 1, (0, 0, 255), 2)
                cv2.circle(image_frame, (
                    int(mile_stones_as_array[next_goal_index][0]), int(mile_stones_as_array[next_goal_index][1])), 5,
                           (304, 68, 66),
                           2)

            else:

                t_end = time.perf_counter()
                dt = t_end - t_start

                kalman_state, Sigma = calculate_next_kalman_state(False, kalman_state, thymio_info, u, dt, Sigma)

            next_goal_reached = is_next_goal_reached(20, mile_stones_as_array[next_goal_index], kalman_state)
            if next_goal_reached:
                print("Proceeding to next milestone..")
                last_goal_index = next_goal_index
                next_goal_index = next_goal_index + 1

                pid.set_error_sum(0)
                pid.set_previous_error(0)
                pid.set_pid_selector("orientation_pid")

                if len(mile_stones_as_array) == next_goal_index:  # if the final goal is reached -> robot stops
                    move(0, 0)
                    break

            motor_speeds = motion_control(150, mile_stones_as_array[last_goal_index],
                                          mile_stones_as_array[next_goal_index], pid, kalman_state)
            u = calc_input(motor_speeds)

            cv2.imshow("Basics in Robotics || Thymio Live Stream", image_frame)

            c = cv2.waitKey(1)
            if c == 27:
                break

    cv2.destroyAllWindows()
    cap.release()
    return True


if main():
    print('Successfully reached the goal!!')
