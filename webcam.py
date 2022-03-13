import numpy as np
import cv2


# Capturing video through webcam
webcam = cv2.VideoCapture(0)

# Start a while loop
while(1):

    # Reading the video from the
    # webcam in image frames
    _, imageFrame = webcam.read()

    # Convert the imageFrame in
    # BGR(RGB color space) to
    # HSV(hue-saturation-value)
    # color space

    imageFrame = cv2.GaussianBlur(imageFrame, (5, 5), 0)

    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

    # Set range for red color and
    # define mask
    red_lower = np.array([136, 87, 111], np.uint8)
    red_upper = np.array([180, 255, 255], np.uint8)
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
    lower_white = np.array([0,0,0])
    upper_white = np.array([0,0,sensitivity])

    black_mask = cv2.inRange(hsvFrame, lower_white, upper_white)

    # Morphological Transform, Dilation
    # for each color and bitwise_and operator
    # between imageFrame and mask determines
    # to detect only that particular color
    kernal = np.ones((5, 5), "uint8")

    # For red color
    red_mask = cv2.dilate(red_mask, kernal)
    res_red = cv2.bitwise_and(imageFrame, imageFrame,
                              mask = red_mask)

    # For green color
    green_mask = cv2.dilate(green_mask, kernal)
    res_green = cv2.bitwise_and(imageFrame, imageFrame,
                                mask = green_mask)

    # For blue color
    blue_mask = cv2.dilate(blue_mask, kernal)
    res_blue = cv2.bitwise_and(imageFrame, imageFrame,
                               mask = blue_mask)

    # Bitwise-AND mask and original image
    black_mask = cv2.dilate(black_mask, kernal)
    res = cv2.bitwise_and(imageFrame, imageFrame, mask= black_mask)

    # Creating contour to track red color
    contours, hierarchy = cv2.findContours(red_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if(area > 300):
            approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)

            # draws boundary of contours.
            cv2.drawContours(imageFrame, [approx], 0, (0, 0, 255), 5)
            # compute the center of the contour
            M = cv2.moments(cnt)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            cv2.circle(imageFrame, (cX, cY), 7, (255, 255, 255), -1)

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

    contours, hierarchy = cv2.findContours(green_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if(area > 300):
            approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)

            # draws boundary of contours.
            cv2.drawContours(imageFrame, [approx], 0, (0, 0, 255), 5)

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

    contours, hierarchy = cv2.findContours(blue_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if(area > 300):
            approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)

            # draws boundary of contours.
            cv2.drawContours(imageFrame, [approx], 0, (0, 0, 255), 5)

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

    contours, hierarchy = cv2.findContours(black_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if(area > 300):
            approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)

            # draws boundary of contours.
            cv2.drawContours(imageFrame, [approx], 0, (0, 0, 255), 5)

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

    cv2.imshow("Multiple Color Detection in Real-TIme", imageFrame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break