from config import *
import cv2
import numpy as np
import itertools
import random

def closest_node(node, nodes):
    print(nodes, node)
    if len(nodes) == 0:
        return 0, np.Inf
    distances = cdist([node], nodes)
    return nodes[distances.argmin()], distances.min()

def add_shape(box, dict_boxes):
    center = np.mean(box, axis=0)
    center = (center[0], center[1])
    closest, distance = closest_node(center, list(dict_boxes.keys()))
    if distance > 100:
        dict_boxes[center] = box
    else:
        dict_boxes[closest] = (dict_boxes[closest] + box)/2


def find_squares(img, boxes):
    image = img.copy()
    # HEURISTIC : board not smaller than 30% of the board and not bigger than 90%
    size_of_image = image.shape[0]* image.shape[1]
    lower_bound_board_size = 0.3 * size_of_image
    upper_bound_board_size = 0.9 * size_of_image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpen = cv2.filter2D(blur, -1, sharpen_kernel)

    # Threshold and morph close
    thresh = cv2.threshold(sharpen, 150, 255, cv2.THRESH_BINARY_INV)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    cnts, _ = cv2.findContours(close, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    for contour in cnts:
        approx = cv2.approxPolyDP(contour, 0.01* cv2.arcLength(contour, True), True)
        x, y , w, h = cv2.boundingRect(approx)
        aspectRatio = float(w)/h
        if  aspectRatio >0.9 and aspectRatio < 1.1 and cv2.contourArea(contour) > 100 and len(approx)< 10 and cv2.contourArea(contour) > lower_bound_board_size and cv2.contourArea(contour) < upper_bound_board_size:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            add_shape(box, boxes)

    for box in boxes.values():
        pts = np.array(box, np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(img,[pts],True,(0,255,0))

    return img, thresh, boxes


def _getNextRegularField(regularFieldsUnvisited, regularFieldsVisited, lastPosition, i):
        if not regularFieldsUnvisited:
                # stop if no points to visit
                return regularFieldsVisited

        sortedUnvisited = sorted(regularFieldsUnvisited.items(), key=lambda x: abs(x[0][0] + x[0][1] - lastPosition[0] - lastPosition[1]))
        newPosition = sortedUnvisited[0][0]
        newKeyPoint = sortedUnvisited[0][1]
        regularFieldsVisited[i] = newKeyPoint
        regularFieldsUnvisited = dict(sortedUnvisited[1:])

        return _getNextRegularField(regularFieldsUnvisited, regularFieldsVisited, newPosition, i + 1)


def createHashmapRegularFields(regularFields):
        '''
        regularFields: key is position of center, value is keypoint

        Creates hashmap for regular fields for easy iteration:
                1. Get initial field (red starter)
                2. Recursive unction for searching next point - _getNextRegularField
        '''

        topFields = dict(sorted(regularFields.items(), key=lambda x: abs(x[0][1]))[:3]) # get highest ones
        topFields = dict(sorted(topFields.items(), key=lambda x: abs(x[0][0])))

        regularFieldsVisited = dict([(i, keypoint) for i, keypoint in enumerate(list(topFields.values()))])
        regularFieldsUnvisited = dict([(key, value) for key, value in regularFields.items() if value not in regularFieldsVisited.values()])
        lastPosition = regularFieldsVisited[len(regularFieldsVisited) - 1].pt

        regularFieldsVisited = _getNextRegularField(regularFieldsUnvisited, regularFieldsVisited, lastPosition, 4)

        return regularFieldsVisited


def get_blobs(frame, draw = True, title = "default gray blob detection"):
        frame_copy = frame.copy()
        frame_blurred = cv2.medianBlur(frame_copy, 3)
        frame_gray = cv2.cvtColor(frame_blurred, cv2.COLOR_BGR2GRAY)
        t_lower = 20 
        t_upper = 250
        canny = cv2.Canny(frame_gray, t_lower, t_upper)
        kernel = np.ones((3, 3), np.uint8)
        
        # dilate the image
        dilate = cv2.dilate(canny, kernel, iterations=1)
        blobs = detector.detect(dilate)

        # overlaying info 
        for blob in blobs:
                pos = blob.pt
                r = blob.size / 2
                
                if draw:
                        cv2.circle(frame_copy, (int(pos[0]), int(pos[1])), int(r), (255, 0, 0), 2)
                        # cv2.putText(frame_copy, str(blob.response),
                        # (int(pos[0]), int(pos[1])),
                        #   cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        
        resized = cv2.resize(frame_copy, (500,500), interpolation = cv2.INTER_AREA)
        if draw:
                cv2.imshow(title, resized)
        
        return frame_copy, blobs

def get_grid(empty_board):
        '''
        TODO: break down into smaller functions
        1. gets regular fields
        2. gets base coordinates
        3. gets home coordinates
        4. geterimes which base is whos etc (by use of other functions)
        '''
        empty_drawn_blobs, circles = get_blobs(empty_board, title = "try on empty", draw = False)
        board_width, board_height, _ = empty_board.shape
        fields = dict()
        for circle in circles:
                pos = circle.pt
                r = circle.size/2
                if r > R_CIRCLE_FIELD * (1 - R_CIRCLE_ACCEPTANCE_THERESHOLD) and  r < R_CIRCLE_FIELD * (1 + R_CIRCLE_ACCEPTANCE_THERESHOLD):
                        fields[(int(pos[0]), int(pos[1]))] =  circle
                        #cv2.circle(empty_drawn_blobs, (int(pos[0]), int(pos[1])), int(r), (255, 0, 0), 2)


        # GET BASE 
        # random needed since else keys wont't be uniqe
        temp_tl_br = sorted({key[0] + key[1] + random.random(): val for key, val in fields.items()}.items())
        temp_tr_bl = sorted({key[0] + board_height - key[1] + random.random(): val for key, val in fields.items()}.items())
       
        # top left
        tl_base =dict(temp_tl_br[:4])
        
        # bottom right
        br_base = dict(temp_tl_br[-4:])

        # top right
        tr_base  = dict(temp_tr_bl[-4:])

        # bottom left
        bl_base  = dict(temp_tr_bl[:4])

        # GET HOME
        # EXPLANATION OF BOARD NOTATION:

        #  __                  __
        # |tl|       tr       |tr|
        #            tr    
        #            tr
        #            tr
        # tl tl tl tl  br br br br br
        #            bl
        #            bl
        #  __        bl        __
        # |bl|       bl       |br|

        temp_y_axis = dict(sorted(fields.items(), key=lambda x: abs(x[0][0] - board_width/2))[:10]) # get middle 10 elements y
        temp_x_axis = dict(sorted(fields.items(), key=lambda x: abs(x[0][1] - board_height/2))[:10]) # get middle 10 elements x

        temp_y_axis = dict(sorted(temp_y_axis.items(), key=lambda x: x[0][1] - board_height/2)[1:9]) # get middle 10 elements y
        temp_x_axis = dict(sorted(temp_x_axis.items(), key=lambda x: x[0][0] - board_width/2)[1:9]) # get middle 10 elements x
        
        tr_home = dict([(i,val) for i,val in enumerate(list(temp_y_axis.values())[:4])])
        bl_home = dict([(i,val) for i,val in enumerate(reversed(list(temp_y_axis.values())[4:]))])

        tl_home = dict([(i,val) for i,val in enumerate(list(temp_x_axis.values())[:4])])
        br_home = dict([(i,val) for i,val in enumerate(reversed(list(temp_x_axis.values())[4:]))])

        base_and_home = [br_home, tl_home, bl_home, tr_home, tr_base, tl_base, bl_base, br_base]
        base_and_home_keypoints = list(itertools.chain.from_iterable( [list(dictionnary.values()) for dictionnary in base_and_home] ))
        
        # get regular fields
        regularFields = dict([(key, value) for key, value in fields.items() if value not in base_and_home_keypoints])
        regularFieldsNumbered = createHashmapRegularFields(regularFields)

        colors = (0,0,0)*40
        for field, color in zip(regularFieldsNumbered.items(), colors):
                i, circle = field[0], field[1]
                pos = circle.pt
                r = circle.size/2
                cv2.circle(empty_drawn_blobs, (int(pos[0]), int(pos[1])), int(r), color, 10)
                cv2.putText(empty_drawn_blobs, str(i),
                        (int(pos[0]), int(pos[1])),
                          cv2.FONT_HERSHEY_PLAIN, 5, color)

        # directory with hash with modulo 40


        resized_board = cv2.resize(empty_drawn_blobs, (500,500), interpolation = cv2.INTER_AREA)
        cv2.imshow("empty_board_blobs " + str(len(circles)) + " size :" + str(r) , resized_board)


if __name__ == "__main__":
    
    prev = cv2.imread('version2.png')
    get_grid(prev)