import cv2
import numpy as np
import itertools
import random

from scipy.spatial.distance import cdist
from config import *

def closest_node(node, nodes):
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


def chizzledBox(box, image):
        '''
        shaves off white boxels off of box:)
        '''
        tl = box[1]
        br = box[3]
        w,h = image.shape[1], image.shape[0]
        OnlyBoardPixels = [(y,x) for y in range(tl[0], br[0] + 1) for x in range(tl[1], br[1] + 1) if image[x,y] == 0]
        tl = np.array(min(OnlyBoardPixels, key= lambda x : x[1]+ x[0]))
        br = np.array(max(OnlyBoardPixels, key= lambda x : x[1]+ x[0]))
        tr = np.array(min(OnlyBoardPixels, key= lambda x : abs(w - x[1]+ x[0])))
        bl = np.array(max(OnlyBoardPixels, key= lambda x : abs(w - x[1]+ x[0])))
       
        box = np.array([tr, tl, bl, br])
        
        return box
        
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
            box = chizzledBox(box, close) # match the boz so that there are no black pixels on sides
            add_shape(box, boxes)

    for box in boxes.values():
        pts = np.array(box, np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(close,[pts],True,100)
    img2 = cv2.resize(close,(int(close.shape[1]/2),int(close.shape[0]/2)))
    #cv2.imshow("check delete later", img2)

    

    return img, thresh, boxes

def getShadowMask(img, originalBoard, maskFields):
        lower = np.array([200]) #lower red color limit
        upper = np.array([255]) # upper limit
        mask = cv2.dilate(cv2.inRange(maskFields,lower,upper), np.ones((15, 15), np.uint8)) + maskOnBases
        imgNoFields = img.copy()
        originalNoFields = originalBoard.copy()
        imgNoFields[maskFields > 100] = 0
        originalNoFields[maskFields > 100] = 0
        difference2 = cv2.subtract( originalNoFields, imgNoFields)
        dst = cv2.inpaint(difference2,mask,3,cv2.INPAINT_TELEA)
        return dst

        
def _getNextRegularField(regularFieldsUnvisited, regularFieldsVisited, lastPosition, i):
        if not regularFieldsUnvisited:
                # stop if no points to visit
                return regularFieldsVisited

        sortedUnvisited = sorted(regularFieldsUnvisited.items(), key=lambda x: abs(x[0][0]  - lastPosition[0])+ abs(x[0][1] - lastPosition[1]))
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


        regularFieldsVisited = _getNextRegularField(regularFieldsUnvisited, regularFieldsVisited, lastPosition, 3)

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
        
        resized = cv2.resize(frame_copy, (500,500), interpolation = cv2.INTER_AREA)
        if draw:
                #cv2.imshow(title, resized)
                pass
        
        return frame_copy, blobs

def get_grid(empty_board):
        '''
        TODO: break down into smaller functions

        1. gets regular fields
        2. gets base coordinates
        3. gets home coordinates
        4. geterimes which base is whos etc (by use of other functions)
        5. scale to 500x500

        @return dictionnary assigning fields to numbers, as below       
                REGULAR FIELDS : 1-40
                HOMES:
                - 41-44 red home
                - 45-48 blue home
                - 49-52 yellow home
                - 53-56 green home
                BASES:
                - 57-60 red base
                - 61-64 blue base
                - 65-68 yellow base
                - 69-72 green base
        @return dictionnary field number and what it means (verbally)
        '''
        empty_drawn_blobs, circles = get_blobs(empty_board, title = "try on empty", draw = False)
        board_width, board_height, _ = empty_board.shape
        fields = dict()
        for circle in circles:
                pos = circle.pt
                r = circle.size/2
                if r > R_CIRCLE_FIELD * (1 - R_CIRCLE_ACCEPTANCE_THERESHOLD) and  r < R_CIRCLE_FIELD * (1 + R_CIRCLE_ACCEPTANCE_THERESHOLD):
                        fields[(int(pos[0]), int(pos[1]))] =  circle


        # GET BASE 
        # random needed since else keys wont't be uniqe
        temp_tl_br = sorted({key[0] + key[1] + random.random(): val for key, val in fields.items()}.items())
        temp_tr_bl = sorted({key[0] + board_height - key[1] + random.random(): val for key, val in fields.items()}.items())
       
        # top left
        tl_base =dict([(i,val) for i,val in enumerate(list(dict(temp_tl_br[:4]).values()))])
        
        # bottom right
        br_base = dict([(i,val) for i,val in enumerate(list(dict(temp_tl_br[-4:]).values()))])

        # top right
        tr_base  = dict([(i,val) for i,val in enumerate(list(dict(temp_tr_bl[-4:]).values()))])

        # bottom left
        bl_base  = dict([(i,val) for i,val in enumerate(list(dict(temp_tr_bl[:4]).values()))])

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

        # directory with hash with modulo 40 for regular points

        # assuming tr is blue, br is green, bl is yellow and tl is red - read board will be rotated
        # each filed gets assigned a number and then the number is checked (what it means)
        ListOfFieldDictionnaries = [regularFieldsNumbered, tl_home, tr_home, bl_home, br_home, tl_base, tr_base, bl_base, br_base]
        ListOfDescription = ['RegularField']*40 + ['Red Home']*4 +  ['Blue Home']*4 +['Yellow Home']*4 +['Green Home']*4 +['Red Base']*4 +['Blue Base']*4 +['Yellow Base']*4 +['Green Base']*4
        DescriptionDictionnaryFinal = dict([(i + 1, description) for i, description in enumerate(ListOfDescription)])
        FieldDictionnaryFinal = dict()
        l = 1

        scale = 1000/(empty_board.shape[0]+ empty_board.shape[1])

        for dictionnary in ListOfFieldDictionnaries:
                for i, keypoint in dictionnary.items():
                        keypoint.pt = (keypoint.pt[0]*scale, keypoint.pt[1]*scale)
                        keypoint.size = keypoint.size*scale
                        FieldDictionnaryFinal[i + l] = keypoint
                l = l + i + 1

        # scaled to 500x500 for easier processing

        return FieldDictionnaryFinal, DescriptionDictionnaryFinal


def createDescriptiveBoard(FieldNumbering):
        '''
        returns 500x500 field with encoded values of fields for nice and quick lookup
        '''
        masks = [np.zeros((500,500)) for _ in range(72)]
        for i, keypoint in FieldNumbering.items():
                pos = keypoint.pt
                r = keypoint.size/2
                cv2.circle(masks[i-1], (int(pos[0]), int(pos[1])), int(r), i, -1)

        return masks

def createMaskFieldBoardExistance(FieldNumbering):
        '''
        returns 500x500 field with encoded values of fields for nice and quick lookup
        '''
        masks = [np.zeros((500,500)) for _ in range(72)]
        for i, keypoint in enumerate(FieldNumbering.values()):
                pos = keypoint.pt
                r = keypoint.size/2
                cv2.circle(masks[i], (int(pos[0]), int(pos[1])), int(r), 255, -1)

        return masks

def getFieldsMask(masks):
        tempMask = masks[0]
        for mask in masks[1:]:
                tempMask += mask
        return tempMask

if __name__ == "__main__":
    
    prev = cv2.imread('version2.png')
    FieldNumbering, FieldDescription = get_grid(prev)
    masks = createMaskFieldBoardExistance(FieldNumbering)
    cv2.imshow("mask wow", masks[1])
    cv2.waitKey(0)
    
