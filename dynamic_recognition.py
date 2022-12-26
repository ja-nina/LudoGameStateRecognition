# goal of this file is to basically see where tokens are
import cv2
import numpy as np
import random
from config import *
def four_point_transform(image, pts):
    '''
    Transforms photo of board in a weird angle into nice board that can be compared to field placements of original
    '''
    rect = order_points(pts)
    print("rect: ",  rect)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


def order_points(pts):
        print(pts)
        rect = np.zeros((4, 2), dtype = "float32")
        s = pts.sum(axis = 1)
        print("lol", s, np.argmin(s), pts[1][:2])
        rect[0] = pts[np.argmin(s)][:2] + [WIDTH_BOXIE, WIDTH_BOXIE]
        rect[2] = pts[np.argmax(s)][:2] + [WIDTH_BOXIE, WIDTH_BOXIE]
        diff = np.diff(pts, axis = 1)
        print(diff[:,0], np.argmin(diff))
        rect[1] = pts[np.argmin(diff[:,0])][:2] + [WIDTH_BOXIE, WIDTH_BOXIE]
        rect[3] = pts[np.argmax(diff[:,0])][:2] + [WIDTH_BOXIE, WIDTH_BOXIE]
        return rect

def get_token_placement(frame, maskFieldExistances, maskFieldDescriptions, title = " default gray blob token detection on difference"):
        frame_copy = frame.copy()
        frame_blurred = cv2.medianBlur(frame_copy, 3)
        frame_gray = cv2.cvtColor(frame_blurred, cv2.COLOR_BGR2GRAY)
        mask = cv2.inRange(frame_gray, 60, 255)
        #mask = cv2.inRange(frame_hsv, lower_green_tokens, upper_green_tokens)
        #mask = cv2.inRange(frame_hsv, lower_red_tokens, upper_red_tokens)
        frame_gray[mask == 0] = 0
        # Creating maxican hat filter
        #filter = np.array([[0,0,-1,0,0],[0,-1,-2,-1,0],[-1,-2,16,-2,-1],[0,-1,-2,-1,0],[0,0,-1,0,0]])
        # Applying cv2.filter2D function on our Logo image
        #mexican_hat_img2=cv2.filter2D(frame_gray,-1,filter)
        t_lower = 20 
        t_upper = 250
        #canny = cv2.Canny(frame_gray, t_lower, t_upper)
        kernel = np.ones((3, 3), np.uint8)
        
        # erode the image
        #erosion = cv2.erode(frame_gray, kernel, iterations=1)
        erosion = frame_gray
        blobs = detector.detect(erosion)

        # overlaying info 
        for blob in blobs:
                pos = blob.pt
                r = blob.size / 2
                cv2.circle(erosion, (int(pos[0]), int(pos[1])), int(r), (255, 0, 0), 2)
                cv2.putText(erosion, str(blob.response),
                    (int(pos[0]), int(pos[1])),
                    cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        resized_eroded = cv2.resize(erosion,(500,500), interpolation = cv2.INTER_AREA)
        
        scoresForFields = dict()
        for i, maskField in enumerate(maskFieldExistances, start=1):
            scoresForFields[i] = np.sum(resized_eroded[maskField == 255]) + random.random()
        temp = sorted(scoresForFields.items(),  key=lambda x: x[1])[-8:]
        indicesOfFields = list(dict(temp).keys())
        print("indices_of fields: ", indicesOfFields)
        
        
        masks_temp = [maskFieldExistances[i - 1] for i in indicesOfFields]
        fields = masks_temp[0] + masks_temp[1] + masks_temp[2] + masks_temp[3] + masks_temp[4] + masks_temp[5] + masks_temp[6] + masks_temp[7]
        cv2.imshow("fields", fields)
        
        cv2.imshow("token_detection" +str(resized_eroded.shape) , resized_eroded)
        
        return resized_eroded , indicesOfFields
    
    
def get_token_color_groups(board, tokenFields, difference, masksExistance):
    overallColors = dict()
    reds = dict()
    blues = dict()
    yellows = dict()
    greens = dict()
    boardHsv = cv2.cvtColor(board, cv2.COLOR_BGR2HSV)
    h = boardHsv[:,:,0]
    for tokenField in tokenFields:
        wights = difference[masksExistance[tokenField- 1] == 255]
        max_weight = max(wights)
        normalized_weights = [i / max_weight for i in wights]
        colors = h[masksExistance[tokenField - 1] == 255]
        print("shape h", h.shape, " shape masksExistance", (masksExistance[tokenField - 1] == 255).shape)

        overallColors[tokenField - 1] = np.sum(colors * normalized_weights)/ sum(normalized_weights)
    print("colors", overallColors)
    return overallColors

        

