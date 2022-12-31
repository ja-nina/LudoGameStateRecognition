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
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
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
        rect = np.zeros((4, 2), dtype = "float32")
        s = pts.sum(axis = 1)
        rect[0] = pts[np.argmin(s)][:2] + [WIDTH_BOXIE, WIDTH_BOXIE]
        rect[2] = pts[np.argmax(s)][:2] + [WIDTH_BOXIE, WIDTH_BOXIE]
        diff = np.diff(pts, axis = 1)
        rect[1] = pts[np.argmin(diff[:,0])][:2] + [WIDTH_BOXIE, WIDTH_BOXIE]
        rect[3] = pts[np.argmax(diff[:,0])][:2] + [WIDTH_BOXIE, WIDTH_BOXIE]
        return rect

def get_token_placement(frame, maskFieldExistances, maskFieldDescriptions, isPlayersSet, title = " default gray blob token detection on difference"):
        frame_copy = frame.copy()
        frame_blurred = cv2.medianBlur(frame_copy, 3)
        frame_gray = frame_blurred
        erosion = frame_gray
        resized_eroded = cv2.resize(erosion,(500,500), interpolation = cv2.INTER_AREA)
        
        scoresForFields = dict()
        for i, maskField in enumerate(maskFieldExistances, start=1):
            factor = 1
            if i > 40:
                factor = 1.5 # cince hones and bases are mor tricky to detect
            scoresForFields[i] = np.sum(resized_eroded[maskField == 255])*factor + random.random() 
            
        indicesTaken = 8
        if isPlayersSet:
            indicesTaken = 20
        temp = sorted(scoresForFields.items(),  key=lambda x: x[1])[-indicesTaken:]
        
        print("token scores", temp)

        indicesOfFields = reversed(list(dict(temp).keys()))
        return resized_eroded , indicesOfFields
    
    
def get_token_color_groups(board, tokenFields, difference, masksExistance):
    overallColors = dict()
    reds = []
    blues = []
    yellows = []
    greens = []
    boardHsv = cv2.cvtColor(board, cv2.COLOR_BGR2HSV)
    h = boardHsv[:,:,0]
    for tokenField in tokenFields:
        wights = difference[masksExistance[tokenField- 1] > 0]
        max_weight = max(wights) + 1
        normalized_weights = np.array([i / max_weight for i in wights])
        colors = h[masksExistance[tokenField - 1] > 0]
        colors = np.array([color if color > 10 else 180 for color in colors]) # else red is super imbalanced
        overallColors[tokenField] = np.sum(colors * normalized_weights)/ sum(normalized_weights)

    reds, blues, yellows, greens = find_color(overallColors, reds, blues, yellows, greens) # all colors with  probaliblity highest at end
    return reds, blues, yellows, greens


def find_color(colorHues, reds, blues, yellows, greens):
    
    for token_field, colorHue in list(colorHues.items()):
        if int(colorHue) in hsvRedRange1 or int(colorHue) in hsvRedRange2 and len(reds) < 4:
            reds.append(token_field)   
        elif int(colorHue) in hsvBlueRange and len(blues) < 4:
            blues.append(token_field)  
        elif int(colorHue) in hsvYellowRange and len(yellows) < 4:
            yellows.append(token_field)
        elif int(colorHue) in hsvGreenRange and len(greens) < 4:
            greens.append(token_field)
            
    return reds, blues, yellows, greens
        

