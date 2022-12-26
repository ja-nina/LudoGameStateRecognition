import cv2 
import numpy as np
import random
import matplotlib.pyplot as plt 
import itertools
from config import *
from random import randint
trackerTypes = ['BOOSTING']#, 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
def createTrackerByName(trackerType):
       # Create a tracker based on tracker name
        if trackerType == trackerTypes[0]:
                tracker = cv2.TrackerBoosting_create()
        elif trackerType == trackerTypes[1]:
                tracker = cv2.TrackerMIL_create()
        elif trackerType == trackerTypes[2]:
                tracker = cv2.TrackerKCF_create()
        elif trackerType == trackerTypes[3]:
                tracker = cv2.TrackerTLD_create()
        elif trackerType == trackerTypes[4]:
                tracker = cv2.TrackerMedianFlow_create()
        elif trackerType == trackerTypes[5]:
                tracker = cv2.TrackerGOTURN_create()
        elif trackerType == trackerTypes[6]:
                tracker = cv2.TrackerMOSSE_create()
        elif trackerType == trackerTypes[7]:
                tracker = cv2.TrackerCSRT_create()
        else:
                tracker = None
                print('Incorrect tracker name')
                print('Available trackers are:')
                for t in trackerTypes:
                        print(t)

        return tracker



def calculate_noise(img, prev, board_coords):
        # dodaj rozpoznawanie noisu w okolicach samej planszy
        # on noise on the board (hand) calculate if player is making a move
        # do it on masks - heuristic
        is_turned = False

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
        mask = np.zeros(image.shape, dtype='uint8')
        thresh = cv2.threshold(sharpen, 150, 255, cv2.THRESH_BINARY_INV)[1]
        mask[:,:,:] = 0
        if prev is None:
                is_turned = True
                difference = thresh
        else:
                print("1")
                difference = cv2.subtract(thresh, prev)
                mask[:,:,0] = difference
                print("2")
                print(mask[:,:,0])
                image[mask[:,:,:] > 10] = 255
                print("3")
                print("difference: ", cv2.countNonZero(difference))
        return is_turned, image, thresh


def calculate_hands(img, board_coords):
        # works -  TODO: add coords of  board and look if noise around it is prominent : if yess pass a variable that will account for noise
        # more frewquent recalculation - if no noise - no need for recalculation, especially noise around the corners
        # hands on board should indicate if players do turns right now.
        # on noise on the board (hand) calculate if player is making a move
        # do it on masks - heuristic
        is_turned = False
        lower = (20,0,0)  #130,150,80 # hard set
        upper = (130,255,255) #250,250,120

        lower2 = (0,90,0)  #130,150,80 # hard set
        upper2 = (180,190,255) #250,250,120
        image = img.copy()

        blur = cv2.medianBlur(image, 5)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, lower, upper)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        
        mask_final = cv2.bitwise_and(cv2.bitwise_not(mask1), mask2)
        image[mask_final > 0] = 0
        suma = mask_final.sum()
        cv2.putText(image, " Sum of hands: " + str(suma),
                   (int(200), int(200)),
                    cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

        #img[cv2.bitwise_not(mask1) > 0] = 0
        return image, suma

def calculate_board_misplacement(image, board_coords):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 5)
        sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpen = cv2.filter2D(blur, -1, sharpen_kernel)

        # Threshold and morph close
        thresh = cv2.threshold(sharpen, 100, 255, cv2.THRESH_BINARY_INV)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        #close = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, kernel, iterations=1) # good for token spotting
        close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        return close

        # or of masks will be off


def calculate_bases(image):
        pass


def add_tuples(t1,t2):
        return tuple([e1+e2 for e1,e2 in zip(t1,t2)])
def get_corner_players(board):
        '''
        done on aleady resized board 500 x 500
        '''
        print("Shape of board: ", board.shape)
        width_board, heigh_board, board_depth  = board.shape
        # parametrers 
        margin_t = 20
        margin_b = 20
        margin_l = 20
        margin_r = 20
        width_base = 90
        heigh_base = 90
        
        box = (width_base, heigh_base)
        temp_starting_point1 = (margin_l, margin_t)
        temp_endpoint1 = add_tuples(temp_starting_point1, box)
        cv2.rectangle(board, temp_starting_point1, temp_endpoint1, red, 2, 1)
        temp_starting_point2 = (width_board - width_base - margin_r, margin_t) 
        temp_endpoint2 = add_tuples(temp_starting_point2, box)
        cv2.rectangle(board, temp_starting_point2, temp_endpoint2,blue, 2, 1)
        temp_starting_point3 = (margin_l, heigh_board - heigh_base - margin_b)
        temp_endpoint3 = add_tuples(temp_starting_point3, box)
        cv2.rectangle(board, temp_starting_point3, temp_endpoint3, yellow, 2, 1)
        temp_starting_point4 = (width_board - width_base - margin_r, heigh_board - heigh_base - margin_b)
        temp_endpoint4 = add_tuples(temp_starting_point4, box)
        cv2.rectangle(board, temp_starting_point4, temp_endpoint4, green, 2, 1)
        
        

        board_hsv = cv2.cvtColor(board, cv2.COLOR_BGR2HSV)
        #cv2.imshow(" Trying bboxes on bases", board)
        colors_t = ('red','blue', 'yellow', 'green')
        h = board_hsv[:,:,0]
        # histred = cv2.calcHist([h[temp_starting_point1[1]:temp_endpoint1[1], temp_starting_point1[0]:temp_endpoint1[0]]],[0],None,[256],[0,256])
        # plt.plot(histred,color = colors_t[0])
        
        # histblue = cv2.calcHist([h[temp_starting_point2[1]:temp_endpoint2[1], temp_starting_point2[0]:temp_endpoint2[0]]],[0],None,[256],[0,256])
        # plt.plot(histblue,color = colors_t[1])
        
        # histyellow = cv2.calcHist([h[temp_starting_point3[1]:temp_endpoint3[1], temp_starting_point3[0]:temp_endpoint3[1]]],[0],None,[256],[0,256])
        # plt.plot(histyellow,color = colors_t[2])
        
        # histgreen = cv2.calcHist([h[temp_starting_point4[1]:temp_endpoint4[1], temp_starting_point4[0]:temp_endpoint4[1]]],[0],None,[256],[0,256])
        # plt.plot(histgreen,color = colors_t[3])
        
        print("Sum red: ", np.sum(h[temp_starting_point1[1]:temp_endpoint1[1], temp_starting_point1[0]:temp_endpoint1[0]]))
        print("Sum green: ", np.sum(h[temp_starting_point4[1]:temp_endpoint4[1], temp_starting_point4[0]:temp_endpoint4[0]]))
        print("Sum yellow: ", np.sum(h[temp_starting_point3[1]:temp_endpoint3[1], temp_starting_point3[0]:temp_endpoint3[0]]))
        print("Sum blue: ", np.sum(h[temp_starting_point2[1]:temp_endpoint2[1], temp_starting_point2[0]:temp_endpoint2[0]]))


        return board

def get_who_plays():
        pass

def get_tokens(frame, title = " default gray blob token detection on difference"):
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

        #cv2.imshow("token_detection", resized_eroded)
        
        return frame_copy


def calculate_circles(image):

        pass

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

def four_point_transform(image, pts):
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

        
if __name__ == "__main__":
    colors = []
    for i in range(15):
        colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
    cap = cv2.VideoCapture(input_video_path)
    boxes = dict()
    print(cv2.__version__)
    board_box = dict()
    recalculate_board = True
    
    ok, frame = cap.read()
    frame_2,  _, bboxes = find_squares(frame, boxes)
    scale_percent = 50 # percent of original size

    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)

    resized = cv2.resize(frame_2, dim, interpolation = cv2.INTER_AREA)
    #cv2.imshow("frame", resized)
        
    res = cv2.waitKey(1)
    
    prev = cv2.imread('version2.png')
    get_grid(prev)
    dim = (500,500)
    prev = cv2.resize(prev, dim, interpolation = cv2.INTER_AREA)
    #get_corner_players(prev)
    #bbox = (88, 68, 886, 920)
    #multiTracker = cv2.MultiTracker_create()
    # Initialize MultiTracker
    #for bbox in bboxes.values():
    #    for corner in bbox:
    #            print(" corner: ", (corner[0], corner[1] , corner[0] + 20, corner[1] + 20))
    #            for trackerType in trackerTypes:
    #                    multiTracker.add(createTrackerByName(trackerType), frame, (corner[0] - WIDTH_BOXIE , corner[1] -WIDTH_BOXIE, 40, 40))
    while(cap.isOpened()):


        ret, frame = cap.read()
        
        frame_2, score = calculate_hands(frame, "lol")
        if score < 15000000:
                success, boxes = multiTracker.update(frame)
        print("boxes: ", boxes)
        
        timer = cv2.getTickCount()
        fps = cv2.getTickFrequency() / (cv2.getTickCount())
        if ok:
                for i, newbox in enumerate(boxes):
                        p1 = (int(newbox[0]), int(newbox[1]))
                        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                        cv2.rectangle(frame, p1, p2, colors[i], 2, 1)
        else :
                cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
        cv2.putText(frame, " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)
        cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)

        
        if recalculate_board:
                #frame2, tresh, boxes = find_squares(frame, boxes)
                pass
        #is_turned, frame2, prev = calculate_noise(frame, prev)
        #frame2 = calculate_hands(frame)
        
        # get mask
        #frame2 = get_blobs(frame)
        
        print("transform")
        frame = four_point_transform(frame, boxes)
        frame_w_blobs, _ = get_blobs(frame)


        scale_percent = 50 # percent of original size

        width = 500
        height = 500
        dim = (width, height)

        resized_no_blobs = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        get_corner_players(resized_no_blobs)
        resized_blobs = cv2.resize(frame_w_blobs, dim, interpolation = cv2.INTER_AREA)
        
        #cv2.imshow("frame with blobs", resized_blobs)
        if prev is not None:
                #print(f.shape, prev.shape)
                difference1 = cv2.subtract(resized_no_blobs, prev )
                difference2 = cv2.subtract(prev, resized_no_blobs )
                difference = difference2
                # color the mask red
                Conv_hsv_Gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
                ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255, cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
                
                #difference[mask != 255] = [0, 0, 255]
        
                #resized2 = cv2.resize(difference, dim, interpolation = cv2.INTER_AREA)
                blobous_difference = get_tokens(difference)
                #cv2.imshow("difference", blobous_difference)

        res = cv2.waitKey(1)

        # Stop if the user presses "q"
        if res & 0xFF == ord('q'):
            break


    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()