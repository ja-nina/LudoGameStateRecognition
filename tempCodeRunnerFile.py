import cv2 
import numpy as np
import random
import itertools
import matplotlib.pyplot as plt 
from static_recognition import find_squares, get_grid, createDescriptiveBoard, createMaskFieldBoardExistance
from dynamic_recognition import four_point_transform, get_token_placement, get_token_color_groups
from config import *
from itertools import compress
from random import randint
from sklearn import cluster
trackerTypes = ['BOOSTING']#, 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']


paramsDice = cv2.SimpleBlobDetector_Params()

# Filter by Area
paramsDice.filterByArea = True
paramsDice.minArea = 50
paramsDice.maxArea = 80
# Filter by Circularity
paramsDice.filterByCircularity = True
paramsDice.minCircularity = 0.7
# Filter by Convexity
paramsDice.filterByConvexity = True
paramsDice.minConvexity = 0.7
# Filter by Inertia
paramsDice.filterByInertia = True
paramsDice.minInertiaRatio = 0.7

lower = (50,0,0)  #130,150,80 # hard set
upper = (200,200,200) #250,250,120


def get_blobsDice(frame, detectorDice):
    frameCopy = frame.copy()

    mask = cv2.inRange(frameCopy, lower, upper)
    frameCopy[mask > 0] = 0
    frame_blurred = cv2.medianBlur(frameCopy, 7)
    frame_gray = cv2.cvtColor(frame_blurred, cv2.COLOR_BGR2GRAY)
    blobs = detectorDice.detect(frame_gray)
    return blobs

def get_dice_from_blobs(blobs):
    X = np.asarray([b.pt for b in blobs if b.pt != None])
    if len(X) > 0:
        # Important to set min_sample to 0, as a dice may only have one dot
        clustering = cluster.DBSCAN(eps=40, min_samples=0).fit(X)
        dice = []
        # Calculate centroid of each dice, the average between all a dice's dots
        X_dice = X[clustering.labels_ == 0]
        centroid_dice = np.mean(X_dice, axis=0)
        dice.append([len(X_dice), *centroid_dice])

        return dice

    else:
        return []


def overlay_info(frame, dice, blobs):
    if len(dice) > 0:
        dice = dice[0]
        # Overlay dice number
        textsize = cv2.getTextSize(
                str(dice[0]), cv2.FONT_HERSHEY_PLAIN, 3, 2)[0]

        cv2.putText(frame, str(dice[0]),
                        (int(dice[1] - textsize[0] / 2),
                        int(dice[2] + textsize[1] / 2)),
                        cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        
def getDiceValue(frame):
    blobs = get_blobsDice(frame, detectorDice)
    dice = get_dice_from_blobs(blobs)
    if len(dice) == 0:
        return 0
    else: 
        return dice[0][0]
    
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
        determines if 'original' board is rotated
        
        result - how to rotate board
        '''
        #print("Shape of board: ", board.shape)
        results = [None, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180]
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
        
        # print("Sum red: ", np.sum(h[temp_starting_point1[1]:temp_endpoint1[1], temp_starting_point1[0]:temp_endpoint1[0]]))
        # print("Sum green: ", np.sum(h[temp_starting_point4[1]:temp_endpoint4[1], temp_starting_point4[0]:temp_endpoint4[0]]))
        # print("Sum yellow: ", np.sum(h[temp_starting_point3[1]:temp_endpoint3[1], temp_starting_point3[0]:temp_endpoint3[0]]))
        # print("Sum blue: ", np.sum(h[temp_starting_point2[1]:temp_endpoint2[1], temp_starting_point2[0]:temp_endpoint2[0]]))

        sums = [np.sum(h[temp_starting_point1[1]:temp_endpoint1[1], temp_starting_point1[0]:temp_endpoint1[0]]),
                np.sum(h[temp_starting_point2[1]:temp_endpoint2[1], temp_starting_point2[0]:temp_endpoint2[0]]),
                np.sum(h[temp_starting_point3[1]:temp_endpoint3[1], temp_starting_point3[0]:temp_endpoint3[0]]),
                np.sum(h[temp_starting_point4[1]:temp_endpoint4[1], temp_starting_point4[0]:temp_endpoint4[0]])]

        whichIsRed = np.argmax(sums)
        
        return results[whichIsRed]

def calculateBoardAnimation(masks, playersInCorners, reds, blues, yellows, greens, players, FieldNumberingToKeypoints, preliminaryFilled):
        '''
        1. Render animation of current state of board
        2. Display text
        
        @return image of animation
        '''
        height,width = 500,500
        tokens = [reds, blues, yellows, greens]
        # ANIMATION

        if preliminaryFilled is None:
                tempMask = masks[0]
                for mask in masks[1:]:
                        tempMask += mask
                
                #boardAnimated = np.zeros((height,width,3), np.uint8)
                #boardAnimated[ :, :, :] = (255,255,255)
                boardAnimated = cv2.imread('backgrounds/martyn.png') # make it more fun
                for i, field in FieldNumberingToKeypoints.items():
                        cv2.circle(boardAnimated, (int(field.pt[0]), int(field.pt[1])), 15, (0,0,0), 2)
                        
                boardAnimated[tempMask > 0, :] = beigeColor
                preliminaryFilled = boardAnimated
        
        filledBoard = preliminaryFilled.copy()
        heartStencilWidth, heartStencilHeight = heartStencil.shape
        for tokens, color, isPlayer in zip(tokens, tokenColors, players):
            if isPlayer is True:
                for token in tokens:
                        centerOfToken = FieldNumberingToKeypoints[token].pt
                        centerOfToken = (int(centerOfToken[1]), int(centerOfToken[0]))
                        startOfHeart = (centerOfToken[0] - heartStencilWidth//2, centerOfToken[1] - heartStencilHeight//2)
                        endOfHeart = (startOfHeart[0] + heartStencilWidth, startOfHeart[1] + heartStencilHeight)
                        filledBoard[startOfHeart[0]: endOfHeart[0], startOfHeart[1]: endOfHeart[1], :][heartStencil < 101, :] = color

        return filledBoard, preliminaryFilled



def calculateStatesText(masks, playersInCorners, reds, blues, yellows, greens, dice, handsMoving, players, lastMove):
        '''
        players = bool array indicating who is playing against who
        1. Calculation of dice rolled
        2. Move conducted
        3. If players can kill each others tokens!
        4. Tokens in base
        5. Tokens in home
        6. Which players play
        
        @return image of animation
        '''
        tokens = [reds, blues, yellows, greens]
        handsMoving = handsMoving
        playerNames = list(compress(playableColors, players))
        player1BaseCount, player2BaseCount = [len([token for token in color if token in base]) for color, base in list(zip(list(compress(tokens, players)), list(compress(basesFieldNumbers, players))))]
        player1HomeCount, player2HomeCount = [len([token for token in color if token in home]) for color, home in list(zip(list(compress(tokens, players)), list(compress(homesFieldNumbers, players))))]
        player1RegularCount, player2RegularCount  = [[token for token in color if token in regulatFieldNUmbers] for color in list(compress(tokens, players))]
        startingPlayer1, startingPlayer2 = list(compress(startingFields, players))
        possibleKillsPlayer1 = dict([(('field ' + str(element[0]), 'field ' + str(element[1])), (element[1] - element[0]) % 40) for element in itertools.product(player1RegularCount, player2RegularCount) if (element[1] - element[0]) % 40 <= 6])
        if startingPlayer1 in player2RegularCount and player1BaseCount > 0:
                possibleKillsPlayer1[('base', 'field ' + str(startingPlayer1))] = 6
                
        possibleKillsPlayer2 = dict([(('field ' + str(element[0]), 'field ' + str(element[1])), (element[1] - element[0]) % 40) for element in itertools.product(player2RegularCount, player1RegularCount) if (element[1] - element[0]) % 40 <= 6])
        if startingPlayer2 in player1RegularCount and player2BaseCount > 0:
                possibleKillsPlayer2[('base', 'field ' + str(startingPlayer2))] = 6
        
        textToDisplay = []

        textToDisplay.append("BASE " + playerNames[0] +" : " + str(player1BaseCount) + "/ 4")
        textToDisplay.append("HOME " + playerNames[0] +" : " + str(player1HomeCount) + "/ 4")
        textToDisplay.append("BASE " + playerNames[1] +" : " + str(player2BaseCount) + "/ 4")
        textToDisplay.append("HOME " + playerNames[1] +" : " + str(player2HomeCount) + "/ 4")
        
        print("text", textToDisplay)
        return playerNames[0],playerNames[1], textToDisplay

def displayGame(masks, playersInCorners, reds, blues, yellows, greens, dice, handsMoving, players, lastMove, FieldNumberingToKeypoints, preliminaryFilled):
        w,h = 1000, 600
        startBoardX, startStartY = 50, 50
        player1, player2, textList = calculateStatesText(masks, playersInCorners, reds, blues, yellows, greens, dice, handsMoving, players, lastMove)
        boardAnimated, preliminaryFilled =  calculateBoardAnimation(masks, playersInCorners, reds, blues, yellows, greens, players,  FieldNumberingToKeypoints, preliminaryFilled)
        bw, wh = boardAnimated.shape[0], boardAnimated.shape[1]

        wholeDisplay = np.zeros((h,w,3), np.uint8)
        wholeDisplay[startBoardX: startBoardX + bw, startStartY: startStartY + wh] = boardAnimated
        
        pixelsNextLine = 30
        
        cv2.putText(wholeDisplay, player1.upper() + "  VS  "+ player2.upper(), (600,50), cv2.FONT_HERSHEY_TRIPLEX, 1,(203,192,255),2)
        
        if handsMoving:
            cv2.putText(wholeDisplay, "Moving, wait for update...", (600,100 ), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255,255,255),2)
        for i, textPiece in enumerate(textList, start = 1):
                cv2.putText(wholeDisplay, textPiece, (600,100 + i*pixelsNextLine), cv2.FONT_HERSHEY_TRIPLEX, 0.5,(255,255,255),1)
                
        if dice > 0:
            dicePic = dicePics[dice - 1]
            cv2.imshow("dice", dicePic)
            print(dicePic.shape)
            startX, startY = 850, 130
            wholeDisplay[startY: startY + 90, startX: startX + 90, :] = dicePic
        
        cv2.imshow("whole", wholeDisplay)
    

def displayStatus():
        pass
        
if __name__ == "__main__":
    #needs to be initiated
    detectorDice = cv2.SimpleBlobDetector_create(paramsDice)
    colors = []
    for i in range(15):
        colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
    cap = cv2.VideoCapture(input_video_path)
    boxes = dict()
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
    dim = (500,500)
    FieldNumbering, FieldDescription = get_grid(prev)
    maskFieldExistance = createMaskFieldBoardExistance(FieldNumbering)
    maskFieldDescription = createDescriptiveBoard(FieldNumbering)
    prev = cv2.resize(prev, dim, interpolation = cv2.INTER_AREA)
    #get_corner_players(prev)
    bbox = (88, 68, 886, 920)
    multiTracker = cv2.MultiTracker_create()
    # Initialize MultiTracker
    for bbox in bboxes.values():
        for corner in bbox:
                print(" corner: ", (corner[0], corner[1] , corner[0] + 20, corner[1] + 20))
                for trackerType in trackerTypes:
                        multiTracker.add(createTrackerByName(trackerType), frame, (corner[0] - WIDTH_BOXIE , corner[1] -WIDTH_BOXIE, 40, 40))
    while(cap.isOpened()):
        noHands = False

        ret, frame = cap.read()
        blobs = get_blobsDice(frame, detectorDice)
        dice = get_dice_from_blobs(blobs)
        out_frame = overlay_info(frame, dice, blobs)
        frame_original_resized = cv2.resize(frame, (400,700), interpolation = cv2.INTER_AREA)
        dice = getDiceValue(frame) # check if dice placement is same as compared to the four points od the board
        frame_2, scoreMovingHands = calculate_hands(frame, "lol")
        if scoreMovingHands < HANDS_OCCURANCE_THERESHOLD: # value set by trial and error
                noHands = True
                success, boxes = multiTracker.update(frame)
                pass
        

        if ok:
                for i, newbox in enumerate(boxes):
                        p1 = (int(newbox[0]), int(newbox[1]))
                        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                        cv2.rectangle(frame, p1, p2, colors[i], 2, 1)
        else :
                cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

        if recalculate_board:
                #frame2, tresh, boxes = find_squares(frame, boxes)
                pass
        #is_turned, frame2, prev = calculate_noise(frame, prev)
        #frame2 = calculate_hands(frame)
        
        # get mask
        #frame2 = get_blobs(frame)
        
        frame = four_point_transform(frame, boxes)


        width = 500
        height = 500
        dim = (width, height)

        resized_no_blobs = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        #get_corner_players(resized_no_blobs)
        
        #cv2.imshow("frame with blobs", resized_blobs)
        if prev is not None:
                #print(f.shape, prev.shape)
                difference1 = cv2.subtract(resized_no_blobs, prev )
                difference2 = cv2.subtract(prev, resized_no_blobs )
                difference = difference2
                # color the mask red
                Conv_hsv_Gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
                ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255, cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
                blobous_difference, tokensPlacement = get_token_placement(difference, maskFieldExistance, maskFieldDescription)
                reds, blues, yellows, greens = get_token_color_groups(resized_no_blobs,tokensPlacement,blobous_difference, maskFieldExistance)
                
                for redToken in reds:
                        resized_no_blobs[maskFieldExistance[redToken - 1] == 255] = (0,0,255)
                for greenToken in greens:
                        resized_no_blobs[maskFieldExistance[greenToken - 1] == 255] = (0,255,0)
                for blueToken in blues:
                        resized_no_blobs[maskFieldExistance[blueToken - 1] == 255] = (255,0,0)
                for yellowToken in yellows:
                        resized_no_blobs[maskFieldExistance[yellowToken - 1] == 255] = (0,255,255)
                cv2.imshow("original frame", frame_original_resized)
                
                handsMoving = False
                if scoreMovingHands > HANDS_OCCURANCE_THERESHOLD*(1 + 0.5):
                    handsMoving = True
                displayGame(maskFieldExistance, [True, False, False, True], reds, blues, yellows,greens, dice, handsMoving, [True, False, False, True], None, FieldNumbering, None)

        res = cv2.waitKey(1)

        # Stop if the user presses "q"
        if res & 0xFF == ord('q'):
            break


    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()