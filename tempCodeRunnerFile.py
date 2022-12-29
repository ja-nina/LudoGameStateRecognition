import cv2 
import numpy as np
import random
import itertools
import matplotlib.pyplot as plt 
from static_recognition import find_squares, get_grid, createDescriptiveBoard, createMaskFieldBoardExistance, getShadowMask, getFieldsMask
from dynamic_recognition import four_point_transform, get_token_placement, get_token_color_groups
from config import *
from itertools import compress
from random import randint
from sklearn import cluster
trackerTypes = ['BOOSTING']#, 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

playersPlaying = None
paramsDice = cv2.SimpleBlobDetector_Params()

# Filter by Area
paramsDice.filterByArea = True
paramsDice.minArea = 30
paramsDice.maxArea = 100
# Filter by Circularity
paramsDice.filterByCircularity = True
paramsDice.minCircularity = 0.6
# Filter by Convexity
paramsDice.filterByConvexity = True
paramsDice.minConvexity = 0.6
# Filter by Inertia
paramsDice.filterByInertia = True
paramsDice.minInertiaRatio = 0.6

lower = (0,0,0)  #130,150,80 # hard set
upper = (0,0,0) #250,250,120

class Game(object):
        def __init__(self):
                self._image = None
                self._text = []
                self._dice = 0
                self._players = []
                self._diceSeries = [0 for i in range(7)]
                self._noHands = True
                self._noHandsSeries = [True for i in range(5)]
                self._tokenPlacements = [[],[]]
                self._Events = ""
        
        @property 
        def text(self):
                return self._text
        @text.setter
        def text(self, text):
                self._text = text
        @property 
        def image(self):
                return self._image               
    
        @image.setter
        def image(self, image):
                self._image = image
        

        @property 
        def dice(self):
                if self.diceSeries.count(self.diceSeries[0]) == len(self.diceSeries):
                        self._dice = self.diceSeries[0]
                return self._dice
                
        @dice.setter
        def dice(self, dice):
                self._dice = dice
                
        @property 
        def diceSeries(self):
                return self._diceSeries
                
        @diceSeries.setter
        def diceSeries(self, diceSeries):
                self._diceSeries = diceSeries
        
        @property      
        def players(self):
                return self._players
        
        
        @players.setter
        def players(self, players):
                self._players = players
                
        def addDiceValue(self, diceValue):
                diceSeries = self.diceSeries
                diceSeries.append(diceValue)
                diceSeries.pop(0)
                self.diceSeries = diceSeries
        
        @property      
        def noHands(self):
                if self.noHandsSeries.count(self.noHandsSeries[0]) == len(self.noHandsSeries) and self.noHandsSeries[0] == True:
                        self._noHands = self.noHandsSeries[0]
                else:
                        self._noHands = False
                return self._noHands
        
        
        @noHands.setter
        def noHands(self, noHands):
                self._noHands = noHands
                
        @property 
        def noHandsSeries(self):
                return self._noHandsSeries
                
        @noHandsSeries.setter
        def noHandsSeries(self, noHandsSeries):
                self._noHandsSeries = noHandsSeries
        
        def addnoHandsValue(self, noHands):
                noHandsSeries = self.noHandsSeries
                noHandsSeries.append(noHands)
                noHandsSeries.pop(0)
                self.noHandsSeries = noHandsSeries
                
        @property
        def tokenPlacements(self):
                return self._tokenPlacements # is a list - player 1 has some placements [0] and player 2 others [1]

        
        @tokenPlacements.setter
        def tokenPlacements(self, tokenPlacements):
                self._tokenPlacements = tokenPlacements
                
        @property
        def Events(self):
                return self._Events
    
        @Events.setter
        def Events(self, Events):
                self._Events = Events
    
        def checkForTokenPlacementEventKill(self, supposedNewTokenPlacements):
                # make sure to use fields 1- 40 only
                #old new
                possibleKills = dict([((element[0], element[1]), (element[1] - element[0]) % 40) for element in itertools.product(self.tokenPlacements[0], self.tokenPlacements[1]) if (element[1] - element[0]) % 40 <= 6]
                                     + [((element[0], element[1]), (element[1] - element[0]) % 40) for element in itertools.product(self.tokenPlacements[1], self.tokenPlacements[0]) if (element[1] - element[0]) % 40 <= 6] )
                reallyPossibleKills = dict([(posKillKey[0], posKillKey[1]) for posKillKey, posKillvalue in possibleKills.items() if posKillvalue == self.dice])
                
                p1_tokensOld = self.tokenPlacements[0]
                p1_tokensSupposed = supposedNewTokenPlacements[0]
                
                p2_tokensOld = self.tokenPlacements[1]
                p2_tokensSupposed = supposedNewTokenPlacements[1]

                differenceP1 = list(set(p1_tokensSupposed) - set(p1_tokensOld)) + list(set(p1_tokensOld) - set(p1_tokensSupposed))
                differenceP2 = list(set(p2_tokensSupposed) - set(p2_tokensOld)) + list(set(p2_tokensOld) - set(p2_tokensSupposed))
                
                differenceP1revised = [element for element in differenceP1 if element < 41]
                differenceP2revised = [element for element in differenceP2 if element < 41]
                
                movements = [movement for movement in reallyPossibleKills if (movement[0] in differenceP1revised and movement[1] in differenceP1revised) 
                             or (movement[0] in differenceP2revised and movement[1] in differenceP2revised) ]
                
                kills = [movement for movement in reallyPossibleKills if (movement[0] in differenceP1revised and movement[1] in differenceP2revised) 
                             or (movement[0] in differenceP2revised and movement[1] in differenceP1revised) ]
                
                self.tokenPlacements = supposedNewTokenPlacements
                
                print('movements', movements)
                print('kills', kills)

                self.Events = "Movements: " + len(movements), " Kills: " + len(kills)

                
                
    
def get_blobsDice(frame, detectorDice):
    frameCopy = frame.copy()
    frameBlurred = cv2.medianBlur(frameCopy, 3)
    frameGray = cv2.cvtColor(frameBlurred, cv2.COLOR_BGR2GRAY)
    blobs = detectorDice.detect(frameGray)
    return blobs, frameGray

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


def overlay_info(frameOriginal, dice, blobs):
    frame = frameOriginal.copy()
    if len(dice) > 0:
        dice = dice[0]
        # Overlay dice number
        textsize = cv2.getTextSize(
                str(dice[0]), cv2.FONT_HERSHEY_PLAIN, 3, 2)[0]

        cv2.putText(frame, str(dice[0]),
                        (int(dice[1] - textsize[0] / 2),
                        int(dice[2] + textsize[1] / 2)),
                        cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        for blob in blobs:
                cv2.circle(frame, (int(blob.pt[0]),int(blob.pt[1])), 3, (0, 255, 0), -1)
        return frame
        
        
def getDiceValue(frame):
    blobs, _ = get_blobsDice(frame, detectorDice)
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
                difference = cv2.subtract(thresh, prev)
                mask[:,:,0] = difference

                image[mask[:,:,:] > 10] = 255

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

def getRotationNeeded(board):
        '''
        done on aleady resized board 500 x 500
        determines if 'original' board is rotated
        
        result - how to rotate board
        '''
        results = [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]
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
        h = board_hsv[:,:,0]
        
        sums = [np.sum(h[temp_starting_point1[1]:temp_endpoint1[1], temp_starting_point1[0]:temp_endpoint1[0]]),
                np.sum(h[temp_starting_point2[1]:temp_endpoint2[1], temp_starting_point2[0]:temp_endpoint2[0]]),
                np.sum(h[temp_starting_point3[1]:temp_endpoint3[1], temp_starting_point3[0]:temp_endpoint3[0]]),
                np.sum(h[temp_starting_point4[1]:temp_endpoint4[1], temp_starting_point4[0]:temp_endpoint4[0]])]

        whichIsRed = np.argmax(sums)
        
        return results[whichIsRed]

def shadowCorrection(image, prev):
        '''
        returns new prev adjusted to shadow and score for hands moving, 
        if any of those scores differ significantly by hue (especially mask) then we know snth is up (hands moving)
        
        -> this turned out to be quite a bad idea
        '''
def calculateBoardAnimation(masks, playersInCorners, reds, blues, yellows, greens, players, FieldNumberingToKeypoints, preliminaryFilled):
        '''
        1. Render animation of current state of board
        2. Display text
        
        @return image of animationq
        '''
        height,width = 500,500
        tokens = [reds, blues, yellows, greens]
        # ANIMATION

        if preliminaryFilled is None:
                tempMask = getFieldsMask(masks)
                #boardAnimated = np.zeros((height,width,3), np.uint8)
                #boardAnimated[ :, :, :] = (255,255,255)
                boardAnimated = cv2.imread('backgrounds/baldDog.png') # make it more fun
                boardAnimated[tempMask > 0, :] = beigeColor
                for i, field in FieldNumberingToKeypoints.items():
                        cv2.circle(boardAnimated, (int(field.pt[0]), int(field.pt[1])), 15, (0,0,0), 2)
                        if i < 41:
                                startX = field.pt[0] - 9
                                if i < 10:
                                        startX += 4
                                cv2.putText(boardAnimated, str(i), (int(startX), int(field.pt[1] + 4)), cv2.FONT_HERSHEY_TRIPLEX, 0.5,(0,0,0),1)
                
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
        
        if len(possibleKillsPlayer1) > 0:
                textToDisplay.append('Tips for ' + playerNames[0] +" : ")
                for possibleKillTuple, diceValue in possibleKillsPlayer1.items():
                        textToDisplay.append("If dice:" + str(diceValue) + " from " + possibleKillTuple[0] + " to " + possibleKillTuple[1])
        
        if len(possibleKillsPlayer2) > 0:
                textToDisplay.append('Tips for ' + playerNames[1] +" : ")
                for possibleKillTuple, diceValue in possibleKillsPlayer2.items():
                        textToDisplay.append("If dice:" + str(diceValue) + " from " + possibleKillTuple[0] + " to " + possibleKillTuple[1])

        
        print("text", textToDisplay)
        return playerNames[0],playerNames[1], textToDisplay

def displayGame(masks, playersInCorners, reds, blues, yellows, greens, dice, handsMoving, players, lastMove, FieldNumberingToKeypoints, preliminaryFilled, Game):
        w,h = 1000, 600
        
        startBoardX, startStartY = 50, 50
        if handsMoving:
                player1, player2 = Game.players
                textList = Game.text
                boardAnimated = Game.image

                
        else:
                player1, player2, textList = calculateStatesText(masks, playersInCorners, reds, blues, yellows, greens, dice, handsMoving, players, lastMove)
                Game.text = textList
                boardAnimated, _ =  calculateBoardAnimation(masks, playersInCorners, reds, blues, yellows, greens, players,  FieldNumberingToKeypoints, preliminaryFilled)
                Game.image = boardAnimated
                Game.players = [player1, player2]
                Game.addDiceValue(dice)
        
        bw, wh = boardAnimated.shape[0], boardAnimated.shape[1]

        wholeDisplay = np.zeros((h,w,3), np.uint8)
        wholeDisplay[startBoardX: startBoardX + bw, startStartY: startStartY + wh] = boardAnimated
        
        pixelsNextLine = 30
        
        cv2.putText(wholeDisplay, player1.upper() + "  VS  "+ player2.upper(), (600,50), cv2.FONT_HERSHEY_TRIPLEX, 1,(203,192,255),2)
        
        if handsMoving:
            cv2.putText(wholeDisplay, "Moving, wait for update...", (600,100 ), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255,255,255),2)
        for i, textPiece in enumerate(textList, start = 1):
                cv2.putText(wholeDisplay, textPiece, (600,100 + i*pixelsNextLine), cv2.FONT_HERSHEY_TRIPLEX, 0.5,(255,255,255),1)
        
        dice = Game.dice
        if dice > 0:
            dicePic = dicePics[dice - 1]
            startX, startY = 850, 130
            wholeDisplay[startY: startY + 90, startX: startX + 90, :] = dicePic
        
        cv2.imshow("whole", wholeDisplay)
        return Game
    

def setPlayers(red, blues, yellows, greens):
        resultPlayes = [False, False, False, False]
        playingPlayersIndices = list(dict(sorted(dict([(len(player) + random.random()/10,i) for i, player in enumerate([red, blues, yellows, greens])]).items())).values())[-2:]
        for player in playingPlayersIndices:
                resultPlayes[player] = True
        return resultPlayes

def skinDetection(image):
        imageShow = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        maskSkin = cv2.inRange(imageShow, lowerSkin2, upperSkin2) + cv2.inRange(imageShow, lowerSkin1, upperSkin1)
        imageShow = cv2.cvtColor(cv2.cvtColor(imageShow, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)
        imageShow[maskSkin < 100] = 0
        imageShow = cv2.resize(imageShow, (image.shape[1]//2, image.shape[0]//2))
        cv2.imshow("mask on skin", imageShow)
        print("SCORE HANDS: ", np.sum(imageShow[:,:]))
        return np.sum(imageShow[:,:])


if __name__ == "__main__":
    #needs to be initiated
    Game = Game()
    detectorDice = cv2.SimpleBlobDetector_create(paramsDice)
    colors = []
    for i in range(15):
        colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
    cap = cv2.VideoCapture(input_video_path_hard)
    boxes = dict()
    board_box = dict()
    recalculate_board = True
    
    ok, frame = cap.read()
    
    frame_2,  _, bboxes = find_squares(frame, boxes)
    board = four_point_transform(frame, list(bboxes.values())[0])
    rotation = getRotationNeeded(board)
    boardRotated = cv2.rotate(board,rotation)
    frame_2,  _, bboxes = find_squares(frame, boxes)
    scale_percent = 50 # percent of original size
    res = cv2.waitKey(1)
    
    prev = cv2.imread('version2.png')
    dim = (500,500)
    FieldNumbering, FieldDescription = get_grid(prev)
    maskFieldExistance = createMaskFieldBoardExistance(FieldNumbering)
    maskFieldsGlobal = getFieldsMask(maskFieldExistance).copy()
    maskFieldDescription = createDescriptiveBoard(FieldNumbering)
    prev = cv2.resize(prev, dim, interpolation = cv2.INTER_AREA)
    multiTracker = cv2.MultiTracker_create()
    # Initialize MultiTracker
    for bbox in bboxes.values():
        for corner in bbox:
                for trackerType in trackerTypes:
                        multiTracker.add(createTrackerByName(trackerType), frame, (corner[0] - WIDTH_BOXIE , corner[1] -WIDTH_BOXIE, WIDTH_BOXIE*2, WIDTH_BOXIE*2))
    while(cap.isOpened()):
        noHands = True

        ret, frame = cap.read() 
        frame_original_resized = cv2.resize(frame, (400,700), interpolation = cv2.INTER_AREA)
        dice = getDiceValue(frame) # check if dice placement is same as compared to the four points od the board
        scoreSkinDetection = skinDetection(frame)
        Game.addnoHandsValue(scoreSkinDetection  < 2000000)
        noHands = Game.noHands
        print("NO HANDS", noHands)

        if noHands: # value set by trial and error
                success, boxes = multiTracker.update(frame)
        
        
        frameresize = cv2.resize(frame, (int(frame.shape[1]*0.6), int(frame.shape[0]*0.6)))
        cv2.imshow('frame', frameresize)
        # cv2.waitKey(0)
        if ok:
                for i, newbox in enumerate(boxes):
                        p1 = (int(newbox[0]), int(newbox[1]))
                        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                        cv2.rectangle(frame, p1, p2, colors[i], 2, 1)
        else :
                cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

        
        board = four_point_transform(frame, boxes)
        boardRotated = cv2.rotate(board, rotation)
        width = 500
        height = 500
        dim = (width, height)
        boardRotatedResized = cv2.resize(boardRotated, dim, interpolation = cv2.INTER_AREA)
        if prev is not None:
                shadow = getShadowMask(boardRotatedResized, prev, maskFieldsGlobal)
                prevShadowAccounted = cv2.addWeighted(prev, 1, shadow, -1, 0)
                prevShadowAccounted = prev
                boardRotatedResized = boardRotatedResized
                difference2 = cv2.subtract(prevShadowAccounted, boardRotatedResized )
                difference = cv2.cvtColor(difference2 , cv2.COLOR_BGR2GRAY)
                blobous_difference, tokensPlacement = get_token_placement(difference, maskFieldExistance, maskFieldDescription)
                reds, blues, yellows, greens = get_token_color_groups(boardRotatedResized,tokensPlacement,blobous_difference, maskFieldExistance)
                if playersPlaying is None:
                        playersPlaying = setPlayers(reds,blues,yellows,greens)
                Game. heckForTokenPlacementEventKill(list(compress([reds, blues, yellows, greens], playersPlaying)))
                cv2.imshow('d2', cv2.cvtColor(cv2.subtract( boardRotatedResized, prevShadowAccounted ) , cv2.COLOR_BGR2GRAY))
                handsMoving = not noHands
                Game = displayGame(maskFieldExistance, None, reds, blues, yellows, greens, dice, handsMoving, playersPlaying, None, FieldNumbering, None, Game)
        res = cv2.waitKey(1)

        # Stop if the user presses "q"
        if res & 0xFF == ord('q'):
            break


    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()