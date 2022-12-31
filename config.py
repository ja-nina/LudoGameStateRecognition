import cv2
import numpy as np
input_video_path = 'videos/IMG_1912.mp4'
input_video_path_easy1 = 'videos/easy.mp4'
input_video_path_easy2 = 'videos/IMG_1912.mp4'
input_video_path_shadow = 'videos/weirdShadow.mp4'
input_video_path_hard = 'videos/hard.mp4'

# HEURISTIC: TODO ADJUST TO BOARD SIZE
noise_level_non_turn = 3000

# Initialize a video feed
params = cv2.SimpleBlobDetector_Params()

# Filter by Area
params.filterByArea = True
params.minArea = 1000

lower_saturation = (0,60,60)  #130,150,80 # hard set
upper_saturation = (15,255,255) #250,250, 120

lower_yellow = np.array([25, 0, 0], dtype="uint8")
upper_yellow = np.array([35, 255, 255], dtype="uint8")

lower_green_tokens = np.array([40, 80, 0], dtype="uint8")
upper_green_tokens = np.array([80, 255, 255], dtype="uint8")

lower_red_tokens = np.array([170, 150, 0], dtype="uint8")
upper_red_tokens = np.array([180, 255, 255], dtype="uint8")

HANDS_OCCURANCE_THERESHOLD = 15000000

WIDTH_BOXIE = 30

R_CIRCLE_FIELD = 36
R_CIRCLE_ACCEPTANCE_THERESHOLD = 0.1
R_TOKEN = 15
#params.maxArea = 800
# # Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.1
# # Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.05
# # Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.05

lower = (50,0,0)  #130,150,80 # hard set
upper = (200,200,200) #250,250,120

detector = cv2.SimpleBlobDetector_create(params)

red = (0,0,250)
blue = (250,0,0)
green = (0,250,0)
yellow = (0,255,255)




redHomeOffset = 40
redBaseOffset = 56


hsvGreenRange = range(40,80)
hsvYellowRange = range(25,35)
hsvBlueRange = range(80,130)
hsvRedRange1 = range(0,10)
hsvRedRange2 = range(145,180)

playableColors = ['  red', ' blue', 'yellow', ' green']

# red, blue, yellow, green
basesFieldNumbers = [tuple(range(41,45)), tuple(range(45, 49)), tuple(range(49, 53)), tuple(range(53,57))]
homesFieldNumbers = [range(57,61), range(61, 65), range(65, 69), range(69, 73)]
regulatFieldNUmbers = range(1,41)
startingFields = [33,3,23,13]


# ANIMATION : 
beigeColor = (203, 215, 223)
# red, blue, yellow, greenq
tokenColors = [(0,0,255),(255,0,0), (0,255,255), (0,255,0)]
specialFieldColors = [(132,135,240),(195,175,122), (137,174,225), (158,200,178)]
ROfToken = 10
heartStencilTemp = cv2.cvtColor(cv2.cv2.imread('heartStencil.png'), cv2.COLOR_BGR2GRAY)
heartStencil = cv2.threshold(heartStencilTemp, 0, 150, cv2.THRESH_BINARY_INV)[1]

dicePics = [cv2.imread('dice/' + str(i) + '.png') for i in range(1,7)]

maskOnBases = cv2.imread('maskOnBases.png', cv2.IMREAD_GRAYSCALE)


lowerSkin1 = np.array([0, 0, 0], dtype = "uint8")
upperSkin1 = np.array([7, 255, 255], dtype = "uint8")

lowerSkin2 = np.array([160, 0, 0], dtype = "uint8")
upperSkin2 = np.array([180, 255, 255], dtype = "uint8")

