import cv2
import numpy as np
input_video_path = 'videos/IMG_1912.mp4'

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

WIDTH_BOXIE = 20
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


hsvGreenRange = (40,80)
hsvYellowRange = (25,35)
hsvBlueRange = (105,130)
hsvRedRange1 = (0,10)
hsvRedRange2 = (150,180)

