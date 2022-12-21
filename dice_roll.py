import cv2
import numpy as np
from sklearn import cluster

input_video_path = 'videos/IMG_1912.mp4'

# Initialize a video feed

params = cv2.SimpleBlobDetector_Params()

# Filter by Area
params.filterByArea = True
params.minArea = 50
params.maxArea = 80
# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.7
# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.7
# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.7

lower = (50,0,0)  #130,150,80 # hard set
upper = (200,200,200) #250,250,120


def get_blobs(frame):
    mask = cv2.inRange(frame, lower, upper)
    frame[mask > 0] = 0
    frame_blurred = cv2.medianBlur(frame, 7)
    frame_gray = cv2.cvtColor(frame_blurred, cv2.COLOR_BGR2GRAY)
    blobs = detector.detect(frame_gray)
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

if __name__ == "__main__":
    cap = cv2.VideoCapture(input_video_path)
    detector = cv2.SimpleBlobDetector_create(params)
    while(cap.isOpened()):

        ret, frame = cap.read()

        blobs = get_blobs(frame)
        dice = get_dice_from_blobs(blobs)
        out_frame = overlay_info(frame, dice, blobs)

        scale_percent = 50 # percent of original size
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        cv2.imshow("frame", resized)
        
        res = cv2.waitKey(1)

        # Stop if the user presses "q"
        if res & 0xFF == ord('q'):
            break


    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()