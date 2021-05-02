import cv2
import numpy as np
import os

cams = ['../dataset/MachineHall01_reduced/cam0/data',
        '../dataset/MachineHall01_reduced/cam1/data']

videos = np.array(['../data_out/vids/video_0.avi','../data_out/vids/video_1.avi'], dtype=str)

for i, cam in enumerate(cams):
    images = [img for img in sorted(os.listdir(cam)) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(cam, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(videos[i], 0, 1, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(cam, image)))

    cv2.destroyAllWindows()
    video.release()

