import cv2
import numpy as np
import os

cams = ['../dataset/MachineHall01_reduced/cam0/data',
                  '../dataset/MachineHall01_reduced/cam1/data']

image_folder = '../dataset/MachineHall01_reduced/cam0/data'

video_name = '../data_out/vids/video.avi'
videos = np.array(['../data_out/vids/video_0.avi','../data_out/vids/video_1.avi'], dtype=str)


for i, cam in enumerate(cams):
    images = [img for img in os.listdir(cam) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(videos[i], 0, 1, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()










#################################################
# images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
# frame = cv2.imread(os.path.join(image_folder, images[0]))
# height, width, layers = frame.shape
#
# video = cv2.VideoWriter(video_name, 0, 1, (width,height))
#
# for image in images:
#     video.write(cv2.imread(os.path.join(image_folder, image)))
#
# cv2.destroyAllWindows()
# video.release()