#
#  opticalFlow.py
#
#  Created by Francesco Fiorelli on 17/12/2020.
#  Copyright © 2020 Francesco Fiorelli. All rights reserved.
#

import numpy as np
import imutils
import cv2
import sys
import os
from tqdm import tqdm

def read_video_optical(videopath):
    '''
    Parameter:
    - videopath - video path to read

    Return video, width, height, fps, num_frame
    '''

    #Opening the video and preparing the output
    video = cv2.VideoCapture(videopath)

    #Video info
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    total_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    if not video.isOpened():
        print("Cloud not open the video")
#        exit()

    return video, width, height, fps, total_frame

def save_video_optical(outpath, width, height, fps, displayHeight, live):
    '''
    Parameter:
    - outpath - Path to the folder where the files will be saved
    - width - Width of the video
    - fps - FPS of the video

    Return video_encoded
    '''

    #Output info
    outpath = os.path.join(outpath, "OpticalFlow")
    os.makedirs(outpath, exist_ok=True)
    save_path_encoded = os.path.join(outpath, "OpticalFlow.avi")
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')

    if live:
        width = int(width * (displayHeight/height))
        height = displayHeight
        save_path_encoded = os.path.join(outpath, "LiveOpticalFlow.avi")

    #Opening output stream
    video_encoded = cv2.VideoWriter(save_path_encoded, fourcc, fps, (width,height))

    return video_encoded

def liveOpticalFlow(video, blockSize, output, displayHeight, show=True, save=False, live=False, dense=False):
    """
    Parameter:
    - video - Video input
    - blockSize - Size of the blocks into which the image is divided (Default: 16)
    - output - Video output
    - displayHeight - Size of the height of the image, use to resize
    - show - If set to True, show th video during processing
    - save - If set to True, save the video
    - live - If set to True, use the camera of the computer
    - dense - Is set to True, use the dense opticalFlow

    Return frame - with opticalFlow vector draw
    """

    num_frame = 0

    #Use to show the progress bar
    print("\033[93m")
    if live:
        pbar = tqdm(total=0, desc="Live OpticalFlow")
    else:
        total_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = tqdm(total=total_frame, desc="Video OpticalFlow")

    while(video.isOpened()):
        pbar.update(1)
        ret, frame = video.read()

        if ret == False: #Breaks the loop if we have run out of frames
            break

        if live:
            frame = imutils.resize(frame, height=displayHeight)

        frame_BW = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Use Gray Color

        if num_frame == 0:
            prev_frame = frame_BW
        else:
            flow = cv2.calcOpticalFlowFarneback(prev_frame, frame_BW, None, 0.5, 3, 15, 3, 5, 1.2, 0) #	InputArray prev, InputArray next, InputOutputArray flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags
            prev_frame = frame_BW

            if dense:
                magnitude, angle = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])
                frame[:, :, 0] = angle * 180 / np.pi / 2
                frame[:, :, 1] = 255
                frame[:, :, 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
                frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR) #Use Gray Color
            else:
                frame = showFlow(frame_BW, flow, blockSize)

            if save:
                output.write(frame)
                cv2.imwrite(f"ThreeStepSearch/draw_frame_{num_frame}.png", frame)

            if show:
                cv2.imshow('Optical Flow', frame)
                key = cv2.waitKey(10)
                if key == 27:
                    break

        num_frame += 1

    pbar.close()
    print("\033[0m")
    video.release()
    output.release()
    cv2.destroyAllWindows()

def showFlow(frame, flow, blockSize=16):
    """
    Parameter:
    - frame - Frame of the video
    - flow - opticalFlow of the frame
    - blockSize - Size of the blocks into which the image is divided (Default: 16)

    Return - frame, with opticalFlow vector draw
    """

    color = (0, 255, 0)
    color2 = (255, 0, 0)
    height, width = frame.shape

    #Crea una griglia, con punti centrati in blocchi da dimensione stepSize*stepSize, e separo le coordinate y e x
    y, x = np.mgrid[blockSize/2:height:blockSize, blockSize/2:width:blockSize].reshape(2, -1).astype(int)
    flowX, flowY = flow[y, x].T

    #Ragguppo per ogni punto le sue coordinate con le coordinate del flusso, e faccio un reshape per ottenere una struttura del tipa [[x,y][x+flowX, y+flowY]]
    lines = np.vstack([x, y, x + flowX, y + flowY]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5) #Per far vedere di più il flusso di movimento

    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    #Disegno le linee
    cv2.polylines(frame, lines, 0, color)
    #Disegno i punti che rappresentano il centro dei blocchi
    for (x, y), (xFlow, yFlow) in lines:
        cv2.circle(frame, (x, y), 1, color2, -1) #Thickness = -1, use to create dot

    return frame


def main(videopath, outpath, blockSize, displayHeight=640, show=True, save=False, live=False, dense=False):
    """
    Main function

    Parameter:
    - videopath - Path of the video to be encoded
    - outpath - Path of the folder where the result is to be saved
    - blockSize - Size of the blocks into which the image is divided (Default: 16)
    - displayHeight - Size of the height of the image, use to resize only in live
    - show - If set to True, show th video during processing
    - save - If set to True, save the video
    - live - If set to True, use the camera of the computer
    - dense - Is set to True, use the dense opticalFlow
    """

    video_encoded = None

    if live:
        videopath = 0

    video, width, height, fps, total_frame = read_video_optical(videopath)
    if save:
        video_encoded = save_video_optical(outpath, width, height, fps, displayHeight, live)

    liveOpticalFlow(video, blockSize, video_encoded, displayHeight, show, save, live, dense)


if __name__ == '__main__':

    videopath = "Input/foreman_cif.mov"
    outpath = "Output"
    blockSize = 16

    main(videopath, outpath, blockSize, displayHeight=480, show=True, save=True, live=False, dense=False)
