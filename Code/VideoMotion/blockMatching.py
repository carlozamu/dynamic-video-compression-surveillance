#
#  blockMatching.py
#
#  Created by Francesco Fiorelli on 17/12/2020.
#  Copyright © 2020 Francesco Fiorelli. All rights reserved.
#

import numpy as np
import cv2
import sys
import os
from scipy.fftpack import dctn, idctn
from tqdm import tqdm
import matplotlib.pyplot as plt

"""
Type of search implemented:

1 - Exhaustive Search

The simplest approach to BBME is the exhaustive search (EBBME):

    1.  The anchor frame is subdivided into equally sized blocks (e.g., 16x16)
    2.  For each block, a search window is set in the target frame, centered at the position of the anchor block
    3.  The search window is raster scanned, and at each position the DFD between anchor block and displaced target block is calculated
    4.  The position corresponding to the lower DFD is selected

2 - Three Step Search

The method relies on the locality principle, the number of steps (and DFD computations) is fixed a-priori

    1. Search a subset of equally-spaced solutions, using a step 4 in every direction (9 positions including the center)
    2. Use the best match as the new starting point, halving the step size (+ 8 positions)
    3. Use again the best match as the starting point for a local search, looking at the 8-connected (+ 8 positions)

3 - 2D Log Search

Similar to 3-step, but number of inspected positions is not fixed

    1. Initialize the initial step-size to S
    2. Inspect the current position (center of search window) and the positions a distance S in the 4 directions N, E, S, W
    3. If the best match is the center, then half the step-size, otherwise move the center in the best match keeping the same step size
    4. If S > 2, then goto 2, otherwise select the best among the current center and its 8-connected

"""

def read_video(videopath):
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
        #exit()

    return video, width, height, fps, total_frame

def save_video(outpath, width, height, fps, searchType, errore_enable, pixel_Accuracy):
    '''
    Parameter:
    - outpath - Path to the folder where the files will be saved
    - width - Width of the video
    - height - Height of the video
    - fps - FPS of the video
    - searchType - Type of research to be applied - "ThreeStepSearch", "ExhaustiveSearch", "2DLogSearch"
    - errore_enable - If set to True show the error in the encoded video, and save the error separately
    - pixel_Accuracy - Select the pixel accuracy - "Normal", "Half", "Quarter"

    Return video_encoded, video_error
    '''

    #Output info
    os.makedirs(outpath, exist_ok=True)
    save_path_encoded = os.path.join(outpath, searchType + f"_{pixel_Accuracy}" + "_encoded.avi")

    if errore_enable:
        save_path_encoded = os.path.join(outpath, searchType + f"_{pixel_Accuracy}" + "_encoded_with_error.avi")
        save_path_error = os.path.join(outpath, searchType + f"_{pixel_Accuracy}" + "_error.avi")

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')

    #Opening output stream
    video_encoded = cv2.VideoWriter(save_path_encoded, fourcc, fps, (width,height))
    video_error = None
    if errore_enable:
        video_error = cv2.VideoWriter(save_path_error, fourcc, fps, (width,height), 0)

    return video_encoded, video_error

def dct2(block):
    """
    Parameter:
    - block - block of the frame

    return - DTC block
    """
    return dctn(block, norm='ortho')

def idct2(block):
    """
    Parameter:
    - block - block of the frame

    return - IDTC block
    """
    return idctn(block, norm='ortho')

def quantization(frame):
    """
    Parameter:
    - frame - Frame on the videopath

    return  - Frame after applying the DCT - Quantization - IDCT
    """

    quantization_matrix=[[8,16,19,22,26,27,29,34],
        [16,16,22,24,27,29,34,37],
        [19,22,26,27,29,34,34,38],
        [22,22,26,27,29,34,37,40],
        [22,26,27,29,32,35,40,48],
        [26,27,29,32,35,40,48,58],
        [26,27,29,34,38,46,56,69],
        [27,29,35,38,46,56,69,83]]

    size = frame.shape
    frame_dct = np.zeros(size)

    #Apply DCT and Quantization_Matrix to 8x8 blocks
    for i in range(0, size[0], 8):
        for j in range(0, size[1], 8):
            block = frame[i:(i+8),j:(j+8)]
            frame_dct[i:(i+8),j:(j+8)] = np.around(np.around(dct2(block))/quantization_matrix)

    #Apply IDCT and Quantization_Matrix to 8x8 blocks
    for i in np.r_[:size[0]:8]:
        for j in np.r_[:size[1]:8]:
            block = frame_dct[i:(i+8),j:(j+8)]
            frame_dct[i:(i+8),j:(j+8)] = np.around(idct2(block*quantization_matrix))

    return frame_dct


def calculationMAD(targetBlock, anchorBlock):
    """
    Calculate the MAD, subtract pixel-by-pixel the anchorBlock to targetBlock, make the value absolute, some ALL of them
    and divide for the dimension of the targetBlock (Default dimension: 16x16)

    Parameter:
    - targetBlock - block of the target frame
    - anchorBlock - block of the anchor frame

    Return MAD (Mean Absolute Difference) between targetBlock and anchorBlock
    """
    return np.sum(np.abs(np.subtract(targetBlock, anchorBlock)))/(targetBlock.shape[0]*targetBlock.shape[1])

def plotMotionVector(height, width, flow, blockSize, precision, outpath, num_frame):
    """
    Parameter:
    - height - Height of the video
    - width - Width of the video
    - flow - Motion vector matrix
    - blockSize - Size of the blocks into which the image is divided (Default: 16)
    - precision - Select the precision of the motion vector Plot  - "Normal", "Double", "Quadruple"
    - outpath - Path of the folder where the result is to be saved
    - num_frame - Number of the current frame
    """

    if precision == "Normal":
        step = blockSize
    elif precision == "Double":
        step = int(blockSize/2)
    elif precision == "Quadruple":
        step = int(blockSize/4)

    y, x = np.mgrid[0:height:step, 0:width:step].reshape(2, -1).astype(int)
    flowX = flow[:,:,0]
    flowY = flow[:,:,1]
    flowX = flowX[y, x]
    flowY = flowY[y, x]

    scale = (height/blockSize-1) * (width/blockSize-1)
    fig, ax = plt.subplots(figsize=(10,int(height*10/width)))
    q = ax.quiver(x,y,flowX,flowY, scale=scale, width=0.002)
    plt.gca().invert_yaxis()
    # plt.show()

    os.makedirs(outpath, exist_ok=True)
    plt.savefig(f"{outpath}/motionVector_{num_frame}.eps")
    plt.close()


def exhaustiveSearch(anchorFrame, targetFrame, anchorFrame_Cr, anchorFrame_Cb, blockSize=16, searchArea=7):
    """
    Exhaustive Search implementation

    Parameter:
    - anchorFrame - anchorFrame that we use for reference
    - targetFrame - targetFrame, for which we want to calculate the Motion Field
    - anchorFrame_Cr - Cr-channel of the anchroFrame
    - anchorFrame_Cb - Cb-channel of the anchroFrame
    - blockSize - Size of the blocks into which the image is divided (Default: 16)
    - searchArea - Size of the search area in pixels (Default: 7)

    Return - best matchBlock
    """

    boundY, boundX = anchorFrame.shape
    predictedYCrCb = np.zeros((boundY, boundX, 3))
    flow = np.zeros((boundY, boundX, 2))

    for y in range(0, boundY-blockSize+1, blockSize):
        for x in range(0, boundX-blockSize+1, blockSize):

            temp_y = y
            temp_x = x
            minMAD = float("inf")
            minPoint = None

            for m in range(-searchArea, searchArea+1):
                for n in range(-searchArea, searchArea+1):

                    target_y = temp_y + m
                    target_x = temp_x + n

                    if target_y >= 0 and target_y+blockSize <= boundY and target_x >= 0 and target_x+blockSize <= boundX:

                        targetBlock = targetFrame[target_y:target_y+blockSize, target_x:target_x+blockSize]
                        anchorBlock = anchorFrame[y:y+blockSize, x:x+blockSize]
                        MAD = calculationMAD(targetBlock, anchorBlock)

                        if MAD < minMAD:
                            minMAD = MAD
                            best_x, best_y = target_x, target_y

            predictedYCrCb[y:y+blockSize, x:x+blockSize, 0] = targetFrame[best_y:best_y+blockSize, best_x:best_x+blockSize]
            predictedYCrCb[y:y+blockSize, x:x+blockSize, 1] = anchorFrame_Cr[best_y:best_y+blockSize, best_x:best_x+blockSize]
            predictedYCrCb[y:y+blockSize, x:x+blockSize, 2] = anchorFrame_Cb[best_y:best_y+blockSize, best_x:best_x+blockSize]
            flow[y][x][0] = int(x - best_x)
            flow[y][x][1] = int(y - best_y)

    return predictedYCrCb, flow

def threeStepSearch(anchorFrame, targetFrame, anchorFrame_Cr, anchorFrame_Cb, blockSize=16, searchArea=7, pixel_Accuracy="Normal"):
    """
    Three-Step Search implementation

    Parameter:
    - anchorFrame - anchorFrame that we use for reference
    - targetFrame - targetFrame, for which we want to calculate the Motion Field
    - anchorFrame_Cr - Cr-channel of the anchroFrame
    - anchorFrame_Cb - Cb-channel of the anchroFrame
    - blockSize - Size of the blocks into which the image is divided (Default: 16)
    - searchArea - Size of the search area in pixels (Default: 7)

    Return - predictedY, predictedCr, predictedCb, after applying motion compensation
    """

    maxStep = int(2**(np.floor(np.log10(searchArea+1)/np.log10(2))-1))
    boundY, boundX = anchorFrame.shape
    predictedYCrCb = np.zeros((boundY, boundX, 3))
    flow = np.zeros((boundY, boundX, 2))

    # Python range go from the first value, to the second (upper limit), with step of third value
    for y in range(0, boundY-blockSize+1, blockSize):
        for x in range(0, boundX-blockSize+1, blockSize):

            temp_y = y
            temp_x = x
            stepSize = maxStep
            minMAD = float("inf")
            minPoint = None

            while stepSize >= 1:

                for m in range(-stepSize, stepSize+1, stepSize):
                    for n in range(-stepSize, stepSize+1, stepSize):

                        target_y = temp_y + m
                        target_x = temp_x + n

                        if target_y >= 0 and target_y+blockSize <= boundY and target_x >= 0 and target_x+blockSize <= boundX:

                            targetBlock = targetFrame[target_y:target_y+blockSize, target_x:target_x+blockSize]
                            anchorBlock = anchorFrame[y:y+blockSize, x:x+blockSize]
                            MAD = calculationMAD(targetBlock, anchorBlock)

                            if MAD < minMAD:
                                minMAD = MAD
                                minPoint = (target_x, target_y) #X, Y

                #Add the end of the first while cycle, check the MAD, and change the center to the point with minimum MAD
                if minMAD != float("inf"):
                    temp_x = minPoint[0]
                    temp_y = minPoint[1]
                else:
                    print(f"Error - no MAD under infinity found")
                    exit()

                stepSize = int(stepSize/2)

            predictedYCrCb[y:y+blockSize, x:x+blockSize, 0] = targetFrame[temp_y:temp_y+blockSize, temp_x:temp_x+blockSize]
            predictedYCrCb[y:y+blockSize, x:x+blockSize, 1] = anchorFrame_Cr[temp_y:temp_y+blockSize, temp_x:temp_x+blockSize]
            predictedYCrCb[y:y+blockSize, x:x+blockSize, 2] = anchorFrame_Cb[temp_y:temp_y+blockSize, temp_x:temp_x+blockSize]
            flow[y][x][0] = int(x - temp_x)
            flow[y][x][1] = int(y - temp_y)

    return predictedYCrCb, flow

def getPointList(x, y, stepSize, point_number):
    """
    Parameter:
    - x - Top Left coordinate of the block
    - y - Top Left coordinate of the block
    - stepSize - Size of the the step
    - point_number - number of point to be return, 5 for 4-connected, 9 for 8-connected

    Return - pointList, list of the point order C, N, E, S, W, NE, SE, SW, NW
    """

    point1 = (x, y) #Center
    point2 = (x, y-stepSize) #Top, Center
    point3 = (x+stepSize, y-stepSize) #Top Right
    point4 = (x+stepSize, y) #Center Right
    point5 = (x+stepSize, y+stepSize) #Bottom Right
    point6 = (x, y+stepSize) #Bottom Center
    point7 = (x-stepSize, y+stepSize) #Bottom Left
    point8 = (x-stepSize, y) #Center Left
    point9 = (x-stepSize, y-stepSize) #Top Left

    #Center block, 4 blocks for 4-connected, the 4 blocks for 8-connected
    if point_number == 5:
        return [point1, point2, point4, point6, point8]
    else:
        return [point1, point2, point4, point6, point8, point3, point5, point7, point9]

def logSearch2D(anchorFrame, targetFrame, anchorFrame_Cr, anchorFrame_Cb, blockSize=16, searchArea=7):
    """
    2D Log-Search implementation

    Parameter:
    - anchorFrame - anchorFrame that we use for reference
    - targetFrame - targetFrame, for which we want to calculate the Motion Field
    - anchorFrame_Cr - Cr-channel of the anchroFrame
    - anchorFrame_Cb - Cb-channel of the anchroFrame
    - blockSize - Size of the blocks into which the image is divided (Default: 16)
    - searchArea - Size of the search area in pixels (Default: 7)

    return - best matchBlock
    """

    # maxStep = int(2**(np.floor(np.log10(searchArea+1)/np.log10(2))-1))
    maxStep = 8
    boundY, boundX = anchorFrame.shape
    predictedYCrCb = np.zeros((boundY, boundX, 3))
    flow = np.zeros((boundY, boundX, 2))

    # Python range go from the first value, to the second (upper limit), with step of third value
    for y in range(0, boundY-blockSize+1, blockSize):
        for x in range(0, boundX-blockSize+1, blockSize):

            stepSize = maxStep
            minMAD = float("inf")
            minPoint = None
            center = False #Use to indicate if the best point is the center

            pointList = getPointList(x, y, stepSize, 5)

            while stepSize >= 2:

                for i in range(len(pointList)):
                    target_x = pointList[i][0]
                    target_y = pointList[i][1]

                    if target_y >= 0 and target_y+blockSize <= boundY and target_x >= 0 and target_x+blockSize <= boundX:

                        targetBlock = targetFrame[target_y:target_y+blockSize, target_x:target_x+blockSize]
                        anchorBlock = anchorFrame[y:y+blockSize, x:x+blockSize]
                        MAD = calculationMAD(targetBlock, anchorBlock)

                        if MAD < minMAD:
                            minMAD = MAD
                            minPoint = (target_x, target_y) #X, Y
                            if i == 0:
                                center = True

                if minMAD != float("inf"):
                    temp_x = minPoint[0]
                    temp_y = minPoint[1]
                else:
                    print(f"Error - no MAD under infinity found")
                    exit()

                if center:#If the best point is the center, we have to half the stepSize
                    stepSize = int(stepSize/2)

                if stepSize > 2:
                    pointList = getPointList(temp_x, temp_y, stepSize, 5) #4-connected
                else:
                    pointList = getPointList(temp_x, temp_y, stepSize, 9) #8-connected

            predictedYCrCb[y:y+blockSize, x:x+blockSize, 0] = targetFrame[temp_y:temp_y+blockSize, temp_x:temp_x+blockSize]
            predictedYCrCb[y:y+blockSize, x:x+blockSize, 1] = anchorFrame_Cr[temp_y:temp_y+blockSize, temp_x:temp_x+blockSize]
            predictedYCrCb[y:y+blockSize, x:x+blockSize, 2] = anchorFrame_Cb[temp_y:temp_y+blockSize, temp_x:temp_x+blockSize]
            flow[y][x][0] = int(x - temp_x)
            flow[y][x][1] = int(y - temp_y)

    return predictedYCrCb, flow

def preprocessing(frame, pixel_Accuracy):
    """
    Frame preparation for processing. Frame conversion and decomposition in YCrCb

    Parameter:
    - frame - Frame to be processed
    - pixel_Accuracy - Select the pixel accuracy - "Normal", "Half", "Quarter"

    Return - frame_YCrCb, processed_Y (Channel Y, processed with DCT, IDCT and Quantization Matrix)
    """

    frame_YCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    processed_Y = quantization(frame_YCrCb[:,:,0])

    if pixel_Accuracy != "Normal":
        if pixel_Accuracy == "Half":
            fx = fy = 2
        elif pixel_Accuracy == "Quarter":
            fx = fy = 4

        frame_YCrCb = cv2.resize(frame_YCrCb, dsize=(0,0), fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
        processed_Y = cv2.resize(processed_Y, dsize=(0,0), fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)


    return frame_YCrCb, processed_Y

def postprocessing(predictedFrame, frame_YCrCb, errore_enable, pixel_Accuracy):
    """
    Parameter:

    - predictedFrame - Prediceted Frame in YCrCb
    - frame_YCrCb - frame converted in YCrCb
    - errore_enable - If set to True show the error in the encoded video, and save the error separately
    - pixel_Accuracy - Select the pixel accuracy - "Normal", "Half", "Quarter"

    Return predictedFrame (with/without error), image_subtract, codec_Memory
    """

    if pixel_Accuracy != "Normal":
        if pixel_Accuracy == "Half":
            fx = fy = 0.5
        elif pixel_Accuracy == "Quarter":
            fx = fy = 0.25

        predictedFrame = cv2.resize(predictedFrame, dsize=(0,0), fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
        frame_YCrCb = cv2.resize(frame_YCrCb, dsize=(0,0), fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)

    #Calculate and show the error
    if errore_enable:
        image_subtract = np.subtract(np.clip(predictedFrame[:,:,0], 0, 255), frame_YCrCb[:,:,0])
        image_subtract = np.clip(image_subtract, 0, 255)

        error = quantization(image_subtract)
        predictedFrame[:,:,0] = np.add(np.clip(error, 0, 255), np.clip(predictedFrame[:,:,0], 0, 255))
        predictedFrame[:,:,0] = np.clip(predictedFrame[:,:,0], 0, 255)
        codec_Memory = predictedFrame[:,:,0]
    else:
        image_subtract = None
        predictedFrame[:,:,0] = np.clip(predictedFrame[:,:,0], 0, 255)
        codec_Memory = predictedFrame[:,:,0]

    predictedFrame = cv2.cvtColor(np.uint8(predictedFrame), cv2.COLOR_YCrCb2BGR)

    if pixel_Accuracy != "Normal":
        if pixel_Accuracy == "Half":
            fx = fy = 2
        elif pixel_Accuracy == "Quarter":
            fx = fy = 4

        codec_Memory = cv2.resize(codec_Memory, dsize=(0,0), fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)

    return predictedFrame, image_subtract, codec_Memory

def main(videopath, outpath, blockSize=16, searchArea=7, searchType="ThreeStepSearch", errore_enable=True, pixel_Accuracy="Normal", plot_Vector=False):
    """
    Main function

    Parameter:
    - videopath - Path of the video to be encoded
    - outpath - Path of the folder where the result is to be saved
    - blockSize - Size of the blocks into which the image is divided (Default: 16)
    - searchArea - Size of the search area in pixels (Default: 7)
    - searchType - Type of research to be applied - "ThreeStepSearch", "ExhaustiveSearch", "2DLogSearch"
    - errore_enable - If set to True show the error in the encoded video, and save the error separately (Default: True)
    - pixel_Accuracy - Select the pixel accuracy - "Normal", "Half", "Quarter" (Default: "Normal")
    - plot_Vector - If set to True, plot and save the motionVector
    """

    #Opening the video and preparing the output
    #Video info
    video, width, height, fps, total_frame = read_video(videopath)

    #Output info
    outpath = os.path.join(outpath, searchType)
    video_encoded, video_error = save_video(outpath, width, height, fps, searchType, errore_enable, pixel_Accuracy)

    #Use to show the progress bar
    print("\033[93m")
    pbar = tqdm(total=total_frame, desc=searchType + " with " + pixel_Accuracy + " accuracy")

    while(video.isOpened()):
        num_frame = int(video.get(cv2.CAP_PROP_POS_FRAMES))
        pbar.update(1) #Use to update the progress bar in terminal

        ret, frame = video.read()

        if ret == False: #Breaks the loop if we have run out of frames
            break

        #Y,Cr,Cb - channel level, Y - reconstructed
        frame_YCrCb, processed_Y = preprocessing(frame, pixel_Accuracy)

        #First frame is the Intra Frame, then every 16 frame we have a new Intra Frame to recover the error
        #codec_Memory contains the frame of the previous iteration
        if num_frame == 0 or num_frame%15 == 0:
            codec_Memory = processed_Y
        else:
            if searchType == "ThreeStepSearch":
                predictedFrame, flow = threeStepSearch(processed_Y, codec_Memory, frame_YCrCb[:,:,1], frame_YCrCb[:,:,2], blockSize, searchArea)
            elif searchType == "ExhaustiveSearch":
                predictedFrame, flow = exhaustiveSearch(processed_Y, codec_Memory, frame_YCrCb[:,:,1], frame_YCrCb[:,:,2], blockSize, searchArea)
            elif searchType == "2DLogSearch":
                predictedFrame, flow = logSearch2D(processed_Y, codec_Memory, frame_YCrCb[:,:,1], frame_YCrCb[:,:,2], blockSize, searchArea)

            frame, image_subtract, codec_Memory  = postprocessing(predictedFrame, frame_YCrCb, errore_enable, pixel_Accuracy)

            if errore_enable:
                video_error.write(np.uint8(image_subtract))

            if plot_Vector:
                outPlot = os.path.join(outpath, "Plot")
                plotMotionVector(height, width, flow, blockSize, "Double", outPlot, num_frame)

        video_encoded.write(frame)

    pbar.close()
    print("\033[0m")
    video.release()
    video_encoded.release()
    if errore_enable:
        video_error.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    videopath = "Input/foreman_cif.mov"
    outpath = "Output"
    blockSize = 16
    searchArea = 7

    if len(sys.argv)==2 and sys.argv[1]=='--help':
        import blockMatching
        help(blockMatching)
        exit()

    main(videopath, outpath, blockSize, searchArea, searchType="ThreeStepSearch", errore_enable=True, pixel_Accuracy="Normal", plot_Vector=False)
    main(videopath, outpath, blockSize, searchArea, searchType="ExhaustiveSearch", errore_enable=True, pixel_Accuracy="Normal", plot_Vector=False)
    main(videopath, outpath, blockSize, searchArea, searchType="2DLogSearch", errore_enable=True, pixel_Accuracy="Normal", plot_Vector=False)











































# End of file
