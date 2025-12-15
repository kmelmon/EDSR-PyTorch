# Copyright (c) Microsoft, 2021

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
#import torch
#from torch import nn
#import torch.optim as optim
#import torch.onnx as onnx
#import numpy as np
#import cv2
import argparse
import glob
import time
import math
import random
from PIL import Image


#----------------------------------------------------------------------------------------------------------------------------------------------------#
parser = argparse.ArgumentParser(description='Image Downscaler Tool')
parser.add_argument('--input_data_directory',       type=str,   help='directory for input data')
parser.add_argument('--output_data_directory',      type=str,   help='directory for output data')
parser.add_argument('--mv_block_size',              type=int,   default=0)
parser.add_argument('--visualize_scale',            type=float, default=0.0)
args = parser.parse_args()


#----------------------------------------------------------------------------------------------------------------------------------------------------#
def PrintProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    if iteration == total: 
        print()


#----------------------------------------------------------------------------------------------------------------------------------------------------#
def GatherInputFiles():
    inf_folder = 'Inf'

    if args.mv_block_size > 0:
        inputLowResFullFileList = glob.glob(args.input_data_directory + '\\Infiltrator_*_mv.bin')
    else:
        #inputLowResFullFileList = glob.glob(args.input_data_directory + '\\Infiltrator_*_PreTonemapHDRColor.exr')
        inputLowResFullFileList = glob.glob(args.input_data_directory + '\\Infiltrator_*.exr')

    #inputLowResFullFileList = glob.glob(args.input_data_directory + lr_folder + '\\Infiltrator_*_PostTonemapHDRColor.exr')
    #inputLowResFullFileList = glob.glob(args.input_data_directory + lr_folder + '\\Infiltrator_*.png')

    # Generate index list
    fileIndexList = []
    fileIndexTokenOffset = 3 + len(args.input_data_directory.replace('.', '_').split('_')) - 3
    for inputLowResFile in inputLowResFullFileList:
        fileIndex = int(inputLowResFile.replace('.', '_').split('_')[fileIndexTokenOffset])
        fileIndexList.append(fileIndex)
    fileIndexList.sort()

    # Generate ordered file lists
    lowResMotionVectorFileList = []
    for fileIndex in fileIndexList:
        if args.mv_block_size > 0:
            #lowResMotionVectorFileName = args.input_data_directory + '\\Infiltrator_{0}_mv.bin'.format(fileIndex)
            lowResMotionVectorFileName = args.input_data_directory + '\\Infiltrator_{0}_me200_mv.bin'.format(fileIndex)
        else:
            lowResMotionVectorFileName = args.input_data_directory + '\\Infiltrator_{0}_PreTonemapHDRColor.exr'.format(fileIndex)
            #lowResMotionVectorFileName = args.input_data_directory + '\\Infiltrator_{0:05d}_PreTonemapHDRColor.exr'.format(fileIndex)
            #lowResMotionVectorFileName = args.input_data_directory + '\\Infiltrator_{0:05d}.exr'.format(fileIndex)
        if not os.path.isfile(lowResMotionVectorFileName):
            break

        lowResMotionVectorFileList.append(lowResMotionVectorFileName)

    return lowResMotionVectorFileList, fileIndexList[0]


#----------------------------------------------------------------------------------------------------------------------------------------------------#
def GenerateMotionVectorCoordinateGrid(width, height):
    gx, gy = np.meshgrid(np.arange(width), np.arange(height))
    gx = ((gx.astype(float) + 0.5) / float(width)).reshape((height, width, 1))
    gy = ((gy.astype(float) + 0.5) / float(height)).reshape((height, width, 1))
    grid = np.concatenate((gx, gy), axis=2)
    return grid


#----------------------------------------------------------------------------------------------------------------------------------------------------#
def TransformMotionVectorsToPixelSpace(mvData, width, height):
    mvData = mvData[:, :, 1:3]
    mvData[..., [0, 1]] = mvData[..., [1, 0]]
    mvData = (mvData - 0.5) * 2.0
    mvData[..., 1] = -mvData[..., 1]
    mvData[..., 0] = mvData[..., 0] * float(width)
    mvData[..., 1] = mvData[..., 1] * float(height)
    return mvData


#----------------------------------------------------------------------------------------------------------------------------------------------------#
def TransformMotionVectorsToUVSpace(mvData, width, height):
    mvData[..., 0] = mvData[..., 0] / float(width)
    mvData[..., 1] = mvData[..., 1] / float(height)
    return mvData


#----------------------------------------------------------------------------------------------------------------------------------------------------#
def TransformMotionVectorsFromToUVSpaceToNDCCoordinateSpace(mvData, coordinateGrid):
    #mvData = -mvData + coordinateGrid
    #mvData = mvData * 2.0 - 1.0
    mvData = mvData.astype(np.float32)
    return mvData


#----------------------------------------------------------------------------------------------------------------------------------------------------#
def GenerateFiles(lowResMotionVectorFileList, baseFileIndex):
    
    inputFrameCount = len(lowResMotionVectorFileList)

    for frameIndex in range(0, inputFrameCount):
        PrintProgressBar(frameIndex + 1, inputFrameCount)

        if args.mv_block_size > 0:
            expectedMvFilesize = mvFrameWidth * mvFrameHeight * 4 * 2
            assert os.path.getsize(frameInputFiles['mv']) == expectedMvFilesize
            with open(lowResMotionVectorFileList[frameIndex], 'rb') as mvBinaryFp:
                mvLowResData = np.fromfile(mvBinaryFp, dtype=np.float32, count = mvFrameWidth * mvFrameHeight * 2).astype(np.float32).reshape(mvFrameHeight, mvFrameWidth, 2)

            lowResImageWidth = mvLowResData.shape[1]
            lowResImageHeight = mvLowResData.shape[0]

            # Transform MV data into low resolution pixel space
            mvLowResData[..., 0] = mvLowResData[..., 0] * float(lowResImageWidth)
            mvLowResData[..., 1] = mvLowResData[..., 1] * float(lowResImageHeight)

        else:
            mvLowResData = cv2.imread(lowResMotionVectorFileList[frameIndex], cv2.IMREAD_UNCHANGED).astype(np.float32)

            lowResImageWidth = mvLowResData.shape[1]
            lowResImageHeight = mvLowResData.shape[0]

            # Transform input motion vectors into low-resolution pixel space
            mvLowResData = TransformMotionVectorsToPixelSpace(mvLowResData, lowResImageWidth, lowResImageHeight)

        mvLowResDataArray = np.zeros((lowResImageHeight, lowResImageWidth, 2)).astype(np.float32)

        mvFrameWidth = 0
        mvFrameHeight = 0
        if args.mv_block_size > 0:
            mvFrameWidth = (lowResImageWidth + args.mv_block_size - 1) // args.mv_block_size
            mvFrameHeight = (lowResImageHeight + args.mv_block_size - 1) // args.mv_block_size

        # Coordinate grid for generating motion coordinates from motion vectors
        lowResMotionVectorCoordinateTileGrid = GenerateMotionVectorCoordinateGrid(lowResImageWidth, lowResImageHeight)

        assert mvLowResData.shape[0:2] == (lowResImageHeight, lowResImageWidth)

        if args.mv_block_size == 0:
            # Transform motion vectors back into UV space relative to the tile size
            mvLowResData = TransformMotionVectorsToUVSpace(mvLowResData, lowResImageWidth, lowResImageHeight);

            # Transform motion vectors into coordinates in NDC space
            mvLowResData = TransformMotionVectorsFromToUVSpaceToNDCCoordinateSpace(mvLowResData, lowResMotionVectorCoordinateTileGrid)

        #else:
            # Note: ME200 motion vectors are saved out in low resolution pixel space, not normalized UV space.  This means they should be rescaled when loaded by the network.

        assert mvLowResData.shape == (lowResImageHeight, lowResImageWidth, 2)

        if not os.path.isdir(args.output_data_directory):
            os.makedirs(args.output_data_directory)

        if args.visualize_scale == 0.0:
            outputLowResMotionVectorFile = args.output_data_directory + '\\mv{0}.bin'.format(baseFileIndex + frameIndex)

            with open(outputLowResMotionVectorFile, 'wb') as mvLowResFp:
                mvLowResFp.write(mvLowResData.copy(order='C'))

        else:
            outputLowResMotionVectorFile = args.output_data_directory + '\\mv{0}.png'.format(baseFileIndex + frameIndex)

            mvLowResData = mvLowResData.transpose(2, 0, 1)
            mvLowResData = abs(mvLowResData) * args.visualize_scale * 255.0 # * 0.2 * 255.0
            mvLowResData = np.repeat(mvLowResData, 2, axis=0)
            #mvLowResData[..., [0, 1]] = mvLowResData[..., [1, 0]]
            mvLowResData[0, ...] = mvLowResData[0, ...] * 0.0
            mvLowResData[3, ...] = mvLowResData[3, ...] * 0.0 + 1.0
            mvLowResData = mvLowResData.transpose(1, 2, 0).clip(0.0, 1.0) * 255.0
            Image.fromarray(mvLowResData.astype(np.uint8)).save(outputLowResMotionVectorFile, 'PNG')


#----------------------------------------------------------------------------------------------------------------------------------------------------#
def EntryFunc():
    lowResMotionVectorFileList, baseFileIndex = GatherInputFiles()

    inputFrameCount = len(lowResMotionVectorFileList)

    print('Found {0} input frames\n'.format(inputFrameCount))

    assert len(lowResMotionVectorFileList) > 0

    GenerateFiles(lowResMotionVectorFileList, baseFileIndex)

    print('\nComplete.')

def RenameEm():
    inputFileList = glob.glob('D:\\edsr-pytorch\\experiment\\test\\results-Demo\\*.png')
    for inputFile in inputFileList:
        print(inputFile)
        print('\n')
        base, extension = os.path.splitext(inputFile)
        parts = inputFile.split('_x1_SR.png')
        outputFile = f"{parts[0]}.png"
        os.rename(inputFile, outputFile)

def RenameEmJP():
    inputFileList = glob.glob('D:\\TestJR\\1280x800\\*.png')
    for inputFile in inputFileList:
        print(inputFile)
        print('\n')
        parts = inputFile.split('JungleRuins')
        outputFile = f"{parts[0] + 'tktkjr' + parts[1]}"
        os.rename(inputFile, outputFile)

def RenameEm2():
    inputFileList = glob.glob('D:\\TrainingDataDownscaler\\**\\1280x800\\*.png')
    for inputFile in inputFileList:
        print(inputFile)
        print('\n')
        base, extension = os.path.splitext(inputFile)
        parts = inputFile.split('x2')
        if len(parts) == 2:
            outputFile = f"{parts[0]}x2{extension}"
        else:
            outputFile = f"{base}x2{extension}"
        print("rename to: " + outputFile)
        os.rename(inputFile, outputFile)

def RenameEm3():
    directory_path = 'D:\\edsr-pytorch\\experiment\\test\\results-Demo'
    for filename in os.listdir(directory_path):
        # Split the filename into parts based on underscores
        parts = filename.split('_')

        # Check if the filename matches the expected format
        if len(parts) == 4 and parts[3] == 'SR.png':
            # Construct the new filename
            new_filename = f"{parts[0]}_{parts[1]}.png"

            # Construct the full file paths
            old_filepath = os.path.join(directory_path, filename)
            new_filepath = os.path.join(directory_path, new_filename)

            # Rename the file
            os.rename(old_filepath, new_filepath)
            print(f'Renamed: {filename} to {new_filename}')        

def RenameEm4():
    directory_path = 'D:\\doublefine\\dfp\\Unreal\\Psychonauts2\\Psychonauts2\\Saved\\Screenshots\\HELM\\V2\\2560x1600'
    for filename in os.listdir(directory_path):
        # Split the filename into parts based on underscores
        parts = filename.split('Psychonauts2')

        # Construct the new filename
        new_filename = f"HELM{parts[1]}"

        # Construct the full file paths
        old_filepath = os.path.join(directory_path, filename)
        new_filepath = os.path.join(directory_path, new_filename)

        # Rename the file
        os.rename(old_filepath, new_filepath)
        print(f'Renamed: {filename} to {new_filename}')        

def RenameEm5():
    directory_path = 'D:\\TextCaptures\\ArialBold8\\LowRes\\'
    i = 100
    while i > 0:
        inputFilename = f"Text{i}.png"
        outputFilename = f"Text{i+3}.png"

        # Construct the full file paths
        old_filepath = os.path.join(directory_path, inputFilename)
        new_filepath = os.path.join(directory_path, outputFilename)

        # Rename the file
        os.rename(old_filepath, new_filepath)
        print(f'Renamed: {inputFilename} to {outputFilename}')    
        i -= 1    

def RenameEm6():
    inputFileList = glob.glob('D:\\DIV2KTAAToUltraHighQualityTAA\\DIV2K\\DIV2K_train_LR_bicubic\\X1\\*.png')
    for inputFile in inputFileList:
        print(inputFile)
        print('\n')
        base, extension = os.path.splitext(inputFile)
        outputFile = f"{base}x1.png"
        os.rename(inputFile, outputFile)

def RenameEm6Reverse():
    inputFileList = glob.glob('D:\\DIV2KDownscaleFXAA\\DIV2K\\DIV2K_train_LR_bicubic\\X1\\*.png')
    for inputFile in inputFileList:
        print(inputFile)
        print('\n')
        parts = inputFile.split('x1.png')
        outputFile = f"{parts[0]}.png"
        os.rename(inputFile, outputFile)

def RenameEm7():
    inputFileList = glob.glob('D:\\edsr-pytorch\\experiment\\test\\results-Demo\\*.png')
    for inputFile in inputFileList:
        print(inputFile)
        print('\n')
        parts = inputFile.split('_.png')
        outputFile = f"{parts[0]}.png"
        os.rename(inputFile, outputFile)
        
def RenameEm7(inputFileList, bk, fontname):
    for inputFile in inputFileList:
        print(inputFile)
        print('\n')
        parts = inputFile.split('.png')
        outputFile = f"{parts[0]}_{bk}__{fontname}.png"
        os.rename(inputFile, outputFile)

def RenameEmRemoveWxH():
    inputFileList = glob.glob('D:\\UICaptures\\1280x800\\*.png')
    for inputFile in inputFileList:
        print(inputFile)
        print('\n')
        parts = inputFile.split('_1280x800')
        if len(parts) == 2:
            outputFile = f"{parts[0]}{parts[1]}"
            os.rename(inputFile, outputFile)

#----------------------------------------------------------------------------------------------------------------------------------------------------#
if __name__ == '__main__':

    RenameEmRemoveWxH()

def booboobooboo():
    inputFileList = glob.glob('D:\\TextCaptures_NormalEDSR\\ArialBold8\\1280x800Bilinear\\*.png')
    RenameEm7(inputFileList, 'Normal', 'ArialBold8')
    inputFileList = glob.glob('D:\\TextCaptures_NormalEDSR\\ArialBold14\\1280x800Bilinear\\*.png')
    RenameEm7(inputFileList, 'Normal', 'ArialBold14')
    inputFileList = glob.glob('D:\\TextCaptures_NormalEDSR\\ArialBold16\\1280x800Bilinear\\*.png')
    RenameEm7(inputFileList, 'Normal', 'ArialBold16')
    inputFileList = glob.glob('D:\\TextCaptures_NormalEDSR\\ArialItalic8\\1280x800Bilinear\\*.png')
    RenameEm7(inputFileList, 'Normal', 'ArialItalic8')
    inputFileList = glob.glob('D:\\TextCaptures_NormalEDSR\\ArialItalic14\\1280x800Bilinear\\*.png')
    RenameEm7(inputFileList, 'Normal', 'ArialItalic14')
    inputFileList = glob.glob('D:\\TextCaptures_NormalEDSR\\ArialItalic16\\1280x800Bilinear\\*.png')
    RenameEm7(inputFileList, 'Normal', 'ArialItalic16')
    inputFileList = glob.glob('D:\\TextCaptures_NormalEDSR\\ArialNormal8\\1280x800Bilinear\\*.png')
    RenameEm7(inputFileList, 'Normal', 'ArialNormal8')
    inputFileList = glob.glob('D:\\TextCaptures_NormalEDSR\\ArialNormal14\\1280x800Bilinear\\*.png')
    RenameEm7(inputFileList, 'Normal', 'ArialNorma14')
    inputFileList = glob.glob('D:\\TextCaptures_NormalEDSR\\ArialNormal16\\1280x800Bilinear\\*.png')
    RenameEm7(inputFileList, 'Normal', 'ArialNormal16')

    inputFileList = glob.glob('D:\\TextCaptures_PaleEDSR\\ArialBold8\\1280x800Bilinear\\*.png')
    RenameEm7(inputFileList, 'Pale', 'ArialBold8')
    inputFileList = glob.glob('D:\\TextCaptures_PaleEDSR\\ArialBold14\\1280x800Bilinear\\*.png')
    RenameEm7(inputFileList, 'Pale', 'ArialBold14')
    inputFileList = glob.glob('D:\\TextCaptures_PaleEDSR\\ArialBold16\\1280x800Bilinear\\*.png')
    RenameEm7(inputFileList, 'Pale', 'ArialBold16')
    inputFileList = glob.glob('D:\\TextCaptures_PaleEDSR\\ArialItalic8\\1280x800Bilinear\\*.png')
    RenameEm7(inputFileList, 'Pale', 'ArialItalic8')
    inputFileList = glob.glob('D:\\TextCaptures_PaleEDSR\\ArialItalic14\\1280x800Bilinear\\*.png')
    RenameEm7(inputFileList, 'Pale', 'ArialItalic14')
    inputFileList = glob.glob('D:\\TextCaptures_PaleEDSR\\ArialItalic16\\1280x800Bilinear\\*.png')
    RenameEm7(inputFileList, 'Pale', 'ArialItalic16')
    inputFileList = glob.glob('D:\\TextCaptures_PaleEDSR\\ArialNormal8\\1280x800Bilinear\\*.png')
    RenameEm7(inputFileList, 'Pale', 'ArialNormal8')
    inputFileList = glob.glob('D:\\TextCaptures_PaleEDSR\\ArialNormal14\\1280x800Bilinear\\*.png')
    RenameEm7(inputFileList, 'Pale', 'ArialNorma14')
    inputFileList = glob.glob('D:\\TextCaptures_PaleEDSR\\ArialNormal16\\1280x800Bilinear\\*.png')
    RenameEm7(inputFileList, 'Pale', 'ArialNormal16')
