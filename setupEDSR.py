import setup_EDSR_args
from azure.identity import InteractiveBrowserCredential, ManagedIdentityCredential
from azure.storage.blob import BlobServiceClient
from azure.storage.blob import BlobClient
import numpy
import torch
import glob
import os
from PIL import Image
from io import BytesIO
import math

def GlobalLog(msg) :
    Log(msg)    

def Log(msg) :
    print(f'{msg}')

def Abort() :
    Log(f'Aborting')
    torch.distributed.destroy_process_group()

def GetBlobService(args):
   if args.client_id != '' :
        try :
            # Acquire a credential object
            if args.client_id == '?' :               
                credential = InteractiveBrowserCredential(authority="login.microsoftonline.com", login_hint=args.login_hint)

            else :
                credential = ManagedIdentityCredential(client_id=args.client_id);

            token = credential.get_token('https://storage.azure.com/.default')

            GlobalLog(f'token:{token}')

            args.blob_service_client = BlobServiceClient(
                account_url="https://gfxmltrainingstore1.blob.core.windows.net",
                credential=credential)

            args.blob_service_client.MAX_SINGLE_PUT_SIZE = 67108864
            args.blob_service_client.MAX_BLOCK_SIZE = 4194304
            args.blob_service_client.MIN_LARGE_BLOCK_UPLOAD_THRESHOLD = args.blob_service_client.MAX_BLOCK_SIZE + 1

            GlobalLog(f'MAX_SINGLE_PUT_SIZE:{args.blob_service_client.MAX_SINGLE_PUT_SIZE:,}') 
            GlobalLog(f'MAX_BLOCK_SIZE:{args.blob_service_client.MAX_BLOCK_SIZE:,}') 
            #The minimum block size at which the memory-optimized, block upload algorithm is considered. 
            #This algorithm is only applicable to the create_blob_from_file and create_blob_from_stream methods and will prevent the full buffering of blocks. 
            #In addition to the block size, ContentMD5 validation and Encryption must be disabled as these options require the blocks to be buffered.
            GlobalLog(f'MIN_LARGE_BLOCK_UPLOAD_THRESHOLD:{args.blob_service_client.MIN_LARGE_BLOCK_UPLOAD_THRESHOLD:,}') 

        except Exception as e :
            GlobalLog(f'[Exception] {e}')
            Abort()                   

def get_filename_from_path(file_path):
    # Use the os.path.basename function to get the file name from the path
    file_name = os.path.basename(file_path)
    return file_name

def SaveOutputFile(img, args):
    if (args.client_id == '') :
        os.makedirs(os.path.dirname(args.outputFile), exist_ok=True)
        img.save(args.outputFile)
    else:
        success = False
        exception = None

        image_bytes = BytesIO()
        img.save(image_bytes, format='PNG')
        image_bytes.seek(0)
        byte_data = image_bytes.getvalue()

        try :
            blob_client = args.blob_service_client.get_blob_client(container=args.container_name_EDSR_data, blob=args.outputFile)
            blob_client.upload_blob(byte_data, overwrite=True)
            success = True
        except  Exception as e :
            success = False
            exception = e
        if not success :
            Log(f'Unable to save:{args.outputFile}|Exception:{exception}')            

def CopyLowResImages(args):
    inputFileList = glob.glob(args.inputLRPath)
    copyEvery = 1
    filesToCopy = args.files_to_copy
    if (filesToCopy != -1):
        copyEvery = math.floor(len(inputFileList) / filesToCopy)
    counter = 0
    copied = 0

    for inputFile in inputFileList:
        if copied == filesToCopy:
            break
        if counter < copyEvery - 1:
            counter += 1
        else:
            print(inputFile)
            print('\n')
            img = Image.open(inputFile)
            img = img.convert('RGB')
            outputFile = get_filename_from_path(inputFile)
            base, extension = os.path.splitext(outputFile)
            #outputFile = f"{base}x1.png"
            outputFile = f"{base}x2.png"
            args.outputFile = args.output_lr + '/' + outputFile
            SaveOutputFile(img, args)
            copied += 1
            counter = 0

def CopyHighResImages(args):
    inputFileList = glob.glob(args.inputHRPath)
    copyEvery = 1
    filesToCopy = args.files_to_copy
    if (filesToCopy != -1):
        copyEvery = math.floor(len(inputFileList) / filesToCopy)
    counter = 0
    copied = 0

    for inputFile in inputFileList:
        if copied == filesToCopy:
            break
        if counter < copyEvery - 1:
            counter += 1
        else:
            print(inputFile)
            print('\n')
            img = Image.open(inputFile)
            img = img.convert('RGB')
            imgResized = img
            #imgResized = img.resize((args.lr_width, args.lr_height), Image.LANCZOS) 
            outputFile = get_filename_from_path(inputFile)
            args.outputFile = args.output_hr + '/' + outputFile
            SaveOutputFile(imgResized, args)
            copied += 1
            counter = 0

def CopyHighResImagesTemp(args):
    inputFileList = glob.glob(args.inputHRPath)
    copyEvery = 1
    filesToCopy = args.files_to_copy
    if (filesToCopy != -1):
        copyEvery = math.floor(len(inputFileList) / filesToCopy)
    counter = 0
    copied = 0

    for inputFile in inputFileList:
        if copied == filesToCopy:
            break
        if counter < copyEvery - 1:
            counter += 1
        else:
            print(inputFile)
            print('\n')
            img = Image.open(inputFile)
            img = img.convert('RGB')
            outputFile = get_filename_from_path(inputFile)
            args.outputFile = args.output_hr + '/' + outputFile
            SaveOutputFile(img, args)
            copied += 1
            counter = 0

if __name__ == "__main__":
    args = setup_EDSR_args.parser.parse_args()
    if (args.root_folder_training_data != '') :
        args.root_folder_training_data = args.root_folder_training_data + '/'    
    args.inputLRPath = args.root_folder_training_data + args.input_lr + '/*.png'
    args.inputHRPath = args.root_folder_training_data + args.input_hr + '/*.png'
    GetBlobService(args)
    CopyLowResImages(args)
    CopyHighResImagesTemp(args)
