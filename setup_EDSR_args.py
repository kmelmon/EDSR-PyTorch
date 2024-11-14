import argparse

parser = argparse.ArgumentParser(description='Setter-upper for EDSR experiments')
parser.add_argument('--root_folder_training_data', type=str,  help='path to root folder for input training data, only useful for Azure jobs', default='')
parser.add_argument('--input_lr', type=str,  help='path to low-res input folder', default='Data/PC_Captures/cadmusMultiAA/TrainingDataCadmusMultiAA/AbandonedRestaurant/1280x800FXAA')
parser.add_argument('--input_hr', type=str,  help='path to high-res input folder', default='Data/PC_Captures/cadmusMultiAA/TrainingDataCadmusMultiAA/AbandonedRestaurant/2560x1600FXAA')
parser.add_argument('--output_lr', type=str,   help='path to low-res output folder', default='DIV2KAutomationTest/DIV2K/DIV2K_train_LR_bicubic/X1')
parser.add_argument('--output_hr', type=str,   help='path to high-res output folder', default='DIV2KAutomationTest/DIV2K/DIV2K_train_HR')
parser.add_argument('--lr_width', type=int,   help='width of low-res image', default=1280)
parser.add_argument('--lr_height', type=int,   help='height of low-res image', default=800)
parser.add_argument('--client_id', type=str, help = 'The client ID to authenticate for a user-assigned managed identity. If empty string, saves files locally. Otherwise saves into azure storage. Use ? for interactive credentials with --login_hint arg', default='7ea1e259-3558-425a-a8b9-106b5b76ac20')
parser.add_argument('--login_hint', type=str, help = 'login hint', default='')
parser.add_argument('--container_name_training_data', type=str, help='Container name', default='trainingblob')
parser.add_argument('--container_name_EDSR_data', type=str, help='Container name', default='cadmus')
