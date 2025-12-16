# Notes on using EDSR downscaler
EDSR model was changed to downscale instead of upscale.  The change is extremely simple (see src/model/edsr.py):  
The first operation is to perform a strided convolution.  This skips every other pixel and produces a learned downscaled image.  
The last operation that previously upscaled was removed.  
This change is hard-coded into the model.  If you want original behavior, switch to main branch.  
## Training
Below are how to's on training EDSR
### Training Data
The training code expects lr and hr images as loose image files, in a specific directory hierarchy.  The scale factor is baked into this.  For example, if scale factor is 2X (either upscale or downscale):  
root/DIV2K/DIV2K_train_LR_bicubic/X2  => lr images  
root/DIV2K/DIV2K_train_HR => hr images  
<br>
There is also a requirement that the files have a naming convention like this:  
file_1.png/file_1x2.png  
file_2.png/file_2x2.png  
etc
<br>
There is a helper script to copy images into the expected training directories:  
setupEDSR.py  
setup_EDSR_args.py  
The script works both on local files as well as Azure storage using Sigma subscription.  
For an example job that uses Azure storage, see https://ml.azure.com/experiments/id/80821360-5968-4231-9b34-7749fb300955/runs/setupEDSR_1745618828_7675e041?wsid=/subscriptions/68d80131-d556-4763-8084-2a66f90a8efd/resourceGroups/gfxmltraining/providers/Microsoft.MachineLearningServices/workspaces/GfxMLTrainingGPUWorkspace1&tid=72f988bf-86f1-41af-91ab-2d7cd011db47
<br>
There are also these 2 scripts that help copy and rename files:  
CopyEm.py  
RenameEm.py
<br>
### Doing training runs
To launch training:  python main.py  
There are many parameters to control training.  Here are the ones I typically use and the typical value:
- --model EDSR
- --downscale
- --scale 2
- --loss 1*L1+2*VGG
- --save
- --patch_size 96
- --n_resblocks 32
- --n_feats 128
- --res_scale 0.1
- --reset
- --ext img
- --save_models
- --azureml (only specify if running in Azure ML, remove for local training)

For an example training job, see https://ml.azure.com/experiments/id/3497afb9-1838-4f58-8220-e7437ed155ea/runs/EDSR_1L1_2VGG_1738451251_cbb76799?wsid=/subscriptions/68d80131-d556-4763-8084-2a66f90a8efd/resourceGroups/gfxmltraining/providers/Microsoft.MachineLearningServices/workspaces/GfxMLTrainingGPUWorkspace1&tid=72f988bf-86f1-41af-91ab-2d7cd011db47
<br>
You can find model outputs from training in:
- root/experiments/test/model (if training locally)
- outputs/model (if training in Azure ML)




