import torch
import nibabel as nib
import nilearn as nl
import numpy as np
import torchio as tio
from skimage.util import montage
from matplotlib import pyplot as plt
from torchvision import transforms as T
from model_class import UNET3D

def Normalize(image : np.ndarray):
    return (image - np.min(image))/(np.max(image) - np.min(image))

def ConvertToMultiChannel(labels):
   label_TC = labels.copy()
   label_TC[label_TC == 1] = 1
   label_TC[label_TC == 2] = 0
   label_TC[label_TC == 4] = 1
   
   
   label_WT = labels.copy()
   label_WT[label_WT == 1] = 1
   label_WT[label_WT == 2] = 1
   label_WT[label_WT == 4] = 1
   
   label_ET = labels.copy()
   label_ET[label_ET == 1] = 0
   label_ET[label_ET == 2] = 0
   label_ET[label_ET == 4] = 1
   
   label_stack = np.stack([label_WT, label_TC, label_ET])
   label_stack = np.moveaxis(label_stack, (0,1,2,3), (0,3,2,1))
   return label_stack

if __name__ == '__main__':
        
    # load model
    model = UNET3D(in_channels=4, out_channels=64, n_classes=3)
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    # file and functions init
    example_image_file = '/home/ozkan/Desktop/gesund_task/BraTS20_Training_158/BraTS20_Training_158_flair.nii'
    example_mask = '/home/ozkan/Desktop/gesund_task/BraTS20_Training_158/BraTS20_Training_158_seg.nii'


    data_types =  ['_flair.nii', '_t1.nii', '_t1ce.nii', '_t2.nii']
    data_img = []
    resample = tio.Resample((2,2,2))

    # image settings
    with torch.no_grad():
        for data_type in data_types:
            img = tio.ScalarImage(example_image_file)
            img = resample(img)
            img = np.array(img)
            img = np.squeeze(img, axis = 0)
            img = Normalize(img)
            data_img.append(img)

        img_stack = np.stack(data_img)
        img_stack = np.moveaxis(img_stack, (0,1,2,3), (0,3,2,1))
        img_stack = torch.Tensor(img_stack)


        subjects = tio.Subject(image = tio.ScalarImage(tensor = img_stack), id = 25)
        result = model(img_stack.unsqueeze(0))

    # mask settings
    labels = tio.LabelMap(example_mask)
    labels = resample(labels)
    labels = np.array(labels)
    labels = np.squeeze(labels, axis = 0)
    label_stack = ConvertToMultiChannel(labels)
    label_stack = torch.Tensor(label_stack)
    label = tio.LabelMap(tensor = (label_stack > 0.5))


    mask_tensor = label.data.squeeze()[0].squeeze().cpu().detach().numpy()
    image_tensor = subjects.image.data.squeeze()[1].cpu().detach().numpy()
    mask = np.rot90(montage(mask_tensor))
    image = np.rot90(montage(image_tensor))

    fig, (ax1 ) = plt.subplots(1, 1, figsize = (20, 20))
    ax1.imshow(image,cmap = 'gray')
    ax1.imshow(np.ma.masked_where(mask == False, mask),cmap='cool', alpha=0.6)

    plt.savefig('output.png')