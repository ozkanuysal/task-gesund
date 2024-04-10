import os 
from google.cloud import storage
from refactor_code import Normalize, ConvertToMultiChannel
import torchio as tio
import nibabel as nib
import nilearn as nl
import numpy as np
import torch



def read(bucket_name, blob_name):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    data_types =  ['_flair.nii', '_t1.nii', '_t1ce.nii', '_t2.nii', '_seg.nii']
    data_img = []
    resample = tio.Resample((2,2,2))
    for data_type in data_types:
        blob = bucket.blob(blob_name + data_type)
        blob.download_to_filename(blob_name + data_type)

    for data_type in data_types[:-1]:
        img = tio.ScalarImage(blob_name + data_type) 
        img = resample(img)
        img = np.array(img)
        img = np.squeeze(img, axis = 0)
        img = Normalize(img)
        data_img.append(img)
    
    img_stack = np.stack(data_img)
    img_stack = np.moveaxis(img_stack, (0,1,2,3), (0,3,2,1))
    img_stack = torch.Tensor(img_stack)


    labels = tio.LabelMap(blob_name + data_types[-1])
    labels = resample(labels)
    labels = np.array(labels)
    labels = np.squeeze(labels, axis = 0)
    label_stack = ConvertToMultiChannel(labels)
    label_stack = torch.Tensor(label_stack)
    label = tio.LabelMap(tensor = (label_stack > 0.5))
    
    return img_stack, label