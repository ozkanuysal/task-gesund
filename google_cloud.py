import os 
from google.cloud import storage
from refactor_code import Normalize, ConvertToMultiChannel
import torchio as tio
import nibabel as nib
import nilearn as nl
import numpy as np
import torch

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gen-lang-client-0452436008-8e5d369ba13c.json"
storage_client = storage.Client()

def create_bucket():
    """
    Creates a Google Cloud Storage bucket with the name 'gesund_task_data_bucket' in the US region.
    
    Returns:
        The created bucket object.
    """
    storage_client = storage.Client()
    bucket_name = "gesund_task_data_bucket"
    bucket = storage_client.bucket(bucket_name)
    bucket.location = "US"
    bucket = storage_client.create_bucket(bucket)
    my_bucket = storage_client.get_bucket("gesund_task_data_bucket")
    return my_bucket
def upload_to_bucket(blob_name, file_path, bucket_name):
    """
    Uploads a file to a specific bucket in Google Cloud Storage.

    Parameters:
    - blob_name (str): The name of the blob to be created in the bucket.
    - file_path (str): The local file path of the file to be uploaded.
    - bucket_name (str): The name of the bucket in Google Cloud Storage.

    Returns:
    - bool: True if the file was successfully uploaded, False otherwise.
    """
    try:
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(file_path)
        return True
    except Exception as e:
        print(e)
        return False


""" download files """

def download_file_from_bucket(blob_name, file_path, bucket_name):
    """
    Downloads a file from a specific bucket in Google Cloud Storage.

    Parameters:
    - blob_name (str): The name of the blob to be downloaded from the bucket.
    - file_path (str): The local file path where the downloaded file will be saved.
    - bucket_name (str): The name of the bucket in Google Cloud Storage.

    Returns:
    - bool: True if the file was successfully downloaded, False otherwise.
    """
    try:
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(blob_name)
        with open(file_path, 'wb') as f:
            storage_client.download_blob_to_file(blob, f)
        return True
    except Exception as e:
        print(e)
        return False
    

""" read file """
def read(bucket_name, blob_name):
    """
    A function that reads medical image data from a specified Google Cloud Storage bucket.
    
    Args:
        bucket_name (str): The name of the Google Cloud Storage bucket.
        blob_name (str): The name of the blob containing the medical image data.
    
    Returns:
        tuple: A tuple containing the processed medical image stack and corresponding labels.
    """
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