import os
import torch
from model_class import UNET3D
from google_cloud import read
from skimage.util import montage
import numpy as np
from fastapi import FastAPI
import uvicorn
from matplotlib import pyplot as plt
import time
from pymongo import MongoClient
from uuid import uuid4
import gridfs
import dns.resolver

dns.resolver.default_resolver=dns.resolver.Resolver(configure=False)
dns.resolver.default_resolver.nameservers=['8.8.8.8']

app = FastAPI()
cluster = MongoClient(
    host='mongodb+srv://clus.75dbtja.mongodb.net', 
    serverSelectionTimeoutMS=3000,
    socketTimeoutMS=3000, 
    username="ozkan",
    password=os.environ.get("MONGO_PASSWORD"),
    uuidRepresentation='standard'
)
db = cluster["gesund_ai"]
collection = db["case"]
gridfs = gridfs.GridFS(db)

model = UNET3D(in_channels=4, out_channels=64, n_classes=3).cpu()
model.load_state_dict(torch.load("model.pth", map_location='cpu'))

model.eval()

@app.get("/batch")
async def read_batch_input(names: str):
    """
    A function to read batch input data, process it, generate predictions, plot images, and return a list of plot names.
    
    Parameters:
        names (str): A comma-separated string of names representing the batch input data.
    
    Returns:
        dict: A dictionary containing a key "mesasage" with a list of plot names generated.
    """
    datas, labels = [], []
    for each in names.split(","):
        data, label = read('gesund_task_data_bucket',each.strip())
        datas.append(data)
        labels.append(label)
    print(len(datas))
    with torch.no_grad():
        results = model(torch.stack(datas))

    plot_names = []
    for i, each in enumerate(names.split(",")):
        mask_tensor = labels[i].data.squeeze()[0].squeeze().cpu().detach().numpy()
        mask = np.rot90(montage(mask_tensor))

        image_tensor = datas[i].squeeze()[1].cpu().detach().numpy()
        image = np.rot90(montage(image_tensor))

        predict_tensor = results[i].squeeze()[0].squeeze().cpu().detach().numpy()
        mask_predict = np.rot90(montage(predict_tensor))
        
        fig, (ax1 ) = plt.subplots(1, 1, figsize = (20, 20))
        ax1.imshow(image,cmap = 'gray')
        ax1.imshow(np.ma.masked_where(mask == False, mask_predict),cmap='cool', alpha=0.6)
        plt.savefig(f'{each}.png')
        plot_names.append(f'{each}.png')

    return {
        "mesasage": plot_names,
    }


async def process_single(name: str):
    
    start_time = time.time()
    try :
        data, label = read('gesund_task_data_bucket', name)

        with torch.no_grad():
            result = model(data.unsqueeze(0))

        mask_tensor = label.data.squeeze()[0].squeeze().cpu().detach().numpy()
        mask = np.rot90(montage(mask_tensor))

        image_tensor = data.squeeze()[1].cpu().detach().numpy()
        image = np.rot90(montage(image_tensor))

        predict_tensor = result.squeeze()[0].squeeze().cpu().detach().numpy()
        mask_predict = np.rot90(montage(predict_tensor))
        
        _, (ax1 ) = plt.subplots(1, 1, figsize = (20, 20))
        ax1.imshow(image,cmap = 'gray')
        ax1.imshow(np.ma.masked_where(mask == False, mask_predict),cmap='cool', alpha=0.6)
        plt.savefig(f'{name}.png')
        
        filename = f'{name}.png'
        output = os.path.join(filename)

        end_time = time.time()
        duration = end_time - start_time

        collectionid = uuid4()
        collect = {"time": duration , "name": f"{name}.png", "_id": str(collectionid)}
        collection.insert_one(collect)

        with open(output, 'rb') as f:
            contents = f.read()

        gridfs.put(contents, filename=f"{name}.png")

        return {
            "name": f"{name}.png",
            "time": duration,
            "id": str(collectionid)
        }
    
    except Exception as e:
        return {
            "error": str(e)
        }


@app.get("/single")
async def read_single_input(name: str):
    
    result = await process_single(name)
    
    return result

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8080)
