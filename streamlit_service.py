import streamlit as st
import requests
import pymongo
from pymongo import MongoClient
import gridfs
import os
import pandas as pd
from PIL import Image
import io


st.title('Streamlit Service')
#options_names = st.selectbox('Select an option', ['BraTS20_Training_158', 'BraTS20_Training_234'])


cluster = MongoClient(
    host = 'mongodb+srv://clus.75dbtja.mongodb.net', 
    serverSelectionTimeoutMS = 3000,
    socketTimeoutMS = 3000, 
    username="ozkan",
    password=os.environ.get("MONGO_PASSWORD"),
    uuidRepresentation='standard'
)
db = cluster["gesund_ai"]
collection = db["case"]
#gridfs = gridfs.GridFS(db)

df = pd.DataFrame(list(collection.find()))
st.dataframe(df)

db_images = cluster["gesund_ai"]
chunks_collection_images = db["fs.chunks"]
files_collection_images = db["fs.files"]

for file_document in files_collection_images.find():
    file_id = file_document.get('_id')
    chunks_query = {"files_id": file_id}
    chunks = chunks_collection_images.find(chunks_query).sort("n")
    image_data = b"".join(chunk.get('data') for chunk in chunks)
    image = Image.open(io.BytesIO(image_data))
    st.image(image, caption=file_document.get('filename'))

# df = pd.DataFrame(list(files_collection.find()))
# st.dataframe(df)
