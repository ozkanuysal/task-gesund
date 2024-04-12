import streamlit as st
import requests
import pymongo
from pymongo import MongoClient
import gridfs
import os
import pandas as pd
from PIL import Image
import io
import dns.resolver
import streamlit as st
import requests
import pymongo
from pymongo import MongoClient
import gridfs
import os
import pandas as pd
from PIL import Image
import io
import time

dns.resolver.default_resolver=dns.resolver.Resolver(configure=False)
dns.resolver.default_resolver.nameservers=['8.8.8.8']

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
gridfs = gridfs.GridFS(db)

def page_one():
    """
    A function that handles the UI for selecting datasets and running tasks either individually or in batches.
    Makes API requests to a localhost server to retrieve and display images. Calculates and displays average time per image.
    """

    st.title('Gesund AI Task')
    options_names = st.selectbox('Select Single Dataset', ['BraTS20_Training_158', 'BraTS20_Training_234', 'BraTS20_Training_230', 'BraTS20_Training_249', 'BraTS20_Training_238'])
    batch_options = st.multiselect('Select Batch Datasets', ['BraTS20_Training_158', 'BraTS20_Training_234', 'BraTS20_Training_230', 'BraTS20_Training_249', 'BraTS20_Training_238'], max_selections=2)
    st.write('Selected Batch Datasets:', batch_options)
    st.write('Selected Dataset:', ','.join(batch_options))
    db_images = cluster["gesund_ai"]
    chunks_collection_images = db["fs.chunks"]
    files_collection_images = db["fs.files"]


    if st.button('Run Single'):
        with st.status("Processing...", expanded=False) as status :
            response = requests.get(f'http://localhost:8080/single?name={options_names}')
            if response.status_code == 200:
                count = files_collection_images.count_documents({"filename": f"{options_names}.png"})           
                data = files_collection_images.find_one({"filename": f"{options_names}.png"})
                file_id = data.get('_id')
                chunks_query = {"files_id": file_id}
                chunks = chunks_collection_images.find(chunks_query).sort("n")
                image_data = b"".join(chunk.get('data') for chunk in chunks)
                image = Image.open(io.BytesIO(image_data))
                st.image(image, caption=response.json().get('name'))
                status.update(label="Completed!", state="complete", expanded=True)
                
                avg_time = collection.find({"name": f"{options_names}.png"})
                avg_times = 0
                avg_counter = 0

                for each_avg in avg_time:
                    avg_counter += 1
                    avg_times += each_avg['time']

                # st.write('Average time : ' + str(round(avg_times / avg_counter, 3)))
                st.write('Duration : ' + str(round(response.json().get('time'),3)) \
                         + '   ---   \tAverage time : ' + str(round(avg_times / avg_counter, 3)) \
                            + '   ---   \tTotal count : ' + str(count))

            else:
                st.write(f'Error: {response.status_code}') 


    if st.button('Run Batch'):
        with st.status("Processing...", expanded=False) as status :
            response = requests.get(f'http://localhost:8080/batch?names={",".join(batch_options)}')
            if response.status_code == 200:
                image_names = response.json().get('mesasage')
                for each_image in image_names:
                    data = files_collection_images.find_one({"filename": each_image})
                    file_id = data.get('_id')
                    chunks_query = {"files_id": file_id}
                    chunks = chunks_collection_images.find(chunks_query).sort("n")
                    image_data = b"".join(chunk.get('data') for chunk in chunks)
                    image = Image.open(io.BytesIO(image_data))
                    st.image(image, caption=each_image)
                status.update(label="Completed!", state="complete", expanded=True)

            else:
                st.write(f'Error: {response.status_code}') 

        

if __name__ == '__main__':
    PAGES = {
        "Call Endpoint": page_one,    
        }
    st.sidebar.title('Navigation')
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))
    page = PAGES[selection]
    page()
