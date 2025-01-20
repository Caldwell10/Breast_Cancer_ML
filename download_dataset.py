import gdown
import zipfile
import os

# Google Drive file ID (dataset)
file_id ="1b1pUYa-7BTdnpUPc652qRJyvzC28jeWc"
url= f"https://drive.google.com/uc?id=1b1pUYa-7BTdnpUPc652qRJyvzC28jeWc"
output = "data/BreakHis.zip"

#Create the data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

#Download the dataset
print("Downloading dataset from google drive...")
gdown.download(url,output, quiet=False)

#Extract the dataset
print("Extracting dataset...")
with zipfile.ZipFile(output, 'r') as zip_ref:
    zip_ref.extractall("data/")

#Remove the zip file to save space
os.remove(output)
print("Dataset downloaded and extracted successfully.!")

