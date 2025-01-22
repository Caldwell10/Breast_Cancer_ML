from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import io
from PIL import Image
import numpy as np

def authenticate_drive():
    """Authenticate and return the Google Drive client."""
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()
    drive = GoogleDrive(gauth)
    return drive

def get_drive_files(folder_id):
    """List all files in a Google Drive folder."""
    drive = authenticate_drive()
    query = f"'{folder_id}' in parents and trashed=false"
    file_list = drive.ListFile({'q': query}).GetList()
    return file_list

def load_image_from_drive(file):
    """Download an image file from Google Drive and return it as a PIL Image."""
    content = file.GetContentFile(io.BytesIO())
    return Image.open(io.BytesIO(content))
