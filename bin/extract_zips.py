import os
import zipfile
from pathlib import Path

from config.Config import Config
from src.helpers.helper_functions import *

if __name__ == "__main__":
    # get raw folder
    raw_folder = Config.DataPaths.ZippedRawDataFolder
    # ensure it exists
    if not Path(raw_folder).exists():
        print("No folder " + raw_folder + ", means there are no data.")
        print("Please insert data as in following example: " + raw_folder + "2019_05_13/2019_05_13.zip")
    else:
        # get all subfolders
        day_folders = os.listdir(raw_folder)
        day_folders.remove("placeholder.txt") # TODO make smarter
        # for each zip file
        for day_folder in day_folders:
            print("Unzipping file: " + day_folder + ".zip")
            zip_path = raw_folder + day_folder + '/' + day_folder + ".zip"
            # does it exist?
            if Path(zip_path).exists():
                # if so, unzip it to correct location in videos
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        videos_path = Config.DataPaths.VideoFolder
                        extract_to_path = videos_path 
                        safe_mkdir(videos_path)
                        safe_mkdir(extract_to_path)
                        zip_ref.extractall(extract_to_path)
                except:
                    print("There seems to be issue with file " + zip_path + ", please unzip it manualy to video folder with day name")
            else:
                print("There is no zip file named " + day_folder + ".zip" + " in folder for day: " + day_folder)

    print("Press any key to end the program..")
    input()