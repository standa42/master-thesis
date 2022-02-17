import os
import zipfile
from pathlib import Path

from config.Config import Config
from src.helpers.helper_functions import *

# Description:
#
# Unzips raw data in the Raw folder as videos into Video folder
# Zip is expected to contain data all the data for the day 

# get raw folder
raw_folder = Config.DataPaths.ZippedRawDataFolder
video_folder = Config.DataPaths.VideoFolder

# create folders if needed
safe_mkdir(raw_folder)
safe_mkdir(video_folder)

print("This script expects zips for the individual days to be stored in format like:")
print("./data/raw/2019_05_13/2019_05_13.zip")
print("")

# iterate through all subfolders (for individual days) in Raw folder
print("Scanning Raw folder for zips:")

day_folders = os.listdir(raw_folder)
day_folders.remove("placeholder.txt") 

for day_folder in day_folders:
    zip_path = raw_folder + day_folder + '/' + day_folder + ".zip"

    print(f"Folder found: {day_folder}")
    print("Trying to unzip file: " + zip_path)
    
    # check existence of expected zip file in folder for that day
    if Path(zip_path).exists():
        # unzip it to correct location in videos
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(video_folder)
        except:
            print("There seems to be issue with file " + zip_path + ", please unzip it manualy to video folder with day name")
    else:
        print("There is no zip file named " + day_folder + ".zip" + " in folder for day: " + day_folder)

# edge case - no folders were found
if not day_folders:
    print("No subfolders with the data were found")

# script end
print("Script completed")
print("Press any key to end the program..")
input()






# TODO: old code, should be deleted
# # get raw folder
# raw_folder = Config.DataPaths.ZippedRawDataFolder
# # ensure it exists
# if not Path(raw_folder).exists():
#     print("No folder " + raw_folder + ", means there are no data.")
#     print("Please insert data as in following example: " + raw_folder + "2019_05_13/2019_05_13.zip")
# else:
#     # get all subfolders
#     day_folders = os.listdir(raw_folder)
#     day_folders.remove("placeholder.txt") # TODO make smarter
#     # for each zip file
#     for day_folder in day_folders:
#         print("Unzipping file: " + day_folder + ".zip")
#         zip_path = raw_folder + day_folder + '/' + day_folder + ".zip"
#         # does it exist?
#         if Path(zip_path).exists():
#             # if so, unzip it to correct location in videos
#             try:
#                 with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#                     videos_path = Config.DataPaths.VideoFolder
#                     extract_to_path = videos_path 
#                     safe_mkdir(videos_path)
#                     safe_mkdir(extract_to_path)
#                     zip_ref.extractall(extract_to_path)
#             except:
#                 print("There seems to be issue with file " + zip_path + ", please unzip it manualy to video folder with day name")
#         else:
#             print("There is no zip file named " + day_folder + ".zip" + " in folder for day: " + day_folder)

# print("Press any key to end the program..")
# input()