from PIL import Image, ImageDraw, ImageFont
from os import listdir
from os.path import isfile, join

from src.helpers.helper_functions import *
from config.Config import Config

import cv2
import numpy as np

image_width = 256
image_height = 256

unique_rims_path = Config.DataPaths.UniqueRimsCollage
# clipped dataset
columns = 5
rows = 7
files = ['10', '20', '30', '40', '50', '60', '70', '71', '80', '90', '91', '100', '101', '110', '120', '130', '140', '141', '150', '151', '160', '170', '171', '180', '181', '190', '200', '210', '220', '230', '240']
count = ['200','80','200','200','200','200','200','200','200','200','200','200','200','200','200','200','200','86','200','31','200','168','105','200','104','200','200','200','26','200','56']

# geometry only
# columns = 5
# rows = 5
# files = ['10', '20', '30', '40', '50', '60', '70', '80', '90', '100', '110', '120', '130', '140', '150', '160', '170', '180', '190', '200', '210', '220', '230', '240']
# count = ['200','80','200','200','200','200','200','200','200','200','200','200','200','200','200','200','200','200','200','200','200','26','200','56']

image_grid_folder = Config.DataPaths.ImageGridsGeneratedFolder
collage = Image.new('RGB', (image_width * columns, (image_height+25) * rows), color=(255,255,255))

col = -1
row = -1

for index in range(len(files)):
    col = (col + 1) % columns
    if col == 0:
        row = row + 1
    
    x_offset = col * image_width
    y_offset = row * (image_height + 25) 

    image = Image.open(join(unique_rims_path, files[index] + ".png"))

    

        # image = np.array(image)
        # image = increase_brightness(image, value=50)
        # image = Image.fromarray(image)

    collage.paste(image, (x_offset, y_offset))

    
    
    

    font = ImageFont.truetype("arial.ttf",size=22)

    x_text_offset = x_offset + 5
    y_text_offset = y_offset + 225 + 31

    collage_draw = ImageDraw.Draw(collage)
    collage_draw.text(
        (x_text_offset, y_text_offset),
        f"Label: '{files[index]}'",
        fill=(0, 0, 0),
        font=font
    )

    x_text_offset = x_offset - 15 + int(image_width/2.0)
    y_text_offset = y_offset + 225 + 31

    collage_draw.text(
        (x_text_offset, y_text_offset),
        f"Samples: {count[index]}",
        fill=(0, 0, 0),
        font=font
    )

    if int(count[index]) < 200:
        draw = ImageDraw.Draw(collage) 
        left_top = (x_offset,y_offset+image_height+0)
        left_bottom = (x_offset,y_offset+image_height+23)
        right_top = ( x_offset+image_width-2,y_offset+image_height+0)
        right_bottom = ( x_offset+image_width-2,y_offset+image_height+23)
        draw.line([left_bottom, right_bottom], fill="red",width = 3)
        draw.line([left_top,right_top], fill="red",width = 2)
        draw.line([left_top,left_bottom], fill="red",width = 2)
        draw.line([right_top,right_bottom], fill="red",width = 2)

    # original - sample count in the bottom center 
    # x_text_offset = x_offset + 98 # 185
    # y_text_offset = y_offset + 215
    # if int(count[index]) < 100:
    #     x_text_offset = x_text_offset + 10
    # font = ImageFont.truetype("arial.ttf",size=40)
    # collage_draw.text(
    #     (x_text_offset, y_text_offset),
    #     str(count[index]),
    #     fill=(255, 0, 0),
    #     font=font,
    #     stroke_width=1
    # )

safe_mkdir(image_grid_folder)
collage.save(join(image_grid_folder, f"collage_{columns}x{rows}.png"))

