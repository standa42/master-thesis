from src.framesloading.data_loader import day_data
from src.videoparsing.video_dataset import video_dataset

from distutils.sysconfig import get_python_lib
print(get_python_lib())

dataset = video_dataset()
dataset.standalone_cmd_interface()