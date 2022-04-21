from src.helpers.helper_functions import *

from src.data.video.video_dataset import Video_dataset

# Description
#
# Extract frames from videos in Video folder and puts them into Frames folder
# Allows user to select, how many videos will be extracted in during execution
# Provides estimate of time needed for extraction

dataset = Video_dataset()

available_days = dataset.get_available_days()
all_video_pairs = dataset.get_all_video_pairs()

# display overview of available days (of videos) and provide expectation of time it will take
counter = 0
for day in available_days:
    video_pairs_for_day = [pair for pair in all_video_pairs if pair.video_a.date == day]
    parsed_count = len(list(filter(lambda pair : pair.is_parsed(), video_pairs_for_day)))
    unparsed_count = len(video_pairs_for_day) - parsed_count
    print(f"Day {counter} which is {day}")
    print(f"contains {parsed_count} parsed pairs of videos and {unparsed_count} unparsed pairs of videos")
    print(f"with rough estimate 70 seconds per pair, it would take {'{:.2f}'.format(((parsed_count * 70.0) / 60) / 60)} hours to parse the whole day")
    counter = counter + 1

# user selects day(s)
print("")
print("Please select a day you want to parse:")
print("write either number of a day or 'all'")
selection = input()
selection = selection.strip() #trim 

print(f"Your choice was: {selection}")

chosen_pairs = []
if selection == "all":
    chosen_pairs.extend(list(filter(lambda pair: not pair.is_parsed(), all_video_pairs)))
else: 
    try:
        selection_int = int(selection)
        chosen_pairs.extend([pair for pair in all_video_pairs if pair.video_a.date == available_days[selection_int] and (not pair.is_parsed())])
    except:
        print('Your choice is either not integer, or it is larger than available days')
        print('Terminating script')
        quit()

chosen_pairs_count = len(chosen_pairs)

# inform user that he can parse only a part of the day(s) selected above
print(f"Your selection contains {chosen_pairs_count} of unparsed pairs of videos")
print(f"with rough estimate 70 seconds per pair, it would take {'{:.2f}'.format(((chosen_pairs_count * 70.0) / 60) / 60)} hours")
print(f"you can lower this number by typing number of pairs you want to parse or writing 'all'")

# user selects number of videos
selection2 = input()
selection2 = selection2.strip() 

if selection2 == "all":
    pass
else: 
    try:
        selection_int2 = int(selection2)
        chosen_pairs = chosen_pairs[:selection_int2]
    except:
        print(f"Your choice is either not integer, or it is larger than number of unparsed pairs")
        print("Terminating script")
        quit()

# again give an estimate for time it will take
print(f"with rough estimate 70 seconds per pair, it would take {'{:.2f}'.format(((len(chosen_pairs) * 70.0) / 60) / 60)} hours")

# parsing
counter2 = 0 
for pair in chosen_pairs:
    print(f"Parsing pair {counter2} {pair.video_a.datetime} from {len(chosen_pairs)} pairs")
    pair.parse()
    counter2 = counter2 + 1

# end prints
print("Parsing complete")
print("Script successfully ended")















