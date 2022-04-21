from os import listdir
from os.path import isfile, join
from PIL import Image
from PIL import ImageChops

train_path = "./data/tracking_dataset/final/tracking_train/obj_train_data/"
val_path = "./data/tracking_dataset/final/tracking_validation/obj_train_data/"
test_path = "./data/tracking_dataset/final/tracking_test/obj_train_data/"

def load_images_from_folder(folder):
    onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f))]
    onlyfiles = list(filter(lambda x: ".jpg" in x, onlyfiles))
    onlyfiles = list(map(lambda x: Image.open(join(folder,x)), onlyfiles))
    return onlyfiles

def test_similarity_two_images(image1, image2):
    diff = ImageChops.difference(image1, image2)

    if diff.getbbox():
        pass
    else:
        print("OMGF")    

def test_similarity_two_sets(set1, set2):
    print(str(len(set1) * len(set2)))
    for image1 in set1:
        for image2 in set2:
            test_similarity_two_images(image1, image2)

def test_similarity_self_set(s):
    for index in range(len(train_images)):
        set1 = [s[index]]
        set2 = s[index+1:]
        test_similarity_two_sets()

same1 = Image.open("C:/Users/rnsk/Desktop/abc.png")
same2 = Image.open("C:/Users/rnsk/Desktop/abc2.png")
different = Image.open("C:/Users/rnsk/Desktop/2019_05_13_10_48_35_A_frame230_bb0_Wheel.png")
diff_same = ImageChops.difference(same1, same2)
diff_diff = ImageChops.difference(same1, different)


print(f"similarity of the same images is: {bool(diff_same.getbbox())}")
print(f"simmilarity of different images is: {bool(diff_diff.getbbox())}")

print("a")
train_images = load_images_from_folder(train_path)
val_images = load_images_from_folder(val_path)
test_images = load_images_from_folder(test_path)

print("b")
test_similarity_two_sets(val_images, test_images)
print("c")
test_similarity_two_sets(train_images, test_images)
print("d")
test_similarity_two_sets(train_images, val_images)
print("e")

test_similarity_self_set(train_images)
print("f")
test_similarity_self_set(val_images)
print("g")
test_similarity_self_set(test_images)
print("i")

pass







