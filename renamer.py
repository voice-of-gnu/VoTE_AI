import os
import glob
import random
from tqdm import tqdm
file_path = "D:/emotion_split/train"

label_path = os.path.join(file_path, "label")
data_path = os.path.join(file_path, "data")

json_list = glob.glob(os.path.join(label_path,'*.json'))
print(len(json_list))
print("reallt??")
input()
for i in tqdm(json_list):
    name = os.path.basename(i).split(".")[0]
    nan = random.randint(0, 100000000)
    new_name = f"{nan:09}_{name}"

    old_json = i
    new_json = os.path.join(label_path, new_name + ".json")

    old_data = os.path.join(data_path, name + ".wav")
    new_data = os.path.join(data_path, new_name + ".wav")
    os.rename(old_json, new_json)
    os.rename(old_data, new_data)