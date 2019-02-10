import os
import json
import glob
import numpy as np

from PIL import Image
from collections import defaultdict

image_size = (28, 28)
status_update_after = 5000 # images processed

user_class = dict()
user_data = defaultdict(dict)

parent_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
image_paths = os.path.join(parent_path, 'data', 'raw_data',  'images_*', '*', '*', '*.png')

for i, character_file in enumerate(glob.glob(image_paths)):
    character_class = '.'.join(character_file.split('/')[-4:-1])
    user_id = character_file.split('/')[-1].split('_')[0]
    # instance_num = character_file.split('/')[-1].split('_')[1].split('.')[0]

    img = Image.open(character_file).resize(image_size, resample=Image.LANCZOS)
    flattened_img = np.array(img.convert('L')).flatten() / 255.

    if user_id not in user_class:
        user_class[user_id] = character_class
        user_data[user_id]['x'] = list()
        user_data[user_id]['y'] = list()
    user_data[user_id]['x'].append(flattened_img.tolist())
    user_data[user_id]['y'].append(user_id)

    if (i+1) % status_update_after == 0:
        print ("{} images converted".format(i+1))

all_data = dict()
all_data['users'] = list(user_class.keys())
all_data['num_samples'] = [ len(user_data[x]['x']) for x in all_data['users'] ]
all_data['user_data'] = user_data

file_name = 'all_data.json'
file_path = os.path.join(parent_path, 'data', 'all_data', file_name)

with open(file_path, 'w') as outfile:
    json.dump(all_data, outfile)
