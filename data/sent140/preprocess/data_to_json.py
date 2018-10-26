import csv
import json
import os

parent_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

data_dir = os.path.join(parent_path, 'data', 'intermediate', 'all_data.csv')

data = []
with open(data_dir, 'rt', encoding='ISO-8859-1') as f:
    reader = csv.reader(f)
    data = list(reader)

data = sorted(data, key=lambda x: x[4])

# ------------
# get # of users in data, and list of users (note automatically sorted)

num_users = 1
cuser = data[0][4]
users = [cuser]

for i in range(len(data)):
    row = data[i]
    tuser = row[4]
    if tuser != cuser:
        num_users += 1
        cuser = tuser
        users.append(tuser)

# ------------
# get # of samples for each user

num_samples = [0 for _ in range(num_users)]
cuser = data[0][4]
user_i = 0

for i in range(len(data)):
    row = data[i]
    tuser = row[4]
    if tuser != cuser:
        cuser = tuser
        user_i += 1
    num_samples[user_i] += 1

# ------------
# create user_data

user_data = {}
row_i = 0

for u in users:
    user_data[u] = {'x': [], 'y': []}

    while ((row_i < len(data)) and (data[row_i][4] == u)):
        row = data[row_i]
        y = 1 if row[0] == "4" else 0
        user_data[u]['x'].append(row[1:])
        user_data[u]['y'].append(y)

        row_i += 1

# ------------
# create .json file

all_data = {}
all_data['users'] = users
all_data['num_samples'] = num_samples
all_data['user_data'] = user_data

file_path = os.path.join(parent_path, 'data', 'all_data', 'all_data.json')

with open(file_path, 'w') as outfile:
    json.dump(all_data, outfile)
