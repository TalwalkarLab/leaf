import math
import os
import pickle
import random

from reddit_utils import RedditComment


DIR = os.path.join('data', 'reddit_raw')
USERS_PER_REPEAT = 200000
USERS_PER_FILE = 20000
FINAL_DIR = os.path.join('data', 'reddit_merged')

if not os.path.exists(FINAL_DIR):
    os.makedirs(FINAL_DIR)

files = [f for f in os.listdir(DIR) if f.endswith('.pck')]
files.sort()

all_users = {}

for f in files:
    f_path = os.path.join(DIR, f)
    users = pickle.load(open(f_path, 'rb'))
    users = list(users.keys())
    
    for u in users:
        if u not in all_users:
            all_users[u] = []
        all_users[u].append(f)


user_keys = list(all_users.keys())
random.seed(3760145)
random.shuffle(user_keys)

num_users = len(all_users)

num_lots = (num_users // USERS_PER_REPEAT) + 1
print('num_lots', num_lots)

cur_file = 1
for l in range(num_lots):
    min_idx, max_idx = l * USERS_PER_REPEAT, min((l + 1) * USERS_PER_REPEAT, num_users)    
    
    cur_user_keys = user_keys[min_idx:max_idx]
    num_cur_users = len(cur_user_keys)
    cur_users = {u: [] for u in cur_user_keys}
    
    for f in files:
        f_path = os.path.join(DIR, f)
        users = pickle.load(open(f_path, 'rb'))
        
        for u in cur_users:
            if f not in all_users[u]:
                continue

            cur_users[u].extend([c for c in users[u] if len(c.body) > 0])
    
    written_users = 0
    while written_users < num_cur_users:
        low_bound, high_bound = written_users, min(written_users + USERS_PER_FILE, num_cur_users)
        file_keys = cur_user_keys[low_bound:high_bound]
        file_users = {u: cur_users[u] for u in file_keys if len(cur_users[u]) >= 5}
        
        pickle.dump(file_users, open(os.path.join(FINAL_DIR, 'reddit_users_merged_{}.pck'.format(cur_file)), 'wb')) 
        
        written_users += USERS_PER_FILE
        cur_file += 1
    
    print(l + 1)

