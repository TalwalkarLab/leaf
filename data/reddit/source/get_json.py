import math
import json
import os
import pickle

DIR = os.path.join('data', 'reddit_subsampled')
FINAL_DIR = os.path.join('data', 'reddit_json')
FILES_PER_JSON = 10


def merge_dicts(x, y):
	z = x.copy()
	z.update(y)
	return z


def to_leaf_format(some_json, start_idx=0):
	leaf_json = {'users': [], 'num_samples': [], 'user_data': {}}
	new_idx = start_idx
	for u, comments in some_json.items():
		new_idx += 1
		leaf_json['users'].append(str(new_idx))
		leaf_json['num_samples'].append(len(comments))
		
		x = []
		y = []
		for c in comments:
			assert c.author == u
			
			c_x = c.body
			c_y = {
				'subreddit': c.subreddit,
				'created_utc': c.created_utc,
				'score': c.score,
			}

			x.append(c_x)
			y.append(c_y)

		user_data = {'x': x, 'y': y}
		leaf_json['user_data'][str(new_idx)] = user_data

	return leaf_json, new_idx


def files_to_json(files, json_name, start_user_idx=0):
	all_users = {}

	for f in files:
	    f_dir = os.path.join(DIR, f)
	    f_users = pickle.load(open(f_dir, 'rb'))
	    
	    all_users = merge_dicts(all_users, f_users)

	all_users, last_user_idx = to_leaf_format(all_users, start_user_idx)

	with open(os.path.join(FINAL_DIR, json_name), 'w') as outfile:
		json.dump(all_users, outfile)

	return last_user_idx


def main():
	if not os.path.exists(FINAL_DIR):
		os.makedirs(FINAL_DIR)

	files = [f for f in os.listdir(DIR) if f.endswith('.pck')]
	files.sort()
	
	num_files = len(files)
	num_json = math.ceil(num_files / FILES_PER_JSON)

	last_user_idx = 0
	for i in range(num_json):
		cur_files = files[i * FILES_PER_JSON : (i+1) * FILES_PER_JSON]
		print('processing until', (i+1) * FILES_PER_JSON)
		last_user_idx = files_to_json(
			cur_files,
			'reddit_{}.json'.format(i),
			last_user_idx)


if __name__ == '__main__':
    main()



