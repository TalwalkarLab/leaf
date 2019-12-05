import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle


DIR = os.path.join('data', 'reddit_clean')
FINAL_DIR = os.path.join('data', 'reddit_subsampled')


def subsample_file(f):
	reddit = pickle.load(open(os.path.join(DIR, f), 'rb'))
	
	subsampled_reddit = {}
	for u, comments in reddit.items():

		subsampled_comments = [c for c in comments if len(c.body.split()) >= 5]
	
		if len(subsampled_comments) >= 5 and len(subsampled_comments) <= 1000:
			subsampled_reddit[u] = subsampled_comments

	pickle.dump(
		subsampled_reddit,
		open(os.path.join(FINAL_DIR, f.replace('cleaned', 'subsampled')), 'wb'))


def main():
	if not os.path.exists(FINAL_DIR):
		os.makedirs(FINAL_DIR)

	files = [f for f in os.listdir(DIR) if f.endswith('.pck')]
	files.sort()

	num_files = len(files)
	for i, f in enumerate(files):
		subsample_file(f)
		print('Done with {} of {}'.format(i, num_files))

if __name__ == '__main__':
	main()

