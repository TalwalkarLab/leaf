import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import re
import string
import html

from nltk.tokenize import TweetTokenizer


DIR = os.path.join('data', 'reddit_merged')
FINAL_DIR = os.path.join('data', 'reddit_clean')

PHRASES_TO_AVOID = [
	'[ deleted ]',
	'[ removed ]',
	'[deleted]',
	'[removed]',
	'bot',
	'thank you for participating',
	'thank you for your submission',
	'thanks for your submission',
	'your submission has been removed',
	'your comment has been removed',
	'downvote this comment if this is',
	'your post has been removed',
]

def clean_file(f, tknzr):
	reddit = pickle.load(open(os.path.join(DIR, f), 'rb'))
	
	clean_reddit = {}
	for u, comments in reddit.items():

		clean_comments = []
		for c in comments:
			c.clean_body(tknzr)
			if len(c.body) > 0 and not any([p in c.body for p in PHRASES_TO_AVOID]):
				clean_comments.append(c)
				
		if len(clean_comments) > 0:
			clean_reddit[u] = clean_comments

	pickle.dump(
		clean_reddit,
		open(os.path.join(FINAL_DIR, f.replace('merged', 'cleaned')), 'wb'))

def main():
	tknzr = TweetTokenizer()

	if not os.path.exists(FINAL_DIR):
		os.makedirs(FINAL_DIR)

	files = [f for f in os.listdir(DIR) if f.endswith('.pck')]
	files.sort()

	num_files = len(files)
	for i, f in enumerate(files):
		clean_file(f, tknzr)
		print('Done with {} of {}'.format(i, num_files))

if __name__ == '__main__':
	main()

