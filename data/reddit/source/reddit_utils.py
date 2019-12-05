import html
import re
from nltk.tokenize import TweetTokenizer

 
URL_TOKEN = '<URL>'
USER_TOKEN = '<USER>'
SUBREDDIT_TOKEN = '<SUBREDDIT>'

URL_REGEX = r'http\S+'
USER_REGEX = r'(?:\/?u\/\w+)'
SUBREDDIT_REGEX = r'(?:\/?r\/\w+)'


class RedditComment:

	def __init__(self, reddit_dict):
		self.body = reddit_dict['body']
		self.author = reddit_dict['author']
		self.subreddit = reddit_dict['subreddit']
		self.subreddit_id = reddit_dict['subreddit_id']
		self.created_utc = reddit_dict['created_utc']
		self.score = reddit_dict['score']

	def clean_body(self, tknzr=None):
		if tknzr is None:
			tknzr = TweetTokenizer()

		# unescape html symbols.
		new_body = html.unescape(self.body)

		# remove extraneous whitespace.
		new_body = new_body.replace('\n', ' ')
		new_body = new_body.replace('\t', ' ')
		new_body = re.sub(r'\s+', ' ', new_body).strip()

		# remove non-ascii symbols.
		new_body = new_body.encode('ascii', errors='ignore').decode()

		# replace URLS with a special token.
		new_body = re.sub(URL_REGEX, URL_TOKEN, new_body)

		# replace reddit user with a token
		new_body = re.sub(USER_REGEX, USER_TOKEN, new_body)

		# replace subreddit names with a token
		new_body = re.sub(SUBREDDIT_REGEX, SUBREDDIT_TOKEN, new_body)

		# lowercase the text
		new_body = new_body.casefold()

		# Could be done in addition:
		# get rid of comments with quotes

		# tokenize the text
		new_body = tknzr.tokenize(new_body)

		self.body = ' '.join(new_body)

	def __str__(self):
		return str(vars(self))
