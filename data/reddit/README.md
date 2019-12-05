# Reddit Dataset

We preprocess the Reddit data released by [pushshift.io](https://files.pushshift.io/reddit/) corresponding to December 2017. We perform the following operations:

1. Unescape html symbols.
2. Remove extraneous whitespaces.
3. Remove non-ascii symbols.
4. Replace URLS, reddit usernames and subreddit names with special tokens.
5. Lowercase the text.
6. Tokenize the text (using nltk's TweetTokenizer).

We also remove users and comments that simple heuristics or preliminary inspections mark as bots; and remove users with less than 5 or more than 1000 comments (which account for less than 0.01% of users). We include the code for this preprocessing in the ```source``` folder for reference, but host the preprocessed dataset [here](https://drive.google.com/file/d/1CXufUKXNpR7Pn8gUbIerZ1-qHz1KatHH/view?usp=sharing). We further preprocess the data to make it ready for our reference model (by splitting it into train/val/test sets and by creating sequences of 10 tokens for the LSTM) [here](https://drive.google.com/file/d/1lT1Z0N1weG-oA2PgC1Jak_WQ6h3bu7V_/view?usp=sharing). The vocabulary of the 10 thousand most common tokens in the data can be found [here](https://drive.google.com/file/d/1I-CRlfAeiriLmAyICrmlpPE5zWJX4TOY/view?usp=sharing).

## Setup Instructions

1. To use our reference model, download the data [here](https://drive.google.com/file/d/1PwBpAEMYKNpnv64cQ2TIQfSc_vPbq3OQ/view?usp=sharing) into a ```data``` subfolder in this directory. This is a subsampled version of the complete data. Our reference implementation doesn't yet support training on the [complete dataset](https://drive.google.com/file/d/1lT1Z0N1weG-oA2PgC1Jak_WQ6h3bu7V_/view?usp=sharing), as it loads all given clients into memory.
2. With the data in the appropriate directory, run build the training vocabulary by running ```python build_vocab.py --data-dir ./data/train --target-dir vocab```. 
3. With the data and the training vocabulary, you can now run our reference implementation in the ```models``` directory using a command as the following:
  - ```python3 main.py -dataset reddit -model stacked_lstm --eval-every 10 --num-rounds 100 --clients-per-round 10 --batch-size 5 -lr 5.65 --metrics-name reddit_experiment```
