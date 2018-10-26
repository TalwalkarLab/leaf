'''
helper functions for preprocessing shakespeare data
'''

import json
import os
import re

def __txt_to_data(txt_dir, seq_length=80):
    """Parses text file in given directory into data for next-character model

    Reads text file line by line. For each line, generates data by prepending
    the line with seq_length unknown character symbols and then sliding a
    seq_length-sized window rightwards starting from the leftmost seq_length
    characters. The corresponding datapoint (x,y) for each window is
    (the characters in the window, the character after the window).

    Currently representing unknown character symbol with ' '.

    Args:
        txt_dir: path to text file
        seq_length: length of strings in X
    """
    with open(txt_dir, 'r') as inf:
        all_lines = inf.readlines()
    for i, line in enumerate(all_lines):
        all_lines[i] = re.sub(r"   *", r' ', line)
    all_lines = [" " * seq_length + l for l in all_lines]
    dataX = []
    dataY = []
    for line in all_lines:
        for i in range(0, len(line) - seq_length, 1):
            seq_in = line[i:i + seq_length]
            seq_out = line[i + seq_length]
            dataX.append(seq_in)
            dataY.append(seq_out)
    return dataX, dataY

def parse_data_in(data_dir, users_and_plays_path, raw=False):
    '''
    returns dictionary with keys: users, num_samples, user_data
    raw := bool representing whether to include raw text in all_data
    if raw is True, then user_data key
    removes users with no data
    '''
    with open(users_and_plays_path, 'r') as inf:
        users_and_plays = json.load(inf)
    files = os.listdir(data_dir)
    users = []
    hierarchies = []
    num_samples = []
    user_data = {}
    for f in files:
        user = f[:-4]
        passage = ''
        filename = os.path.join(data_dir, f)
        with open(filename, 'r') as inf:
            passage = inf.read()
        dataX, dataY = __txt_to_data(filename)
        if(len(dataX) > 0):
            users.append(user)
            if raw:
                user_data[user] = {'raw': passage}
            else:
                user_data[user] = {}
            user_data[user]['x'] = dataX
            user_data[user]['y'] = dataY
            hierarchies.append(users_and_plays[user])
            num_samples.append(len(dataY))
    all_data = {}
    all_data['users'] = users
    all_data['hierarchies'] = hierarchies
    all_data['num_samples'] = num_samples
    all_data['user_data'] = user_data
    return all_data
