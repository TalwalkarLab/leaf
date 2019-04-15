import json
import numpy as np
import os
from collections import defaultdict

def batch_data(data, batch_size):
    '''
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    '''
    data_x = data['x']
    data_y = data['y']

    # randomly shuffle data
    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i+batch_size]
        batched_y = data_y[i:i+batch_size]
        yield (batched_x, batched_y)


def gen_frac_query(data, query_fraction=0.1):
    X = data['x']
    y = data['y']

    label_mapping = defaultdict(lambda : [])
    for idx, y_val in enumerate(y):
        label_mapping[y_val].append(idx)

    # Split idxes
    support_fraction = (1. - query_fraction)
    support_idx, query_idx = [], []
    for label, label_idx in label_mapping.items():
        support_set_cnt = int(support_fraction * len(label_idx))
        perm = np.random.permutation(label_idx)
        split = np.split(perm, [support_set_cnt, len(perm)])
        support_idx.extend(split[0])
        query_idx.extend(split[1])

    support_X, support_y = [X[idx] for idx in support_idx], [y[idx] for idx in support_idx]
    query_X, query_y = [X[idx] for idx in query_idx], [y[idx] for idx in query_idx]

    return {'x': support_X, 'y': support_y}, {'x': query_X, 'y': query_y}


def read_dir(data_dir):
    clients = []
    groups = []
    data = defaultdict(lambda : None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    return clients, groups, data

def read_data(train_data_dir, test_data_dir, is_client_split=False):
    '''parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with 
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users
    
    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    train_clients, train_groups, train_data = read_dir(train_data_dir)
    test_clients, test_groups, test_data = read_dir(test_data_dir)

    if is_client_split is False:
        return (train_clients, train_clients), (train_groups, train_groups), train_data, test_data
    print ('Creating split by clients')
    return (train_clients, test_clients), (train_groups, test_groups), train_data, test_data
