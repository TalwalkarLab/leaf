import copy
import json
import numpy as np
import os

DATA_DIR = os.path.join('data', 'reddit_json')
FINAL_DIR = os.path.join('data', 'reddit_leaf')
SEQ_LEN = 10


def create_seqs(comments, meta_data, seq_len=SEQ_LEN):
    all_tokens = []
    target_tokens = []
    target_metadata = []
    all_counts = []
    
    for c, meta in zip(comments, meta_data):
        c_tokens = np.array(['<BOS>'] + c.split() + ['<EOS>'], dtype=np.unicode_)
        t_tokens = np.array(c.split() + ['<EOS>', '<PAD>'], dtype=np.unicode_)
        count_tokens = np.ones(c_tokens.shape)
        
        pad_token = '<PAD>'
        
        if len(c_tokens) % seq_len != 0:
        # You need numpy '1.16.4' for np.pad to work as expected.
            c_tokens = np.pad(
                array=c_tokens,
                pad_width=(0, seq_len - len(c_tokens) % seq_len),
                mode='constant',
                constant_values=pad_token)
            count_tokens = np.pad(count_tokens, (0, seq_len - len(count_tokens) % seq_len), 'constant', constant_values=0)
        if len(t_tokens) % seq_len != 0:
            t_tokens = np.pad(t_tokens, (0, seq_len - len(t_tokens) % seq_len), 'constant', constant_values=(pad_token))
        
        c_tokens = c_tokens.reshape(-1, seq_len)
        t_tokens = t_tokens.reshape(-1, seq_len)
        count_tokens = count_tokens.reshape(-1, seq_len)
                
        all_tokens.append(c_tokens.tolist())
        target_tokens.append(t_tokens.tolist())
        target_metadata.append(meta)
        all_counts.append(count_tokens.tolist())
        
    #all_tokens = np.concatenate(all_tokens)
    #target_tokens = np.concatenate(target_tokens)
    #all_counts = np.concatenate(all_counts)
    
    def build_labels_dict(targets, counts, other_meta):
        labels = []
        for i, m in enumerate(other_meta):
            final_meta = copy.deepcopy(m)
            final_meta['target_tokens'] = targets[i]
            final_meta['count_tokens'] = counts[i]
            labels.append(final_meta)
        return labels
        
    return all_tokens, build_labels_dict(target_tokens, all_counts, target_metadata)


def save_json(json_data, set_name, file_name):
    save_dir = os.path.join(FINAL_DIR, set_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print('saving')
    target_file = os.path.join(save_dir, '{}_{}.json'.format(file_name, set_name))
    with open(target_file, 'w') as outfile:
        json.dump(json_data, outfile)


def order_data(user_data):
    zipped = list(zip(user_data['x'], user_data['y']))
    zipped = sorted(zipped, key=lambda x: x[1]['created_utc'])

    return {'x': [z[0] for z in zipped], 'y': [z[1] for z in zipped]}


def create_leaf_json(orig_data):
    return {
        'users': orig_data['users'],
        'num_samples': orig_data['num_samples'],
        'user_data': {}
    }


def process_file(file_name):
    with open(os.path.join(DATA_DIR, file_name)) as json_file:
        data = json.load(json_file)

        train_json = create_leaf_json(data)
        val_json = create_leaf_json(data)
        test_json = create_leaf_json(data)
        
        num_users = len(data['users'])
        print(file_name, 'num_users', num_users)

        for u in data['users']:
            u_data = data['user_data'][u]
            
            ordered_data = order_data(u_data)
            
            num_samples = len(u_data['y'])
            train_thres = int(0.6 * num_samples)
            val_thresh = int(0.8 * num_samples)
                        
            train_seqs, train_labels = create_seqs(
                ordered_data['x'][:train_thres],
                ordered_data['y'][:train_thres])
            train_json['user_data'][u] = {
                'x': train_seqs,
                'y': train_labels,
            }

            val_seqs, val_labels = create_seqs(
                ordered_data['x'][train_thres : val_thresh],
                ordered_data['y'][train_thres : val_thresh])
            val_json['user_data'][u] = {
                'x': val_seqs,
                'y': val_labels,
            }

            test_seqs, test_labels = create_seqs(
                ordered_data['x'][val_thresh:],
                ordered_data['y'][val_thresh:])
            test_json['user_data'][u] = { 
                'x': test_seqs,
                'y': test_labels,
            }
    
        save_json(train_json, 'train', file_name.replace('.json', ''))
        save_json(val_json, 'val', file_name.replace('.json', ''))
        save_json(test_json, 'test', file_name.replace('.json', ''))


def main():
	data_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.json')]
    data_files.sort()
    
	for f in data_files:
	    print(f)
	    process_file(f)


if __name__ == '__main__':
    main()
