SIM_TIMES = ['small', 'medium', 'large']

MAIN_PARAMS = { # (tot_num_rounds, eval_every_num_rounds, clients_per_round)
    'sent140': {
        'small': (10, 2, 2),
        'medium': (16, 2, 2),
        'large': (24, 2, 2)
        },
    'femnist': {
        'small': (30, 10, 2),
        'medium': (100, 10, 2),
        'large': (400, 20, 2)
        },
    'shakespeare': {
        'small': (6, 2, 2),
        'medium': (8, 2, 2),
        'large': (20, 1, 2)
        }
}
MODEL_PARAMS = {
    'sent140.bag_dnn': (0.0003, 2), # lr, num_classes
    'sent140.stacked_lstm': (0.0003, 25, 2, 100), # lr, seq_len, num_classes, num_hidden
    'sent140.bag_log_reg': (0.0003, 2), # lr, num_classes
    'femnist.cnn': (0.0003, 62), # lr, num_classes
    'shakespeare.stacked_lstm': (0.0003, 80, 53, 256) # lr, seq_len, num_classes, num_hidden
}

ACCURACY_KEY = 'accuracy'
BYTES_WRITTEN_KEY = 'bytes_written'
BYTES_READ_KEY = 'bytes_read'
LOCAL_COMPUTATIONS_KEY = 'local_computations'
NUM_ROUND_KEY = 'round_number'
NUM_SAMPLES_KEY = 'num_samples'
CLIENT_ID_KEY = 'client_id'
