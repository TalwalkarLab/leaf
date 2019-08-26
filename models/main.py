"""Script to run the baselines."""
import argparse
import importlib
import numpy as np
import os
import sys
import random
import tensorflow as tf

import metrics.writer as metrics_writer

from baseline_constants import MAIN_PARAMS, MODEL_PARAMS
from client import Client
from server import Server
from model import ServerModel

from utils.args import parse_args
from utils.model_utils import read_data

STAT_METRICS_PATH = 'metrics/stat_metrics.csv'
SYS_METRICS_PATH = 'metrics/sys_metrics.csv'

def main():

    args = parse_args()

    # Set the random seed if provided (affects client sampling, and batching)
    random.seed(1 + args.seed)
    np.random.seed(12 + args.seed)
    tf.set_random_seed(123 + args.seed)

    model_path = '%s/%s.py' % (args.dataset, args.model)
    if not os.path.exists(model_path):
        print('Please specify a valid dataset and a valid model.')
    model_path = '%s.%s' % (args.dataset, args.model)
    
    print('############################## %s ##############################' % model_path)
    mod = importlib.import_module(model_path)
    ClientModel = getattr(mod, 'ClientModel')

    tup = MAIN_PARAMS[args.dataset][args.t]
    num_rounds = args.num_rounds if args.num_rounds != -1 else tup[0]
    eval_every = args.eval_every if args.eval_every != -1 else tup[1]
    clients_per_round = args.clients_per_round if args.clients_per_round != -1 else tup[2]

    # Suppress tf warnings
    tf.logging.set_verbosity(tf.logging.WARN)

    # Create 2 models
    model_params = MODEL_PARAMS[model_path]
    if args.lr != -1:
        model_params_list = list(model_params)
        model_params_list[0] = args.lr
        model_params = tuple(model_params_list)

    # Create client model, and share params with server model
    tf.reset_default_graph()
    client_model = ClientModel(args.seed, *model_params)

    # Create server
    server = Server(client_model)

    # Create clients
    train_clients, test_clients = setup_clients(args.dataset, client_model)
    train_ids, train_groups, num_train_samples, _ = server.get_clients_info(train_clients)
    test_ids, test_groups, _, num_test_samples = server.get_clients_info(test_clients)
    print('Clients in Total: %d train, %d test' % (len(train_clients), len(test_clients)))

    # Initial status
    print('--- Round 0 of %d ---' % (num_rounds))
    train_stat_metrics = server.get_train_stats(train_clients)
    print_metrics(train_stat_metrics, num_train_samples, prefix='train_')
    stat_metrics = server.test_model(test_clients)
    metrics_writer.print_metrics(0, test_ids, stat_metrics, test_groups, num_test_samples, STAT_METRICS_PATH)
    print_metrics(stat_metrics, num_test_samples, prefix='test_')

    # Simulate training
    for i in range(num_rounds):
        print('--- Round %d of %d: Training %d Clients ---' % (i + 1, num_rounds, clients_per_round))

        # Select clients to train this round
        server.select_clients(i, online(train_clients), num_clients=clients_per_round)
        c_ids, c_groups, c_num_samples, _ = server.get_clients_info(server.selected_clients)

        # Simulate server model training on selected clients' data
        sys_metics = server.train_model(num_epochs=args.num_epochs, batch_size=args.batch_size, minibatch=args.minibatch)

        # Update server model
        server.update_model()
        metrics_writer.print_metrics(i, c_ids, sys_metics, c_groups, c_num_samples, SYS_METRICS_PATH)

        # Test model
        if (i + 1) % eval_every == 0 or (i + 1) == num_rounds:
            train_stat_metrics = server.get_train_stats(train_clients)
            print_metrics(train_stat_metrics, num_train_samples, prefix='train_')
            test_stat_metrics = server.test_model(test_clients)
            metrics_writer.print_metrics((i + 1), test_ids, stat_metrics, test_groups, num_test_samples, STAT_METRICS_PATH)
            print_metrics(stat_metrics, num_test_samples, prefix='test_')

    # Save server model
    ckpt_path = os.path.join('checkpoints', args.dataset)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    save_path = server.save_model(os.path.join(ckpt_path, '{}.ckpt'.format(args.model)))
    print('Model saved in path: %s' % save_path)

    # Close models
    server.close_model()

def online(clients):
    """We assume all users are always online."""
    return clients


def create_clients(users, groups, train_data, test_data, model):
    if len(groups) == 0:
        groups = [[] for _ in users]
    clients = [Client(u, g, train_data[u], test_data[u], model) for u, g in zip(users, groups)]
    return clients


def setup_clients(dataset, model=None):
    """Instantiates clients based on given train and test data directories.

    Return:
        all_clients: list of Client objects.
    """
    train_data_dir = os.path.join('..', 'data', dataset, 'data', 'train')
    test_data_dir = os.path.join('..', 'data', dataset, 'data', 'test')

    conv_data = read_data(train_data_dir, test_data_dir)
    users, groups, train_data, test_data = conv_data

    train_clients = create_clients(users, groups, train_data, test_data, model)
    test_clients = create_clients(users, groups, train_data, test_data, model)

    return train_clients, test_clients


def print_metrics(metrics, weights, prefix=''):
    """Prints weighted averages of the given metrics.

    Args:
        metrics: dict with client ids as keys. Each entry is a dict
            with the metrics of that client.
        weights: dict with client ids as keys. Each entry is the weight
            for that client.
    """
    ordered_weights = [weights[c] for c in sorted(weights)]
    metric_names = metrics_writer.get_metrics_names(metrics)
    for metric in metric_names:
        ordered_metric = [metrics[c][metric] for c in sorted(metrics)]
        print('%s: %g, 10th percentile: %g, 90th percentile %g' \
              % (prefix + metric,
                 np.average(ordered_metric, weights=ordered_weights),
                 np.percentile(ordered_metric, 10),
                 np.percentile(ordered_metric, 90)))


if __name__ == '__main__':
    main()
