"""Script to run the baselines."""
import argparse
import importlib
import numpy as np
import os
import sys
import random
import tensorflow as tf

import metrics.writer as metrics_writer

from baseline_constants import MAIN_PARAMS, MODEL_PARAMS, SIM_TIMES
from client import Client
from server import Server
from model import ServerModel

from utils.constants import DATASETS
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
    train_clients, test_clients = setup_clients(args.dataset, client_model, args.is_client_split)
    train_ids, train_groups, num_train_samples, _ = server.get_clients_info(train_clients)
    test_ids, test_groups, _, num_test_samples = server.get_clients_info(test_clients)
    print('Clients in Total: %d train, %d test' % (len(train_clients), len(test_clients)))

    # Initial status
    print('--- Round 0 of %d ---' % (num_rounds))
    train_stat_metrics = server.get_train_stats(train_clients)
    print_metrics(train_stat_metrics, num_train_samples, prefix='train_')
    stat_metrics = server.metatest_model(test_clients, query_fraction=0.1, num_epochs=args.num_epochs, batch_size=args.batch_size)
    metrics_writer.print_metrics(0, test_ids, stat_metrics, test_groups, num_test_samples, STAT_METRICS_PATH)
    print_metrics(stat_metrics, num_test_samples, prefix='test_')

    # Simulate training
    for i in range(num_rounds):
        print('--- Round %d of %d: Training %d Clients ---' % (i+1, num_rounds, clients_per_round))

        # Test model on all clients
        if i % eval_every == 0 or i == num_rounds:
            train_stat_metrics = server.get_train_stats(clients)
            print_metrics(train_stat_metrics, all_train_samples, prefix='train_')
            stat_metrics = server.test_model(clients)
            metrics_writer.print_metrics(i, all_ids, stat_metrics, all_groups, all_num_samples, STAT_METRICS_PATH)
            print_metrics(stat_metrics, all_num_samples, prefix='test_')

        # Select clients to train this round
        server.select_clients(i, online(train_clients), num_clients=clients_per_round)
        c_ids, c_groups, c_num_samples, _ = server.get_clients_info(server.selected_clients)

        # Simulate server model training on selected clients' data
        sys_metics = server.train_model(num_epochs=args.num_epochs, batch_size=args.batch_size, minibatch=args.minibatch)

        # Update server model
        server.update_model()
        metrics_writer.print_metrics(i, c_ids, sys_metics, c_groups, c_num_samples, SYS_METRICS_PATH)

        # Test model
        if (i+1) % eval_every == 0 or i == num_rounds:
            train_stat_metrics = server.get_train_stats(train_clients)
            print_metrics(train_stat_metrics, num_train_samples, prefix='train_')
            stat_metrics = server.metatest_model(test_clients, query_fraction=0.1, num_epochs=args.num_epochs, batch_size=args.batch_size)
            metrics_writer.print_metrics((i+1), test_ids, stat_metrics, test_groups, num_test_samples, STAT_METRICS_PATH)
            print_metrics(stat_metrics, num_test_samples, prefix='test_')

    # Save server model
    # save_model(server_model, dataset, model)

    # Close models
    # server_model.close()
    client_model.close()


def online(clients):
    """We assume all users are always online."""
    return clients


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-dataset',
                    help='name of dataset;',
                    type=str,
                    choices=DATASETS,
                    required=True)
    parser.add_argument('-model',
                    help='name of model;',
                    type=str,
                    required=True)
    parser.add_argument('--num-rounds',
                    help='number of rounds to simulate;',
                    type=int,
                    default=-1)
    parser.add_argument('--eval-every',
                    help='evaluate every ____ rounds;',
                    type=int,
                    default=-1)
    parser.add_argument('--clients-per-round',
                    help='number of clients trained per round;',
                    type=int,
                    default=-1)
    parser.add_argument('--batch_size',
                    help='batch size when clients train on data;',
                    type=int,
                    default=10)
    parser.add_argument('--seed',
                    help='seed for random client sampling and batch splitting',
                    type=int,
                    default=0)
    parser.add_argument('--is-client-split', action='store_true',
                    help='data split is according to clients')

    # Minibatch doesn't support num_epochs, so make them mutually exclusive
    epoch_capability_group = parser.add_mutually_exclusive_group()
    epoch_capability_group.add_argument('--minibatch',
                    help='None for FedAvg, else fraction;',
                    type=float,
                    default=None)
    epoch_capability_group.add_argument('--num_epochs',
                    help='number of epochs when clients train on data;',
                    type=int,
                    default=1)

    parser.add_argument('-t',
                    help='simulation time: small, medium, or large;',
                    type=str,
                    choices=SIM_TIMES,
                    default='large')
    parser.add_argument('-lr',
                    help='learning rate for local optimizers;',
                    type=float,
                    default=-1,
                    required=False)

    return parser.parse_args()

def create_clients(users, groups, train_data, test_data, model):
    if len(groups) == 0:
        groups = [[] for _ in users]
    clients = [Client(u, g, train_data[u], test_data[u], model) for u, g in zip(users, groups)]
    return clients

def setup_clients(dataset, model=None, is_client_split=False):
    """Instantiates clients based on given train and test data directories.

    Return:
        all_clients: list of Client objects.
    """
    train_data_dir = os.path.join('..', 'data', dataset, 'data', 'train')
    test_data_dir = os.path.join('..', 'data', dataset, 'data', 'test')

    conv_data = read_data(train_data_dir, test_data_dir, is_client_split)
    (train_users, test_users), (train_groups, test_groups), train_data, test_data = conv_data

    train_clients = create_clients(train_users, train_groups, train_data, test_data, model)
    test_clients = create_clients(test_users, test_groups, train_data, test_data, model)

    return train_clients, test_clients


def save_model(server_model, dataset, model):
    """Saves the given server model on checkpoints/dataset/model.ckpt."""
    # Save server model
    ckpt_path = os.path.join('checkpoints', dataset)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    save_path = server_model.save(os.path.join(ckpt_path, '%s.ckpt' % model))
    print('Model saved in path: %s' % save_path)


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
