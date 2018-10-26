"""Script to run the baselines."""

import argparse
import importlib
import numpy as np
import os
import sys
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
    tf.reset_default_graph()
    client_model = ClientModel(*model_params)
    server_model = ServerModel(ClientModel(*model_params))

    # Create server
    server = Server(server_model)

    # Create clients
    clients = setup_clients(args.dataset, client_model)
    print('%d Clients in Total' % len(clients))

    # Test untrained model on all clients
    stat_metrics = server.test_model(clients)
    all_ids, all_groups, all_num_samples = server.get_clients_test_info(clients)
    metrics_writer.print_metrics(0, all_ids, stat_metrics, all_groups, all_num_samples, STAT_METRICS_PATH)
    print_metrics(stat_metrics, all_num_samples)

    # Simulate training
    for i in range(num_rounds):
        print('--- Round %d of %d: Training %d Clients ---' % (i+1, num_rounds, clients_per_round))

        # Select clients to train this round
        server.select_clients(online(clients), num_clients=clients_per_round)
        c_ids, c_groups, c_num_samples = server.get_clients_test_info()

        # Simulate server model training on selected clients' data
        sys_metics = server.train_model(num_epochs=1, batch_size=10)
        metrics_writer.print_metrics(i, c_ids, sys_metics, c_groups, c_num_samples, SYS_METRICS_PATH)

        # Update server model
        server.update_model()

        # Test model on all clients
        if (i + 1) % eval_every == 0 or (i + 1) == num_rounds:
            stat_metrics = server.test_model(clients)
            metrics_writer.print_metrics(i, all_ids, stat_metrics, all_groups, all_num_samples, STAT_METRICS_PATH)
            print_metrics(stat_metrics, all_num_samples)

    # Save server model
    save_model(server_model, args.dataset, args.model)

    # Close models
    server_model.close()
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
    parser.add_argument('--batch-size',
                    help='batch size when clients train on data;',
                    type=int,
                    default=10)
    parser.add_argument('--num-epochs',
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


def setup_clients(dataset, model=None):
    """Instantiates clients based on given train and test data directories.

    Return:
        all_clients: list of Client objects.
    """
    train_data_dir = os.path.join('..', 'data', dataset, 'data', 'train')
    test_data_dir = os.path.join('..', 'data', dataset, 'data', 'test')

    users, groups, train_data, test_data = read_data(train_data_dir, test_data_dir)
    if len(groups) == 0:
        groups = [[] for _ in users]
    all_clients = [Client(u, g, train_data[u], test_data[u], model) for u, g in zip(users, groups)]
    return all_clients


def save_model(server_model, dataset, model):
    """Saves the given server model on checkpoints/dataset/model.ckpt."""
    # Save server model
    ckpt_path = os.path.join('checkpoints', dataset)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    save_path = server_model.save(os.path.join(ckpt_path, '%s.ckpt' % model))
    print('Model saved in path: %s' % save_path)


def print_metrics(metrics, weights):
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
              % (metric,
                 np.average(ordered_metric, weights=ordered_weights),
                 np.percentile(ordered_metric, 10),
                 np.percentile(ordered_metric, 90)))


if __name__ == '__main__':
    main()
