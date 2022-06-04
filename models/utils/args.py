import argparse

DATASETS = ['sent140', 'femnist', 'shakespeare', 'celeba', 'synthetic', 'reddit']
SIM_TIMES = ['small', 'medium', 'large']


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
    parser.add_argument('--num_rounds',
                    help='number of rounds to simulate;',
                    type=int,
                    default=-1)
    parser.add_argument('--eval_every',
                    help='evaluate every ____ rounds;',
                    type=int,
                    default=-1)
    parser.add_argument('--clients_per_round',
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
    parser.add_argument('--metrics_name', 
                    help='name for metrics file;',
                    type=str,
                    default='metrics',
                    required=False)
    parser.add_argument('--metrics_dir', 
                    help='dir for metrics file;',
                    type=str,
                    default='metrics',
                    required=False)
    parser.add_argument('--use_val_set', 
                    help='use validation set;', 
                    action='store_true')
    parser.add_argument('--num_epochs',
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
