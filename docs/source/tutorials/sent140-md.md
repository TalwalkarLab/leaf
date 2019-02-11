# Twitter Sentiment Analysis

## Significance of experiments

- Statistical test - vary number of users to see effect on accuracy/performance

## Pre-requisites

- **Dataset**: Generate data with `k` samples per user, run  (`-t sample` required for running baseline impl).

- **GloVe Embeddings**: Setup glove embeddings as a json file (required even for BoW logistic regression since defines vocab dict) - VOCAB_DIR variable in bag_log_reg, sent140/get_embs.sh fetches and extracts embeddings to correct location

## Model Execution

- Run  (2 clients for 10 rounds, converges to accuracy: 0.609676, 10th percentile: 0.25, 90th percentile 1 for k=100). Uses FedAvg for distributed learning (since --minibatch isn’t specified)

- Metrics written out to metrics/stat_metrics.csv and metrics/sys_metrics.csv (configurable via main.py:L20,21)

### Quickstart script

In the root of the LEAF directory, execute `./sent140.sh`

