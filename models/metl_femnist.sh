set -x
# E=1, B=128, C=3, lr=3e-4
python -u main.py -dataset femnist -model cnn -lr 0.0003 \
    --num_epochs 10 --clients-per-round 2 --num-rounds 600 --batch_size 256 --eval-every 1 --is-client-split | \
    tee logs/femnist_e1_b10_c3.txt
