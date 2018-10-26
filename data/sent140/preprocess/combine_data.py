'''
each row of created .csv file is of the form:
polarity, id, date, query, user, comment, test_or_training
'''

import csv
import os

parent_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

train_file_name = os.path.join(parent_path, 'data', 'raw_data', 'training.csv')

training = []
with open(train_file_name, 'rt', encoding='ISO-8859-1') as f:
    reader = csv.reader(f)
    training = list(reader)

test_file_name = os.path.join(parent_path, 'data', 'raw_data', 'test.csv')

test = []
with open(test_file_name, 'rt', encoding='ISO-8859-1') as f:
    reader = csv.reader(f)
    test = list(reader)

out_file_name = os.path.join(parent_path, 'data', 'intermediate', 'all_data.csv')

with open(out_file_name, 'w') as f:
    writer = csv.writer(f)

    for row in training:
        row.append('training')
        writer.writerow(row)

    for row in test:
        row.append('test')
        writer.writerow(row)
