# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from train import train
import csv

from datasets import load_dataset, load_metric
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    import os
    datasetpath = "C:\\Users\\trowb\\OneDrive\\Documents\\experiments.csv"
    with open(datasetpath, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for i, row in enumerate(reader):
            print("TEST", i)
            train(epochs=int(row[1]), batch_size=1, depth=int(row[3]), lr=float(row[4]), model_type=row[5], test_num=int(row[0]))


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
