import os

TOTAL_FOLDS = 10
START_FOLD = 0

for i in range(START_FOLD, TOTAL_FOLDS):
    os.system("python train.py {0}".format(i))
