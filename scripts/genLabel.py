import os
import numpy as np
from PIL import Image
import argparse
import time
import dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some directory.')
    # traindir, testdir, batchsize=6, seqLength=15, step_for_batch=2, step_for_update=5):
    parser.add_argument("--traindir", type=str)
    parser.add_argument("--testdir", type=str)
    parser.add_argument("--batchsize", type=int)
    parser.add_argument("--seqLength", type=int)
    parser.add_argument("--step_for_batch", type=int)
    parser.add_argument("--step_for_update", type=int)
    args = parser.parse_args(["--traindir=/home/gxdai/Guoxian_Dai/data/medicalImage/wustl/TrainingSet", "--testdir=C:\Users\gdai\Downloads\Training Datasets", "--batchsize=5", \
                       "--seqLength=12", "--step_for_batch=2", "--step_for_update=90"])
    medicalData = Dataset(args.traindir, args.testdir, args.batchsize, args.seqLength, args.step_for_batch, args.step_for_update)
    for _ in range(20):
        data, label = medicalData.next_batch('train')
