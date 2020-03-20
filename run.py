import numpy as np
from data import dataSet
from model import KDE
import pandas as pd
import sys
import argparse
from time import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataSetName', nargs = 1, help='MNIST OR CIFAR, case insensitive')
    parser.add_argument('sigma', nargs = 1, help='integer value of sigma')
    args = parser.parse_args()
    
    sigma = float(args.sigma[0])
    dataSetName = "".join(args.dataSetName)
    
    data = dataSet(dataSetName)

    print("\n\nEVALUATING GAUSSIAN KDE ON {} DATA AT SIGMA {} \n".format(dataSetName.upper(), sigma))
    
    model = KDE(data.train[0], sigma)
    
    start = time()
    likelihood = model(data.test[0])
    end = time() 
    
    saveDict = {"sigma": [sigma], "likelihood": [likelihood], "timeTaken": [end - start]}
    df = pd.DataFrame.from_dict(saveDict)
    df.to_csv("result/{}_test.csv".format(dataSetName))