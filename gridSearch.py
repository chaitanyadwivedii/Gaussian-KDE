import numpy as np
from data import dataSet
from model import KDE
import pandas as pd
import sys
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataSetName', nargs = 1, help='MNIST OR CIFAR, case insensitive')
    args = parser.parse_args()
    
    dataSetName = "".join(args.dataSetName)
    data = dataSet(dataSetName)
    data.showSamples(numRow = 10, numCol = 10, imgPath = dataSetName)
    
    grid = [0.05, 0.08, 0.1, 0.2, 0.5, 1, 1.5, 2]
    likelihoodList = []
    for sigma in grid:
        print("\n\nEVALUATING GAUSSIAN KDE ON {} DATA AT SIGMA {} \n".format(dataSetName.upper(), sigma))
        model = KDE(data.train[0], sigma)
        likelihood = model(data.val[0])
        likelihoodList.append(likelihood)

    saveDict = {"sigma": grid, "likelihood": likelihoodList}

    df = pd.DataFrame.from_dict(saveDict)
    df.to_csv("result/{}.csv".format(dataSetName))