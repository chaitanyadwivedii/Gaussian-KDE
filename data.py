
import pickle
import numpy as np
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import config





class dataSet(object):
    
    def __init__(self, dataPath):
        
        if "mnist" in dataPath.lower():
            train, _, test = self.getMNISTData(config.MNISTDataPath)
            self.imShape = 28
        elif "cifar" in dataPath.lower():
            train, _, test = self.getCIFARData(config.CIFARDataPath)
            self.imShape = 32
        else:
            print("invalid dataset")
        self.train, self.val, self.test = self.preprocess(train, test)
        
        
    def preprocess(self, train, test):
        numSamples = len(train[0])
        idx = list(range(len(train[0])))
        np.random.shuffle(idx)
        img = train[0][idx]
        if(img.max() > 1.):
            img = img/255
        label = train[1][idx]
        train = (img, label)
        dataSize   = 10000
        newTrainX = train[0][:dataSize]
        newTrainY = train[1][:dataSize]
        newValX = train[0][dataSize: 2*dataSize]
        newValY = train[1][dataSize: 2*dataSize]
        testX = np.array(test[0])
        if(testX.max() > 1):
            testX = (testX/255.0)
        testY = np.array(test[1])
        return (newTrainX, newTrainY), (newValX,newValY ), (testX, testY )
        
    def getMNISTData(self, dataPath):
        f = open(dataPath, "rb")
        u = pickle._Unpickler( f )
        u.encoding = 'latin1'
        return u.load()
    
    def getCIFARData(self, dataPath):
        trainX, trainY = [], []
        files = glob.glob(dataPath+"*")
        for file in files:
            if "test" in file:
                dat=pickle.load(open(file, "rb"),  encoding='bytes')
                testX = np.array(dat[b'data'])
                testY = np.array(dat[b'fine_labels'])
            elif "train" in file:
                dat=pickle.load(open(file, "rb"),  encoding='bytes')
                trainX.extend(dat[b'data'])
                trainY.extend(dat[b'fine_labels'])
        trainX = np.array(trainX)
        trainY = np.array(trainY)
        return (trainX, trainY), None, (testX, testY)
    
    
    ## CODE FOR VISUALISATION ##
    def showSamples(self, numRow, numCol, imgPath=None):
        fig = plt.figure(figsize=(4., 4.))
        grid = ImageGrid(fig, 111,  
                         nrows_ncols=(numRow, numCol),  
                         axes_pad=0.01,
                         )
        for ax, im in zip(grid, self.train[0][:numRow*numCol]):
            
            if self.imShape == 28:
                ax.imshow(im.reshape(self.imShape,self.imShape))
            else:
                ax.imshow(im.reshape(3,self.imShape,self.imShape).transpose(1,2,0))

        plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
        if imgPath is None: 
            plt.show()
        else:
            plt.savefig("result/"+imgPath)
    
    