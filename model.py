import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


class KDE(object):
    def __init__(self, data, sigma):
        self.kernelSize, self.numFeatures = data.shape
        self.kernel = data
        self.p_z = np.log(1/self.kernelSize)
        self.sigma = sigma
        self.sigmaSq = np.square(self.sigma)
        self.scale_1 = 2*self.sigmaSq
        self.scale_2 = - (0.5*self.numFeatures)*np.log(2*np.pi*self.sigmaSq)
        
    def sumOverFeaturesPerSample(self, testData):
        result = np.sum( - np.square(self.kernel - testData) / self.scale_1, -1)
        return result + self.scale_2
    
    def sumOverFeatures(self, testData):
        pool = Pool(processes = cpu_count())
        result = np.array(pool.map(self.sumOverFeaturesPerSample, testData))
        return result
    
    
    def logSumExp(self, data):
        return np.logaddexp.reduce(data, -1)
    
    def getLikelihood(self, data):
        self.querySize = len(data)
        pairWiseLikelihood  = self.sumOverFeatures( data ) + self.p_z # m x k x 1
        likelihoodPerTerm = self.logSumExp(pairWiseLikelihood) # m x 1
        return np.mean(likelihoodPerTerm)
    
    def __call__(self, data):
        return self.getLikelihood(data)