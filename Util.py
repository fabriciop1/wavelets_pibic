# -*- coding: cp1252 -*-

import numpy as np
import math

class Util:

    def getAccuracy(self, testSet, predictions):     # APENAS 1-NN
        self.acertos = 0
        for i in range (len(testSet)):
            if (testSet[i, -1] == predictions[i]):
                self.acertos = self.acertos + 1
        return (self.acertos / float(len(testSet))) * 100.0
    
