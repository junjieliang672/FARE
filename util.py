import numpy as np
def getIter(it, df):
    try:
        x, y, z = it.__next__()
    except:
        it = iter(df)
        x, y, z = it.__next__()
    return x, y, z


class MovingAverage:
    def __init__(self,value=None):
        self.value = []
        if value:
            self.value.append(value)

    def get(self):
        return np.mean(self.value)

    def add(self,value):
        self.value.append(value)

    def size(self):
        return len(self.value)