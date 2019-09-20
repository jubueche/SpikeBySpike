
import numpy as np

class MovingAverage:

    def __init__(self, shape):

        self.n = 0
        self.V = np.zeros(shape=shape)


    def update(self, value):
        n = self.n + 1
        n_1 = self.n
        self.V = n_1/n * self.V + 1/n*value
        self.n = n

    def get_value(self):
        return self.V


class EMA:

    def __init__(self,shape,decay = 0.999):

        self._decay = decay
        self._shape = shape
        self._shadow_var = np.zeros(shape=shape)


    def update(self, value):
        if(np.sum(self._shadow_var) == 0):
            self._shadow_var = value
        else:
            self._shadow_var -= (1-self._decay)*(self._shadow_var-value)

    def get_value(self):
        return self._shadow_var