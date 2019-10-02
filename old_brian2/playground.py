import matplotlib.pyplot as plt
import numpy as np
from utils import Utils
from pyqtgraph.Qt import QtGui
import pyqtgraph as pg
import sys
import os
from math import sin
np.random.seed(42)
np.set_printoptions(precision=6, suppress=True) # For the rate vector



Omega = np.load("DYNAPS/Resources/Omega_after.dat", allow_pickle=True)
#print(np.sum((Omega != 0), axis=1))

tmp = Omega.ravel()[Omega.ravel() > 0.0]

plt.hist(Omega.ravel()[Omega.ravel() != 0], 50, density=True, facecolor='green', alpha=0.75)
plt.show()
"""
print(tmp)
print(min(tmp))
print(max(tmp))"""