import matplotlib.pyplot as plt
import numpy as np
from utils import Utils
from pyqtgraph.Qt import QtGui
import pyqtgraph as pg
import sys
import os

import pyqtgraph.exporters



"""app = QtGui.QApplication.instance()
if app is None:
        app = QtGui.QApplication(sys.argv)
else:
        print('QApplication instance already exists: %s' % str(app))

pg.setConfigOptions(antialias=True)
labelStyle = {'color': '#FFF', 'font-size': '12pt'}
win = pg.GraphicsWindow()
win.resize(1500, 1500)
win.setWindowTitle('Learning to represent signals spike-by-spike')

p_spikes = win.addPlot(title="Initial Spikes")
win.nextRow()

p_spikes.plot(y=[1,2,3])

pg.QtGui.QApplication.processEvents()

ex = pg.exporters.ImageExporter(win.scene() )
ex.export(fileName="test.png")
app.exec()"""