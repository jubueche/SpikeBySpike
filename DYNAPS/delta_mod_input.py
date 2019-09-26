from utils import Utils
import numpy as np
import matplotlib.pyplot as plt
from brian2 import *
from pyqtgraph.Qt import QtGui
import pyqtgraph as pg
import sys
import pyqtgraph.exporters

seed(43)
penable = False
utils = Utils.from_default()

F = np.load("Resources/F.dat", allow_pickle=True)

print(F.T)
print("")

conn_x1_high = np.copy(F.T[:,0])
conn_x1_high[conn_x1_high < 0] = 0
conn_x1_high = conn_x1_high / (max(conn_x1_high) - min(conn_x1_high))*10

conn_x1_down = np.copy(F.T[:,0])
conn_x1_down[conn_x1_down >= 0] = 0
conn_x1_down = conn_x1_down*-1
conn_x1_down = conn_x1_down / (max(conn_x1_down) - min(conn_x1_down))*10

conn_x2_high = np.copy(F.T[:,0])
conn_x2_high[conn_x2_high < 0] = 0
conn_x2_high = conn_x2_high / (max(conn_x2_high) - min(conn_x2_high))*10

conn_x2_down = np.copy(F.T[:,0])
conn_x2_down[conn_x2_down >= 0] = 0
conn_x2_down = conn_x2_down*-1
conn_x2_down = conn_x2_down / (max(conn_x2_down) - min(conn_x2_down))*10


connections = np.asarray([conn_x1_high, conn_x1_down, conn_x2_high, conn_x2_down])
print("X1 HIGH        X1 DOWN     X2 HIGH     X2 LOW")
print(connections.T)

# Show a heat map of F
utils.save_F("Resources/F.png", F)

# Get the signal
x = utils.get_matlab_like_input()

x1 = x[0,:]
x2 = x[1,:]

thresh = 0.05
spikes1 = utils.signal_to_spike_refractory(1, np.linspace(0,len(x1)-1,len(x1)), x1, thresh, thresh, 0.0001)
spikes2 = utils.signal_to_spike_refractory(1, np.linspace(0,len(x2)-1,len(x2)), x2, thresh, thresh, 0.0001)

up_1 = spikes1[0]; down_1 = spikes1[1]
up_2 = spikes2[0]; down_2 = spikes2[1]

(up_1_isi, neuron_id_1) = utils.spikes_to_isi(up_1, 1*np.ones(utils.duration), use_microseconds=True)
(down_1_isi, neuron_id_2) = utils.spikes_to_isi(down_1, 2*np.ones(utils.duration), use_microseconds=True)
(up_2_isi, neuron_id_3) = utils.spikes_to_isi(up_2, 3*np.ones(utils.duration), use_microseconds=True)
(down_2_isi, neuron_id_4) = utils.spikes_to_isi(down_2, 4*np.ones(utils.duration), use_microseconds=True)

up_1_isi.dump("Resources/up_1_isi.dat")
down_1_isi.dump("Resources/down_1_isi.dat")
up_2_isi.dump("Resources/up_2_isi.dat")
down_2_isi.dump("Resources/down_2_isi.dat")

###### Plotting spike trains ###### 
if(penable):
        app = QtGui.QApplication.instance()
        if app is None:
                app = QtGui.QApplication(sys.argv)
        else:
                print('QApplication instance already exists: %s' % str(app))

        pg.setConfigOptions(antialias=True)
        labelStyle = {'color': '#FFF', 'font-size': '12pt'}
        win = pg.GraphicsWindow()
        win.resize(1500, 1500)
        win.setWindowTitle('Delta converted spike trains')

        p_signal_1 = win.addPlot(title="Signal x1")
        win.nextRow()
        p_spikes_1_up = win.addPlot(title="Spikes x1 up")
        win.nextRow()
        p_spikes_1_down = win.addPlot(title="Spikes x1 down")
        win.nextRow()
        p_signal_2 = win.addPlot(title="Signal x2")
        win.nextRow()
        p_spikes_2_up = win.addPlot(title="Spikes x2 up")
        win.nextRow()
        p_spikes_2_down = win.addPlot(title="Spikes x2 down")
        win.nextRow()


        p_signal_1.plot(y=x1,pen=pg.mkPen('r', width=1, style=pg.QtCore.Qt.DashLine))

        p_spikes_1_up.plot(x=up_1, y=np.zeros(len(up_1)),
                                pen=None, symbol='o', symbolPen=None,
                                symbolSize=3, symbolBrush=(68, 245, 255))
        p_spikes_1_up.setRange(xRange=[0,utils.duration])

        p_spikes_1_down.plot(x=down_1, y=np.zeros(len(down_1)),
                                pen=None, symbol='o', symbolPen=None,
                                symbolSize=3, symbolBrush=(68, 245, 255))
        p_spikes_1_down.setRange(xRange=[0,utils.duration])

        p_signal_2.plot(y=x2,pen=pg.mkPen('r', width=1, style=pg.QtCore.Qt.DashLine))

        p_spikes_2_up.plot(x=up_2, y=np.zeros(len(up_2)),
                                pen=None, symbol='o', symbolPen=None,
                                symbolSize=3, symbolBrush=(68, 245, 255))
        p_spikes_2_up.setRange(xRange=[0,utils.duration])

        p_spikes_2_down.plot(x=down_2, y=np.zeros(len(down_2)),
                                pen=None, symbol='o', symbolPen=None,
                                symbolSize=3, symbolBrush=(68, 245, 255))
        p_spikes_2_down.setRange(xRange=[0,utils.duration])


        app.exec()