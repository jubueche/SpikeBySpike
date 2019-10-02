from brian2 import *

import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

# Attempt at implementing NIPS paper https://papers.nips.cc/paper/4750-learning-optimal-spike-based-representations.pdf

# Start with single neuron experiment


# Build the assumption that dx_hat/dt = -x_hat + T*o(t)
# Where o(t) is the spike train produced by the input x(t)
duration = 1100

def get_impulse_times(duration=550,impulse_width=50,impulse_silence=200, fraction=0.75):
    num_samples = int(duration*fraction)
    times = np.random.randint(low=0,high=duration,size=num_samples)

    allowed_start = np.arange(start=0,stop=duration,step=impulse_width+impulse_silence)
    allowed_indices = np.asarray([np.arange(i,i+impulse_width) for i in allowed_start]).flatten()

    times = [i for i in times if i in allowed_indices]
    times = np.sort(list(dict.fromkeys(times)))
    return np.asarray(times)


times = get_impulse_times(duration=duration, fraction=0.75)
indices = np.zeros(len(times))

T = 1.0
tau = 1*ms
thresh = 1.0
seed(100)
eqs='''
dv/dt = (-v + T*(x) - w*o)/(tau) : 1
o = int(v >= thresh-0.00001) : 1
dol/dt = (-ol + o)/(tau) : 1
dxh/dt = (-xh + T*o)/tau : 1
dw/dt = int(o > 0.5)*(v*ol)/(10*ms) : 1
dx/dt = (-x + 2*rand())/(100*ms) : 1 
#x = 0.1 : 1
'''

group = NeuronGroup(N=1, model=eqs)

sm = StateMonitor(group, variables=True, record=True)

group.w = rand()
group.x = 0.9


run(duration*ms)

# Filter o(t) with a gaussian filter
x_hat = gaussian_filter(np.asarray(sm.o).flatten(), sigma=5)


f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(100,100))
ax1.plot(np.asarray(sm.v).flatten())
ax1.set_title('v(t)')


ax2.plot(np.asarray(sm.w).flatten())
ax2.set_title('w(t)')

ax3.plot(np.asarray(sm.o).flatten())
ax3.set_title('o(t)')

ax4.plot(np.asarray(sm.x).flatten())
ax4.set_title('x(t)')

ax5.plot(np.asarray(sm.xh).flatten())
ax5.plot(np.asarray(sm.x).flatten()-1)
ax5.set_title('x:hat(t)')

err = np.abs(np.asarray(sm.xh).flatten()-np.asarray(sm.x).flatten())
ax6.plot(err)
ax6.set_title('Error |x:hat-x|')


plt.show()
