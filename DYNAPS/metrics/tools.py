"""
General purpose tools. Some of the functions in this submodule
depend on Brian (http://briansimulator.org) to work, as they
are used to generate inputs or run simulations using the
simulator.

TODO: List functions and classes and finish docstrings.
"""
from __future__ import print_function
import sys
import os
from pickle import load
import collections
from numpy import (array, diff, floor, zeros, mean, std, shape,
                   random, cumsum, histogram, where, arange, divide, exp,
                   count_nonzero, bitwise_and, append, ones, flatnonzero,
                   ndarray, convolve, linspace, empty, copy)
from matplotlib.mlab import normpdf
import random as rnd
from warnings import warn
import gc

# TODO: Compare gen_input_groups with SynchronousInputGroup
#       - Unnecessary? fast_synchronous_input_gen
# TODO: Major cleanup required
# TODO: Put brian-dependent functions into a separate file

# brian optional dependency
try:
    import brian
    nobrian = False
    units = brian.units
    ms = msecond = brian.msecond
    volt = brian.volt
    mV = mvolt = brian.mvolt
    second = brian.second
    Hz = hertz = brian.hertz
    PoissonGroup = brian.PoissonGroup
    NeuronGroup = brian.NeuronGroup
    Network = brian.Network
    Connection = brian.Connection
    PulsePacket = brian.PulsePacket
    SpikeGeneratorGroup = brian.SpikeGeneratorGroup
    SpikeMonitor = brian.SpikeMonitor
    StateMonitor = brian.StateMonitor
    plot = brian.plot
    clear = brian.clear
except ImportError:
    nobrian = True
    msecond = 0.001
    second = 1
    volt = 1


class SynchronousInputGroup:
    '''
    Synchronous input generator.
    This class is a generator that generates spike trains where:
    - N*sync spike trains are synchronous,
    - N*(1-sync) spike trains are independent Poisson realisations,
    - the spikes in the n*s synchronous spike trains are shifted by
    Gaussian jitter with standard deviation = `sigma`.

    Constructor Parameters
    ----------
    N : int
        Number of spike trains

    rate : brian Hz (frequency)
        Spike rate for each spike train (units: freq)

    sync : float
        Proportion of synchronous spike trains [0,1]

    jitter : brian second (time)
        Standard deviation of Gaussian random variable which is used to shift
        each spike in a synchronous spike train (units: time)

    dt : brian second (time)
        Simulation time step (default 0.1 ms)
    '''

    def __init__(self, N, rate, synchrony, jitter, dt=0.0001*second):
        self.N = N
        self.rate = rate
        self.sync = synchrony
        self.jitter = jitter
        self._dt = dt
        self._gen = self.configure_generators(self.N, self.rate, self.sync,
                                              self.jitter, self._dt)

    def __call__(self):
        return self._gen

    def configure_generators(self, n, rate, sync, jitter, dt=0.0001*second):
        '''
        Synchronous input generator. This function is called by the parent
        class constructor. See the parent class' doc string for details.

        Parameters
        ----------
        n : int
            Number of spike trains

        rate : brian Hz (frequency)
            Spike rate for each spike train (units: freq)

        sync : float
            Proportion of synchronous spike trains [0,1]

        jitter : brian second (time)
            Standard deviation of Gaussian random variable which is used to
            shift each spike in a synchronous spike train (units: time)

        dt : brian second (time)
            Simulation time step (default 0.1 ms)

        Returns
        -------
        A list of generators with the specific temporal characeteristics
        (see parent constructor for details)

        '''
        if not(0 <= sync <= 1):
            warn("Synchrony should be between 0 and 1. Setting to 0")
            sync = 0

        spiketrains = []
        n_ident = int(floor(n*sync))   # number of identical spike trains
        st_ident = self.sync_inp_gen(n_ident, rate, jitter, dt)
        for i in range(n_ident):
            spiketrains.append(st_ident)

        for i in range(n_ident, n):
            spiketrains.append(self.sync_inp_gen(1, rate, 0*second, dt))

        return spiketrains

    def sync_inp_gen(self, n, rate, jitter, dt=0.0001*second):
        '''
        The synchronous input generator.
        This generator returns the same value (plus a random variate) n times
        before returning a new spike time. This can be used to generate
        n spike trains with the same base spike times, but with jitter applied
        to each spike.
        The n spike trains generated are equivalent to having a series of
        Gaussian pulse packets centred on each base time (t) with
        spread = jitter.

        Parameters
        ----------
        n : int
            Number of synchronous inputs. This defines the number of times
            the same base time (t) will used by the generator.

        rate : brian hertz (frequency)
            The average spike rate

        jitter : brian second (time)
            The jitter applied to each t

        dt : biran second (time)
            Simulation time step (default 0.1 ms)
        '''

        t = 0*second
        prev_t = t
        iter = n
        while(True):
            if iter == n:
                interval = random.exponential(1./rate)*second
                if interval < dt:  # prevents spike stacking
                    interval = dt
                prev_t = t
                t += interval
                iter = 1
            else:
                iter += 1
            if (jitter > 0*second):
                jitt_variate = random.normal(0, jitter)*second
                if (t + jitt_variate) < (prev_t + dt):
                    # limits stacking warnings but has similar effect
                    jitt_variate = prev_t + dt - t
            else:
                jitt_variate = 0*second
            yield t + jitt_variate


def gen_input_groups(N_in, f_in, S_in, sigma, duration):
    """
    Generate two input groups, one for synchronous spikes and the other for
    random (Poisson), independent spiking.
    """
    if nobrian:
        print("Error: gen_input_groups requires Brian", file=sys.stderr)
        return None, None
    N_sync = int(N_in*S_in)
    N_rand = N_in-N_sync
    syncGroup = PoissonGroup(0, 0)  # dummy nrngrp
    randGroup = PoissonGroup(0, 0)
    if N_sync:
        pulse_train = poisson_times(f_in, duration)
        sync_spikes = []
        pp = PulsePacket(0*second, 1, 0*second)  # dummy pp
        for pt in pulse_train:
            if sigma > 0:
                try:
                    pp.generate(t=pt, n=N_sync, sigma=sigma)
                    sync_spikes.extend(pp.spiketimes)
                except ValueError:
                    continue
            else:
                for idx in range(N_sync):
                    sync_spikes.append((idx, pt*second))
        syncGroup = SpikeGeneratorGroup(N=N_sync, spiketimes=sync_spikes)
    if N_rand:
        randGroup = PoissonGroup(N_rand, rates=f_in)
    return syncGroup, randGroup

def poisson_times(f_in, duration):
    poisstrain = []
    spiketime = rnd.expovariate(f_in)
    while float(spiketime) < float(duration):
        poisstrain.append(spiketime)
        spiketime += rnd.expovariate(f_in)
    return array(poisstrain)

def fast_synchronous_input_gen(N_in, f_in, S_in, sigma, duration, shuffle=False):
    """
    Generate combination of synchronous and random spike trains.

    Generate an input group which contains `N_in*S_in` synchronous spike trains,
    with jitter `sigma`, and `1-N_in*S_in` independent (Poisson) spike trains.
    Setting `shuffle=True` will shuffle the order of the spike trains so that
    the synchronous spike trains don't appear in order in spike rasters.

    Parameters
    ----------
    N_in : Total number of spike trains
    f_in : Average spike frequency of each spike train
    S_in : Proportion of synchronous spike trains [0-1]
    sigma : Temporal jitter applied to synchronous spike trains
    duration : Duration of the generated spike trains
    shuffle : Shuffle order in which the spike trains appear (default False)

    Returns
    -------
    SpikeGeneratorGroup (brian object)

    TODO:
        - This could become a SpikeGeneratorGroup derived class constructor
        - Rename function: Remove 'fast' from the name, since it's the default
        now.
    """
    if nobrian:
        print("Error: fast_synchronous_input_gen requires Brian",
                file=sys.stderr)
        return None
    N_sync = int(N_in*S_in)
    N_rand = N_in-N_sync
    spikearray = empty(N_in, dtype=object)
    for trainidx in range(0, N_rand):
        spikearray[trainidx] = poisson_times(f_in, duration)
    if N_sync:
        synctrain = poisson_times(f_in, duration)
        for trainidx in range(N_rand, N_in):
            spikearray[trainidx] = copy(synctrain)
            if sigma:
                jitter = random.normal(0, sigma, len(synctrain))
                spikearray[trainidx] += jitter
    # convert to [(i, t) ...] format for SpikeGeneratorGroup
    # also filter negative spike times
    spiketuples = []
    if shuffle:
        random.shuffle(spikearray)
    for idx, spiketimes in enumerate(spikearray):
        spiketuples.extend([(idx, st) for st in spiketimes if st > 0])
    return SpikeGeneratorGroup(N=N_in, spiketimes=spiketuples)

def _run_calib(nrndef, N_in, f_in, w_in, input_configs, active_idx=None):
    if active_idx is None:
        active_idx = arange(len(input_configs))
    clear(True)
    gc.collect()
    brian.defaultclock.reinit()
    #eqs = nrndef['eqs']
    # V_th = nrndef['V_th']
    #refr = nrndef['refr']
    #reset = nrndef['reset']
    nrngrp = NeuronGroup(N=len(input_configs), **nrndef)
    calib_duration = 500*msecond
    calib_network = Network(nrngrp)
    #syncConns = []
    #randConns = []
    inp_conns = []
    active_configs = array(input_configs)[active_idx]
    for idx, (sync, jitter) in zip(active_idx, active_configs):
        #sg, rg = gen_input_groups(N_in, f_in[idx]*hertz, sync, jitter,
        #                          calib_duration)
        #if len(sg):
        #    sConn = Connection(sg, nrngrp[idx], state='V', weight=w_in)
        #    syncConns.append(sConn)
        #    calib_network.add(sg, sConn)
        #if len(rg):
        #    rConn = Connection(rg, nrngrp[idx], state='V', weight=w_in)
        #    randConns.append(rConn)
        #    calib_network.add(rg, rConn)
        inp_grp = fast_synchronous_input_gen(N_in, f_in[idx]*hertz, sync,
                                             jitter, calib_duration)
        inp_conn = Connection(inp_grp, nrngrp[idx], state='V', weight=w_in)
        inp_conns.append(inp_conn)
        calib_network.add(inp_grp, inp_conn)
    st_mon = SpikeMonitor(nrngrp)
    v_mon = StateMonitor(nrngrp, 'V', record=True)
    calib_network.add(st_mon, v_mon)
    calib_network.run(calib_duration)
    actual_f_out = array([1.0*len(spikes)/calib_duration
                          for spikes in st_mon.spiketimes.itervalues()])
    # del probably unnecessary
    # del(calib_network, syncConns, randConns, st_mon)
    del(calib_network)
    return actual_f_out


def _calc_rate_of_change(X, Y):
    if all(Y > 0):
        return X/Y
    gtz = Y > 0
    ret = zeros(len(X))
    ret[gtz] = X[gtz]/Y[gtz]
    ret[~gtz] = X[~gtz]
    return ret


def calibrate_frequencies(nrndef, N_in, w_in, synchrony_conf, f_out,
                          Vth=None, tau=None, maxtries=None):
    '''
    Calculates the input frequency required to produce the desired output rate
    by assuming a linear relationship and iteratively updating and retesting
    the input frequency on a short simulation.

    If both Vth and tau are supplied, the initial frequency values are
    calculated assuming the LIF equation, which should make the calibration
    faster.

    Requires Brian.

    Parameters
    ----------
    nrndef          : passed directly to NeuronGroup as a dictionary
    N_in            : number of inputs
    w_in            : input weight
    synchrony_conf  : list of tuples (sync, jitter)
    f_out           : desired output frequency
    Vth             : used to calculate initial value for search (optional)
    tau             : used to calculate initial value for search (optional)
    maxtries        : if supplied, limits the number of attempts (optional)

    Returns
    -------
    f_in            : array of input frequencies of the same length
                    as input_configs
    '''
    if nobrian:
        print("Error: calibrate_frequencies requires Brian", file=sys.stderr)
        return -1
    if maxtries is None:
        maxtries = -1
    desired_out = f_out
    Nsims = len(synchrony_conf)
    if (Vth is None) or (tau is None):
        f_in = f_out
    else:
        f_in = Vth/((1-exp(-1.0/(tau*f_out)))*N_in*w_in*tau)
    f_in = ones(Nsims)*f_in
    actual_out = _run_calib(nrndef, N_in, f_in, w_in, synchrony_conf)
    # print(("f_in:  "+"{:5.1f} "*len(f_in)).format(*f_in))
    # print(("f_out: "+"{:5.1f} "*len(actual_out)).format(*actual_out))
    found = abs(desired_out-actual_out) <= max(2*Hz, 0.1*desired_out)  # 10% margin
    # print("Calibrating {} simulations ...".format(Nsims))
    # print("{}/{}".format(sum(found), Nsims), end="")
    sys.stdout.flush()
    fstep = zeros(Nsims)+10
    prevdir = 1*(actual_out < desired_out)-1*(actual_out > desired_out)
    ntry = 0
    while not all(found) and (ntry != maxtries):
        ntry += 1
        fstep[found] = 0
        newdir = 1*(actual_out < desired_out)-1*(actual_out > desired_out)
        fstep[newdir != prevdir] *= 0.5  # direction change
        df_in = fstep*newdir
        prevdir = newdir
        f_in += df_in
        actual_out = _run_calib(nrndef, N_in, f_in, w_in, synchrony_conf,
                                flatnonzero(~found))
        # print(("f_in:  "+"{:5.1f} "*len(f_in)).format(*f_in))
        # print(("f_out: "+"{:5.1f} "*len(actual_out)).format(*actual_out))
        # print()
        found = found | (abs(desired_out-actual_out) <= max(2*Hz,
                                                            0.1*desired_out))
        # print("\r{}/{} {}".format(sum(found), Nsims, "."*ntry), end="")
        # sys.stdout.flush()
        found = found | (f_in > 500)
        f_in[f_in > 800] = 0
    # print("\r{}/{} {}".format(sum(found), Nsims, "."*ntry), end="")
    # print("\nDone!")
    return f_in


def loadsim(simname):
    '''
    Takes a simulation name and loads the data associated with the simulation.
    Searches current directory for:
        "simname.mem" (membrane potential data)
        "simname.out" (output spike trains)
        "simname.stm" (input spike trains)

    If any of the above is not found, the function returns an empty list
    for the respective variable.

    NOTE: Best to use brian.tools.datamanager.DataManager object where possible

    Parameters
    ----------
    simname : string
        The name of the file containing the simulation data
        (excluding extensions)

    Returns
    -------
    mem, spiketrain, stm : numpy arrays
        Simulation data

    '''

    memname = simname+".mem"
    if (os.path.exists(memname)):
        mem = load(open(memname, "rb"))
    else:
        mem = array([])

    outname = simname+".out"
    if (os.path.exists(outname)):
        spiketrain = load(open(outname, "rb"))
    else:
        spiketrain = array([])

    stmname = simname+".stm"
    if (os.path.exists(stmname)):
        stm = load(open(stmname, "rb"))
    else:
        stm = array([])

    if ((not mem.size) and (not spiketrain.size) and (not stm.size)):
        warn("No simulation data exists with name `{}` \n".format(simname))

    return mem, spiketrain, stm


def slope_distribution(v, w, rem_zero=True):
    '''
    Calculates the distribution of membrane potential slope values.

    Parameters
    ----------
    v : numpy array
        Membrane potential values as taken from brian.StateMonitor
    w : float or brian voltage
        Precision of distribution. Slope values are grouped based on the size
        of w and considered equal.
    rem_zero : bool, optional
        If True, the function ignores slope values equal to zero,
        which are caused by refractoriness and are of little interest.

    Returns
    -------
    dist : array
        The values of the distribution. See numpy.histogram for more
        information on the shape of the return array.

    See Also
    --------
    histogram
    '''

    dv = diff(v)
    if (rem_zero):
        dv = dv[dv != 0]

    nbins = (max(dv)-min(dv))/w
    nbins = int(nbins)
    dist = histogram(dv, nbins)
    return dist


def positive_slope_distribution(v, w):
    '''
    Calculates the distribution of positive membrane potential slope values.

    Parameters
    ----------
    v : numpy array
        Membrane potential values as taken from brian.StateMonitor
    w : float or brian voltage
        Precision of distribution. Slope values are grouped based on the size
        of w and considered equal (histogram bin size).

    Return
    -------
    dist : array
        The values of the distribution. See numpy.histogram for more
        information on the shape of the return array.

    See Also
    --------
    numpy.histogram
    '''

    dv = diff(v)
    dv = dv[dv > 0]
    nbins = (max(dv)-min(dv))/w
    nbins = int(nbins)
    dist = histogram(dv, nbins)
    return dist


def get_slope_bounds(spiketrain, v0, vr, vth, tau, dt):
    duration = spiketrain[-1]
    duration_dt = int(duration/dt)
    time_since_spike = ones(duration_dt)*10000
    low_input = zeros(duration_dt)-vr
    for prv, nxt in zip(spiketrain[:-1], spiketrain[1:]):
        prv_dt = int(prv/dt)
        nxt_dt = int(nxt/dt)
        isi_dt = nxt_dt-prv_dt
        time_since_spike[prv_dt:nxt_dt] = arange(isi_dt)*dt
        low_input[prv_dt:nxt_dt] = ones(isi_dt)*(vth-vr) /\
            (1-exp(-(nxt-prv)/tau))
    high_bound = v0+(vr-v0)*exp(-time_since_spike/tau)
    low_bound = vr+low_input*(1-exp(-time_since_spike/tau))
    return high_bound, low_bound


def pre_spike_slopes(mem, spiketrain, vth, w, dt=0.1*msecond):
    # duration = spiketrain[-1]
    # duration_dt = int(duration/dt)
    spiketrain_dt = (spiketrain/dt).astype(int)
    w_dt = int(w/dt)
    pre_spike_mem = mem[spiketrain_dt-w_dt]
    pre_spike_slopes = (vth-pre_spike_mem)/w
    return pre_spike_slopes


def npss(mem, spiketrain, v0, vth, tau, w, dt=0.1*msecond):
    """
    Calculate the normalised pre-spike slopes for the given data.

    Arguments
    ---------
    mem : membrane potential trace
    spiketrain : the spike train produced by the supplied membrane potential
    v0 : resting potential of the neuron
    vth : firing threshold
    tau : membrane leak time constant
    w : coincidence window
    dt : simulation time step (default: 0.1 ms)

    Returns
    -------
    List of values that represent the normalised pre-spike slope for each of
    the output spikes (except the first spike)
    """
    # Cast everything to float to avoid dimension errors
    v0 = float(v0)
    vth = float(vth)
    tau = float(tau)
    w = float(w)
    dt = float(dt)
    w_dt = int(w/dt)
    first_spike = spiketrain[0]
    first_spike_dt = int(first_spike/dt)
    vr = mem[first_spike_dt+1]  # reset potential
    duration = spiketrain[-1]
    duration_dt = int(duration/dt)
    time_since_spike = ones(duration_dt)*10000  # arbitrary large number
    low_input = zeros(duration_dt)-vr
    for prv, nxt in zip(spiketrain[:-1], spiketrain[1:]):
        # can probably vectorise or parallelise this bit
        prv_dt = int(prv/dt)
        nxt_dt = int(nxt/dt)
        isi_dt = nxt_dt-prv_dt
        time_since_spike[prv_dt:nxt_dt] = arange(isi_dt)*dt
        low_input[prv_dt:nxt_dt] = ones(isi_dt)*(vth-vr) /\
            (1-exp(-(nxt-prv)/tau))
    high_bound = v0+(vr-v0)*exp(-time_since_spike/tau)
    low_bound = vr+low_input*(1-exp(-time_since_spike/tau))
    spiketrain_dt = (spiketrain/dt).astype(int)
    window_starts_dt = spiketrain_dt-w_dt
    # there's some redundant processing here for clarity
    # all values, mem, low and high bound at (t_i-w) are converted to slopes by
    # calculating (vth-x)/w, which means we could just avoid it and the
    # normalisation would still work.
    high_slopes = (vth-high_bound[window_starts_dt])/w
    low_slopes = (vth-low_bound[window_starts_dt])/w
    mem_slopes = (vth-mem[window_starts_dt])/w

    # Let's avoid div by zero by adding a tiny value to high_slopes where
    # necessary. Not the most correct solution, but wont affect results. OTOH,
    # we can just drop them, but that would cause issues with spike counts vs
    # slope counts
    dbz_idx = high_slopes == low_slopes
    high_slopes[dbz_idx] += 1e-10
    norm_slopes = (mem_slopes-low_slopes)/(high_slopes-low_slopes)
    norm_slopes[norm_slopes < 0] = 0  # this should be fixed
    return norm_slopes


def sta(v, spiketrain, w, dt=0.0001*second):
    '''
    Calculates the Spike Triggered Average (currently only membrane potential)
    of the supplied data. Single neuron data only.
    This is the average waveform of the membrane potential in a period `w`
    before firing. The standard deviation and the individual windows are also
    returned.

    Parameters
    ----------
    v : numpy array
        Membrane potential values as taken from brian.StateMonitor
    spiketrain : numpy array
        Spike train of the membrane potential data in `v` as taken
        from brian.SpikeMonitor
    w : brian second (time)
        The pre-spike time window
    dt : brian second (time)
        Simulation time step (default 0.1 ms)

    Returns
    -------
    sta_avg : numpy array
        The spike triggered average membrane potential

    sta_std : numpy array
        The standard deviation of the sta_avg

    sta_wins : numpy array
        Two dimensional array containing all the pre-spike membrane potential
        windows
    '''

    if (len(spiketrain) <= 1):
        sta_avg = array([])
        sta_std = array([])
        sta_wins = array([])
        return sta_avg, sta_std, sta_wins

    w_d = int(w/dt)  # window length in dt
    sta_wins = zeros((len(spiketrain), w_d))
    for i, st in enumerate(spiketrain):
        t_d = int(st/dt)
        if (w_d < t_d):
            w_start = t_d-w_d  # window start position index
            w_end = t_d  # window end index
            sta_wins[i, :] = v[w_start:w_end]
        else:
            '''
            We have two options here:
            (a) drop the spike
            (b) pad with zeroes
            --
            (a) would make the size of the `wins` matrix inconsistent with
            the number of spikes and one would expect the rows to match
            (b) can skew the average and variance calculations
            --
            Currently going with (b): if the number of spikes is small enough
            that one would skew the stats, I won't be looking at the stats
            anyway.
            '''
            w_start = 0
            w_end = t_d
            curwin = append(zeros(w_d-t_d), v[:t_d])
            sta_wins[i, :] = curwin
    sta_avg = mean(sta_wins, 0)
    sta_std = std(sta_wins, 0)

    return sta_avg, sta_std, sta_wins


def sync_inp(n, rate, s, sigma, dura, dt=0.0001*second):
    '''
    Generates synchronous spike trains and returns spiketimes compatible
    with Brian's SpikeGeneratorGroup function.
    In other words, the array returned by this module should be passed as the
    argument to the MulitpleSpikeGeneratorGroup in order to define it as an
    input group.

    Parameters
    ----------
    n : int
        Number of spike trains

    rate : brian Hz (frequency)
        Spike rate for each spike train (units: freq)

    s : float
        Proportion of synchronous spike trains [0,1]

    sigma : brian second (time)
        Standard deviation of Gaussian random variable which is used to shift
        each spike in a synchronous spike train (units: time)

    dura : brian second (time)
        Duration of each spike train (units: time)

    dt : brian second (time)
        Simulation time step (units: time)

    Returns
    -------
    spiketimes : list of list
        Each item on the list is a spike train. Each spike train is
        a list of spike times.
    '''

    if not(0 <= s <= 1):
        warn("Synchrony should be between 0 and 1. Setting to 1.")
        s = 1

    n_ident = int(floor(n*s))   # number of identical spike trains
    spiketrains = []
    st_ident = poisson_spikes(dura, rate, dt)
    for i in range(n_ident):
        spiketrains.append(add_gauss_jitter(st_ident, sigma, dt))

    for i in range(n_ident, n):
        spiketrains.append(poisson_spikes(dura, rate, dt))

    return spiketrains


def poisson_spikes(dura, rate, dt=0.0001):
    '''
    Generates a single spike train with exponentially distributed inter-spike
    intervals, i.e., a realisation of a Poisson process.
    Returns a list of spike times.

    Parameters
    ----------
    dura : Duration of spike train (in seconds)

    rate : Spike rate (in Hz)

    dt : Simulation time step (in seconds)

    Returns
    -------
    spiketrain : list
        A spike train as a list of spike times
    '''

    spiketrain = []
    #   generate first interval
    while len(spiketrain) == 0:
        newinterval = random.exponential(1./rate)
        if newinterval < dt:
            newinterval = dt
        if newinterval < dura:
            spiketrain = [newinterval]
    #   generate intervals until we hit the duration
    while spiketrain[-1] < dura:
        newinterval = random.exponential(1./rate)
        if newinterval < dt:
            newinterval = dt
        spiketrain.append(spiketrain[-1]+newinterval)
    #   remove last spike overflow from while condition
    spiketrain = spiketrain[:-1]
    return spiketrain


def add_gauss_jitter(spiketrain, jitter, dt=0.0001*second):
    '''
    Adds jitter to each spike in the supplied spike train and returns the
    resulting spike train.
    Jitter is applied by adding a sample from a Gaussian random variable to
    each spike time.

    Parameters
    ----------
    spiketrain : list
        A spike train characterised as a list of spike times

    jitter : brian second (time)
        Standard deviation of Gaussian random variable which is added to each
        spike in a synchronous spike train (units: time)

    dt : brian second (time)
        Simulation time step (units: time)

    Returns
    -------
    jspiketrain : list
        A spike train characterised by a list of spike times
    '''

    if (jitter == 0*second):
        return spiketrain

    jspiketrain = spiketrain + random.normal(0, jitter, len(spiketrain))

    #   sort the spike train to account for ordering changes
    jspiketrain.sort()
    #   can cause intervals to become shorter than dt
    intervals = diff(jspiketrain)
    while min(intervals) < dt/second:
        index = where(intervals == min(intervals))[0][0]
        intervals[index] += dt/second
    jspiketrain = cumsum(intervals)
    jspiketrain = [st*second for st in jspiketrain]
    return jspiketrain


def times_to_bin(spikes, dt=0.001*second, duration=None):
    '''
    Converts a spike train into a binary string. Each bit is a bin of
    fixed width (dt).
    This function is useful for aligning a binary representation of a spike
    train to recordings of the respective membrane potential and for processing
    spike trains in binary format.

    Parameters
    ----------
    spiketimes : numpy array
        A spiketrain array from a brian SpikeMonitor

    dt : brian second (time)
        The width of each bin (default 1 ms)

    duration: brian second (time)
        The duration of the spike train. If `None`, the length of the spike
        train is determined by the last spike time. If a time is specified, the
        final spike train is either truncated (if duration < last_spike) or
        the spike train is padded with zeros.

    Returns
    -------
    bintimes : numpy array
        Array of 0s and 1s, respectively indicating the absence or presence
        of at least one spike in each bin. Information on potential multiple
        spikes in a bin is lost.
    '''

    if not len(spikes):
        # no spikes
        if duration is None:
            return spikes
        else:
            return zeros(int(duration/dt))
    st = divide(spikes, dt).astype(int)
    if duration is None:
        binlength = max(st)+1
    else:
        binlength = int(duration/dt)
    bintimes = zeros(binlength)
    if len(st) == 0:
        return bintimes
    if st[-1] > binlength:
        st = st[st < binlength]
    bintimes[st] = 1
    return bintimes


def times_to_bin_multi(spikes, dt=0.001*second, duration=None):
    if isinstance(spikes, dict):
        spiketimes = array([st for st in spikes.itervalues()])
    elif isinstance(spikes, (list, ndarray)):
        spiketimes = spikes
    else:
        raise TypeError('dictionary, list or array expected')
    if duration is None:
        # find the maximum value of all
        duration = max(recursiveflat(spiketimes))+float(dt)
    bintimes = array([times_to_bin(st, dt=dt, duration=duration)
                      for st in spiketimes])
    return bintimes


def PSTH(spikes, bin, dt, duration=None):
    '''
    Similar to times_to_bin{_multi} though it doesn't discard multiple spikes
    in a single bin. Allows plotting of the PSTH. Returns the times of the bins
    and the number of spikes in each bin (much like a histogram).

    Parameters
    ----------
    spikes : spike times (list or array)
    bin : the length of the bin (in seconds)
    dt : the temporal resolution of the spike times (e.g., simulation time step)
    duration : defaults to time of last spike

    NB: Entire function can be replaced by a simple call to histogram with an
    appropriate bin size.
    '''
    if bin < dt:
        bin = dt
    spiketimes = []
    if isinstance(spikes, dict):
        spiketimes = array([st for st in spikes.itervalues()])
    elif isinstance(spikes, list) or isinstance(spikes, array):
        spiketimes = array(spikes)
    else:
        raise TypeError('dictionary, list or array expected')
    flatspikes = recursiveflat(spiketimes)
    if duration is None:
        duration = max(flatspikes)+float(dt)
    flatspikes = array(flatspikes)
    nbins = int(duration/bin)
    psth = zeros(nbins)
    for b in arange(0, nbins, 1):
        binspikes = bitwise_and(flatspikes >= b*bin,
                                flatspikes < (b+1)*bin)
        psth[b] = count_nonzero(binspikes)
    return psth


def unitrange(start, stop, step):
    '''
    Returns a list in the same manner as the Python built-in range, but works
    with brian units.

    '''
    if not isinstance(start, units.Quantity):
        raise TypeError("unitrange: `start` argument is not a brian unit."
                        "Use Python build-in range() or numpy.arange()")
    if not isinstance(stop, units.Quantity):
        raise TypeError("unitrange: `stop` argument is not a brian unit."
                        "Use Python build-in range() or numpy.arange()")
    if not isinstance(step, units.Quantity):
        raise TypeError("unitrange: `step` argument is not a brian unit."
                        "Use Python build-in range() or numpy.arange()")
    if not start.has_same_dimensions(stop) \
            or not start.has_same_dimensions(step) \
            or not stop.has_same_dimensions(step):
        raise TypeError("Dimension mismatch in `unitrange`")

    x = start
    retlist = []
    while x < stop:
        retlist.append(x)
        x += step
    return retlist


def spike_period_hist(spiketimes, freq, duration, nbins=10, dt=0.0001*second):
    dt = float(dt)
    period = 1/freq  # in ms
    period = int(period/dt)  # in timesteps
    binwidth = period/nbins  # segment period into 10 bins
    bins = zeros(nbins)
    nper = int((duration/dt)/period)  # number of periods
    st_a = array(spiketimes)/dt
    for i in range(len(bins)):
        for p in range(nper):
            perstart = p*period
            inbin = st_a[(st_a >= perstart+i*binwidth) &
                         (st_a < perstart+(i+1)*binwidth-1)]
            bins[i] += len(inbin)
    left = arange(0, 1, 1./nbins)
    return left, bins


def recursiveflat(ndobject):
    """
    Recursive function that flattens a n-dimensional object such as an array
    or list. Must be accessible in the format ndobject[i].
    Returns a 1-d list.
    """
    if not len(ndobject):
        return ndobject
    elif shape(ndobject[0]) == ():  # TODO: Probably not completely correct
        return ndobject
    else:
        return recursiveflat([item for row in ndobject for item in row])


def spikeconvolve(spikes, sigma, dt=0.0001*second):
    """
    Given a SpikeMonitor, convolves the binned spike trains (binned at `dt`)
    with a Gaussian kernel of width `sigma`.  Instead of a SpikeMonitor
    object, the spike trains can also be provided as a list (or iterable) of
    arrays.  The limits and precision of the array for the convolution kernel
    is determined by the width `sigma` provided for the convolution.

    Parameters
    ----------
    spikes : brian.monitor.SpikeMonitor or any iterable of arrays containing
             spike times
    sigma : width of Gaussian convolution kernel
    dt : bin width for spike binning (default: 0.1 ms)

    Returns
    -------
    Two 1D arrays: the time of the start of each bin and the convolved spike
    train, such that plot(t, convspikes) displays the result of the convolution
    with proper horizontal axis scaling.
    """
    dt = float(dt)
    sigma = float(sigma)
    if isinstance(spikes, SpikeMonitor):
        allspikes = array([sp for train in spikes.spiketimes.itervalues()
                           for sp in train])
    elif isinstance(spikes, collections.Iterable):
        allspikes = array([sp for train in spikes for sp in train])
    binnedspikes = zeros(int(max(allspikes)/dt)+1)
    for spike in allspikes:
        bin_idx = int(spike/dt)
        binnedspikes[bin_idx]+=1
    x = linspace(-4*sigma, 4*sigma, 100)
    convkernel = normpdf(x, 0, sigma)
    convspikes = convolve(binnedspikes, convkernel, mode="same")
    t = arange(0, len(convspikes), 1)*dt
    return t, convspikes

