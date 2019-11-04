"""
Modifications made by Achilleas Koutsou:

- Original functions renamed to `distance` and `pairwise` to comply with the
other modules in spikerlib.

- Added `pairwise_mp` and `interval` functions. See individual
function docstrings for description.


Kreuz SPIKE-distance
Kreuz, Chicharro, Greschner, Andrzejak, 2011, Journal of Neuroscience Methods.
and
Kreuz T, Chicharro D, Houghton C, Andrzejak RG, Mormann F, 2013,
Journal of Neurophysiology.


######################## Original documentation follows #######################

Comment by Thomas Kreuz:

This Python code (including all further comments) was written by Jeremy Fix (see http://jeremy.fix.free.fr/),
based on Matlab code written by Thomas Kreuz.

The SPIKE-distance is described in this paper:

Kreuz T, Chicharro D, Houghton C, Andrzejak RG, Mormann F:
Monitoring spike train synchrony.
J Neurophysiol 109, 1457-1472 (2013).

The Matlab codes as well as more information can be found at http://www.fi.isc.cnr.it/users/thomas.kreuz/sourcecode.html.

"""
from __future__ import division
import numpy as np
import multiprocessing
import itertools


def _find_corner_spikes(t, train, ibegin, start, end):
    """
    Return the times (t1,t2) of the spikes in train[ibegin:]
    such that t1 < t and t2 >= t
    """
    if(ibegin == 0):
        tprev = start
    else:
        tprev = train[ibegin-1]
    for idts, ts in enumerate(train[ibegin:]):
        if(ts >= t):
            return np.array([tprev, ts]), idts+ibegin
        tprev = ts
    return np.array([train[-1],end]), idts+ibegin


def distance(st_one, st_two, start, end, nsamples):
    """

    Computes the bivariate SPIKE distance of Kreuz et al. (2012) t1 and t2 are
    1D arrays with the spiking times of two neurones It returns the array of
    the values of the distance between time ti and te with N samples.  The
    arrays t1, t2 and values ti, te are unit less

    """
    t = np.linspace(start+(end-start)/nsamples, end, nsamples)
    st_one = np.insert(st_one, 0, start)
    st_one = np.append(st_one, end)
    st_two = np.insert(st_two, 0, start)
    st_two = np.append(st_two, end)

    # We compute the corner spikes for all the time instants we consider
    # corner_spikes is a 4 column matrix [t, tp1, tf1, tp2, tf2]
    corner_spikes = np.zeros((nsamples,5))

    ibegin_one = 0
    ibegin_two = 0
    corner_spikes[:,0] = t
    for itc, tc in enumerate(t):
       corner_spikes[itc,1:3], ibegin_t1 = _find_corner_spikes(tc, st_one,
                                                              ibegin_one,
                                                              start, end)
       corner_spikes[itc,3:5], ibegin_t2 = _find_corner_spikes(tc, st_two,
                                                              ibegin_two,
                                                              start, end)

    #print corner_spikes
    xisi = np.zeros((nsamples,2))
    xisi[:,0] = corner_spikes[:,2] - corner_spikes[:,1]
    xisi[:,1] = corner_spikes[:,4] - corner_spikes[:,3]
    norm_xisi = np.sum(xisi,axis=1)**2.0

    # We now compute the smallest distance between the spikes in st_two
    # and the corner spikes of st_one
    # with np.tile(st_two,(N,1)) we build a matrix :
    # np.tile(st_two,(N,1)) =   [st_two st_two st_two]' -
    #                       np.tile(reshape(corner_spikes,(N,1)), st_two.size) =
    #                       [corner corner corner]'

    dp1 = np.min(np.fabs(np.tile(st_two,(nsamples,1))
                         - np.tile(np.reshape(corner_spikes[:,1],(nsamples,1)),
                                   st_two.size)),
                 axis=1)
    df1 = np.min(np.fabs(np.tile(st_two,(nsamples,1))
                         - np.tile(np.reshape(corner_spikes[:,2],(nsamples,1)),
                                   st_two.size)),
                 axis=1)
    # And the smallest distance between the spikes in st_one and the corner spikes of st_two
    dp2 = np.min(np.fabs(np.tile(st_one,(nsamples,1))
                         - np.tile(np.reshape(corner_spikes[:,3],
                                              (nsamples,1)),st_one.size)),
                 axis=1)
    df2 = np.min(np.fabs(np.tile(st_one,(nsamples,1))
                         - np.tile(np.reshape(corner_spikes[:,4],(nsamples,1)),
                                   st_one.size)),
                 axis=1)

    xp1 = t - corner_spikes[:,1]
    xf1 = corner_spikes[:,2] - t
    xp2 = t - corner_spikes[:,3]
    xf2 = corner_spikes[:,4] - t

    S1 = (dp1 * xf1 + df1 * xp1)/xisi[:,0]
    S2 = (dp2 * xf2 + df2 * xp2)/xisi[:,1]

    inst_dist = (S1 * xisi[:,1] + S2 * xisi[:,0]) / (norm_xisi/2.0)

    return t, inst_dist

def _find_prev_spikes(t, spiketrains):
    prv = []
    for st in spiketrains:
        prv.append(max(st[st < t]))
    return prv

def _find_next_spikes(t, spiketrains):
    nxt = []
    for st in spiketrains:
        nxt.append(min(st[st >= t]))
    return nxt

def multivariate(spiketrains, start, end, nsamples):
    """
    Multivariate version as described in Kreuz et al., 2011.

    Parameters
    ==========
    spiketrains : is an array of spike time arrays

    start : the initial time of the recordings

    end : the end time of the recordings

    nsamples : the number of samples used to compute the distance
    """
    t = np.linspace(start+(end-start)/nsamples, end, nsamples)
    N = len(spiketrains)

    strains_se = []
    for idx in range(N):
        newst = np.insert(spiketrains[idx], 0, start)
        strains_se.append(np.append(newst, end))

    # different between t and previous and next spikes for each t
    prev_spikes = np.zeros((nsamples, N))
    next_spikes = np.zeros((nsamples, N))
    dprev_spikes = np.zeros((nsamples, N))
    dnext_spikes = np.zeros((nsamples, N))
    for idx, ti in enumerate(t):
        prev_spikes[idx] = _find_prev_spikes(ti, strains_se)
        next_spikes[idx] = _find_next_spikes(ti, strains_se)
        dprev_spikes[idx] = ti-prev_spikes[idx]
        dnext_spikes[idx] = next_spikes[idx]-ti

    # mean interval from t to previous/next spike on each spiketrain (over t)
    meanp = np.mean(dprev_spikes, axis=1)
    meanf = np.mean(dnext_spikes, axis=1)
    # stdev interval from t to previous/next spike on each spiketrain (over t)
    sigmap = np.std(prev_spikes, axis=1)
    sigmaf = np.std(next_spikes, axis=1)
    # mean inter-spike interval around each t
    xisi = meanp+meanf

    mvdist = ((sigmap*meanf)+(sigmaf*meanp))/(xisi**2)
    return t, mvdist

def pairwise(spiketrains, start, end, nsamples):
    """
    Calculate the instantaneous average over all the pairwise distances

    Parameters
    ==========
    spiketrains : is an array of spike time arrays

    start : the initial time of the recordings

    end : the end time of the recordings

    nsamples : the number of samples used to compute the distance

    spiketrains is a list of arrays of shape (N, T) with N spike trains
    """
    # remove empty spike trains
    spiketrains = [sp for sp in spiketrains if len(sp)]
    d = np.zeros((nsamples,))
    n_trains = len(spiketrains)
    t = 0
    for i, t1 in enumerate(spiketrains[:-1]):
        for t2 in spiketrains[i+1:]:
            tij, dij = distance(t1, t2, start, end, nsamples)
            if(i == 0):
                t = tij  # The times are only dependent on ti, te, and N
            d = d + dij
    d = d / float(n_trains * (n_trains-1) /2)
    return t,d


def pairwise_mp(spiketrains, start, end, N):
    """
    Calculates the multivariate distance (the instantaneous average over all
    the pairwise distances) using Python's multiprocessing.Pool() to
    run sets of calculations in parallel.

    Arguments have the same meaning as for `pairwise`.

    NB: This function has a slight (on the order 1E-16) deviation from the
        single-process version of the function `pairwise`. The cause of
        the difference has yet to be determined.
    """
    # remove empty spike trains
    spiketrains = [sp for sp in spiketrains if len(sp)]
    count = len(spiketrains)
    idx_all = range(count-1)
    pool = multiprocessing.Pool()
    pool_results = pool.map(_all_dist_to_end,
                            zip(idx_all, itertools.repeat(spiketrains),
                                itertools.repeat(start),
                                itertools.repeat(end),
                                itertools.repeat(N)))
    pool.close()
    pool.join()
    # Each pool calculated a different number of distance pairs
    # due to the nature of the `_all_dist_to_end` function.
    # We need to organise them into a proper 2D array.
    times = pool_results[0][0]
    distances = []
    for pr in pool_results:
        distances.extend(pr[1])
    return times, np.mean(distances, axis=0)


def _all_dist_to_end(args):
    """
    Helper function for parallel pairwise distance calculations.
    """
    idx = args[0]
    spiketrains = args[1]
    start = args[2]
    end = args[3]
    N = args[4]
    num_spiketrains = len(spiketrains)
    distances = []
    for jdx in range(idx+1, num_spiketrains):
        dist = distance(spiketrains[idx], spiketrains[jdx], start, end, N)
        times = dist[0]  # should be the same for all
        distances.append(dist[1])
    return times, distances


def pairwise_interval(inputspikes, outputspikes, samples=1, mp=True):
    """
    Calculates the mean pairwise SPIKE-distance in intervals defined
    by a separate spike train. This function is used to calculate the distance
    between *input* spike trains based on the interspike intervals of the
    *output* spike train. The result is therefore the distance between the
    input spikes that caused each response.

    NB: This is NOT the ISI distance. It is the spike distance calculated in
    successive intervals.

    Parameters
    ==========
    inputspikes : A list or array of spike trains whose pairwise distance will
        be calculated

    outputspikes : A single spike train to be used to calculate the
        intervals

    samples : The number of samples to use to for each interval

    mp : Set to True to use the multiprocessing implementation
        of the pairwise calculation function or False to use the
        single process version (default: True)

    """
    times = []
    krdists = []
    pairwise_func = pairwise_mp if mp else pairwise
    for prv, nxt in zip(outputspikes[:-1], outputspikes[1:]):
        krd = pairwise_func(inputspikes, prv, nxt, samples)
        times.append(krd[0])
        krdists.append(krd[1])
    return times, krdists


def interval_multivariate(inputspikes, outputspikes, samples=1):
    """
    Calculates the multivariate SPIKE-distance in intervals defined
    by a separate spike train. This function is used to calculate the distance
    between *input* spike trains based on the interspike intervals of the
    *output* spike train. The result is therefore the distance between the
    input spikes that caused each response.

    NB: This is NOT the ISI distance. It is the spike distance calculated in
    successive intervals.

    Parameters
    ==========
    inputspikes : A list or array of spike trains whose pairwise distance will
        be calculated

    outputspikes : A single spike train to be used to calculate the
        intervals

    samples : The number of samples to use to for each interval

    """
    times = []
    krdists = []
    for prv, nxt in zip(outputspikes[:-1], outputspikes[1:]):
        krd = multivariate(inputspikes, prv, nxt, samples)
        times.append(krd[0])
        krdists.append(krd[1])
    return times, krdists
