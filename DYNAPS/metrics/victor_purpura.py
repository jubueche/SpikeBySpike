"""
Victor-Purpura spike time distance metric
Victor and Purpura, 1996, Journal of Neurophysiology
"""
import numpy as np
import multiprocessing
import itertools


def distance(st_one, st_two, cost):
    """
    Calculates the "spike time" distance (Victor & Purpura, 1996) for a single
    cost.

    Parameters
    ----------
    tli : vector of spike times for first spike train
    tlj : nvector of spike times for second spike train
    cost : cost per unit time to move a spike

    Ported to Python by Achilleas Koutsou from Matlab code by Daniel Reich.
    """
    len_one = len(st_one)
    len_two = len(st_two)
    if cost == 0:
        dist = np.abs(len_one-len_two)
        return dist
    elif cost == float('inf'):
        dist = len_one+len_two
        return dist
    scr = np.zeros((len_one+1, len_two+1))
    scr[:,0] = np.arange(len_one+1)
    scr[0,:] = np.arange(len_two+1)
    if len_one and len_two:
        for i in range(1, len_one+1):
            for j in range(1, len_two+1):
                scr[i,j]=np.min((scr[i-1,j]+1,
                                scr[i,j-1]+1,
                                scr[i-1,j-1]+cost*np.abs(st_one[i-1]-st_two[j-1]))
                               )
    return scr[-1,-1]


def pairwise_mp(spiketrains, cost):
    """
    Calculates the average pairwise distance between a set of spike trains.
    Uses Python's multiprocessing.Pool() to run each pairwise distance
    calculation in parallel.
    """
    # remove empty spike trains
    spiketrains = [sp for sp in spiketrains if len(sp)]
    count = len(spiketrains)
    idx_all = range(count - 1)
    pool = multiprocessing.Pool()
    distances_nested = pool.map(_all_dist_to_end,
                                zip(idx_all, itertools.repeat(spiketrains),
                                    itertools.repeat(cost)))
    distances = []
    pool.close()
    pool.join()
    for dn in distances_nested:
        distances.extend(dn)
    return np.mean(distances)


def pairwise(spiketrains, cost):
    """
    Calculates the average pairwise distance between a set of spike trains.
    """
    # remove empty spike trains
    spiketrains = [sp for sp in spiketrains if len(sp)]
    count = len(spiketrains)
    distances = []
    for i in range(count - 1):
        for j in range(i + 1, count):
            dist = distance(spiketrains[i], spiketrains[j], cost)
            distances.append(dist)
    return np.mean(distances)


def _all_dist_to_end(args):
    """
    Helper function for parallel pairwise distance calculations.
    """
    idx = args[0]
    spiketrains = args[1]
    cost = args[2]
    num_spiketrains = len(spiketrains)
    distances = []
    for jdx in range(idx + 1, num_spiketrains):
        dist = distance(spiketrains[idx], spiketrains[jdx], cost)
        distances.append(dist)
    return distances


def interval(inputspikes, outputspikes, cost, mp=True):
    """
    Calculates the mean pairwise spike time distance in intervals defined
    by a separate spike train. This function is used to calculate the distance
    between *input* spike trains based on the interspike intervals of the
    *output* spike train. The result is the distance between the input
    spikes that caused each response.

    Parameters
    ----------
    inputspikes : A set of spike trains whose pairwise distance will be
        calculated

    outputspikes : A single spike train to be used to calculate the
        intervals

    cost : The cost of moving a spike

    mp : Set to True to use the multiprocessing implementation
        of the pairwise calculation function or False to use the
        single process version (default: True)

    """
    vpdists = []
    pairwise_func = pairwise_mp if mp else pairwise
    for prv, nxt in zip(outputspikes[:-1], outputspikes[1:]):
        interval_inputs = []
        for insp in inputspikes:
            insp = np.array(insp)
            interval_inputs.append(insp[(prv < insp) & (insp <= nxt)])
        vpd = pairwise_func(interval_inputs, cost)
        vpdists.append(vpd)
    return vpdists

