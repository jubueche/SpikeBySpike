"""
Modifications made by Achilleas Koutsou:

- Original function renamed to `distance` to comply with the other modules
in spikerlib.
- Added `pairwise`, `pairwise_mp` and `interval` functions. See individual
function docstrings for description.


######################## Original documentation follows #######################

An optimized algorithm for computing the modulus-metric distance do between
two spike trains T1 and T2.

For more information, please see the folowing papers:

Rusu, C. V. and Florian, R. V, A new class of metrics for spike trains, submitted.
Preprint: arXiv:1209.2918

Rusu, C. V. and Florian, R. V. (2010), A new spike train metric, BMC Neuroscience
11(Suppl. 1), P169. Nineteenth Annual Computational Neuroscience Meeting:
CNS*2010.
###############################################################################

Input:        T1, T2: two sorted, non-empty spike trains
              a, b: some bounds of the spike trains

Output:       the distance do

###############################################################################

 (C) R. V. Florian & C. V. Rusu, 2012

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 of the Software, and to permit persons to whom the Software is furnished to do
 so, subject to the following conditions:

 1. If you publish scientific papers based on work that uses the Software, you
 should consider citing within these papers the following:
 Rusu, C. V. and Florian, R. V. (2010), A new spike train metric,
 BMC Neuroscience 11(Suppl. 1), P169. Nineteenth Annual Computational Neuroscience Meeting:
 CNS*2010.
 2. If you create derivative works using the Sofware and these works have an associated
 list of contributors, you must attribute the work of R. V. Florian and C. V. Rusu according
 to the relevance of the Software to the derivative works.

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
 PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
 FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT
 OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 DEALINGS IN THE SOFTWARE.
 """


import numpy as np
import multiprocessing
import itertools


def distance(T1, T2, a, b):
    """
    T1 and T2 are ordered, nonempty sets of real numbers, indexed starting from 0.
    i1 and i2 are the indices of the currently processed spikes in the two spike
    trains. p1 and p2 are the indices of the previously processed spikes in the
    two spike trains. p is the index of the spike train to which the previously
    processed spike belonged (1 or 2), after at least one spike has been processed,
    or 0 otherwise.
    """

    def d(t,T,i):
        """
        Input:  a timing t, a sorted spike train T and an index i of a spike in T
                such that either t <= T[i] or i is the index of the last spike of T
        Output: the distance d(t, T) between a timing t and a spike train T
        """
        db = abs(T[i] - t)
        j = i - 1
        while j >= 0 and abs(T[j] - t) <= db:
            db = abs(T[j] - t)
            j -= 1
        return db

    i1 = 0; i2 = 0; p1 = 0; p2 = 0; p = 0
    n1 = len(T1); n2 = len(T2)

    #P is an array of structures (s,phi) consisting of a ordered pair of numbers.
    P = [(a, abs(T1[0] - T2[0]) )]

    #Process the spikes until the end of one of the spikes trains is reached.
    while i1 < n1 and i2 < n2:
        if T1[i1] <= T2[i2]:
            if i1 > 0:
                #Adds to P the timing situated at the middle of the interval between the
                #currently processed spike and the previous spike in the same spike train.
                t = (T1[i1] + T1[i1 - 1]) / 2.0
                #We have d(t, T1) = T1[i1] - t = t - T1[i1 - 1] = (T1[i1] - T1[i1 - 1]) / 2.
                P.append((t, abs((T1[i1] - T1[i1 - 1]) / 2.0 - d(t,T2,i2))))
            if p == 2:
                #If the previously processed spike was one from the other spike train than
                #the spike currently processed, adds to P the timing situated at the
                #middle of the interval between the currently processed spike and the
                #previously processed spike.
                t = (T1[i1] + T2[p2]) / 2.0
                #Since t is is at equal distance to the closest spikes in the two spike
                #trains, T1[i1] and T2[p2], we have d(t, T1)=d(t, T2) and phi(t)=0.
                P.append((t, 0))
            #Adds to P the currently processed spike.
            t = T1[i1]
            #We have d(t, T1) = 0. If at least one spike from T2 has been processed, we
            #have T2[p2] <= t <= T2[i2], with i2 = p2 + 1, and thus
            #d(t, T2) = min(|t-T2[p2]|, T2[i2]-t). If no spike from T2 has been processed,
            #we have p2 = i2 = 0, and the previous formula for d(t, T2) still holds.
            P.append((t, min(abs(t - T2[p2]), T2[i2] - t)))
            p1 = i1
            i1 += 1
            p = 1
        else:
            #Proceed analogously for the case T1[i1] > T2[i2]:
            if i2 > 0:
                t = (T2[i2] + T2[i2 - 1]) / 2.0
                P.append((t, abs((T2[i2] - T2[i2 - 1]) / 2.0 - d(t,T1,i1))))
            if p == 1:
                t = (T2[i2] + T1[p1]) / 2.0
                P.append((t, 0))
            t = T2[i2]
            P.append((t, min(abs(t - T1[p1]), T1[i1] - t)))
            p2 = i2
            i2 += 1
            p = 2

    #Process the rest of the spikes in the spike train that has not been fully
    #processed:
    while i1 < n1:
            if i1 > 0:
                #Adds to P the timing situated at the middle of the interval between the
                #currently processed spike and the previous spike in the same spike train
                t = (T1[i1] + T1[i1 - 1]) / 2.0
                #We have d(t, T1) = T1[i1] - t = t - T1[i1 - 1] = (T1[i1] - T1[i1 - 1]) / 2.
                P.append((t, abs((T1[i1] - T1[i1 - 1]) / 2.0 - d(t,T2,p2))))
            if p == 2:
                #If the previously processed spike was one from the other spike train than
                #the spike currently processed (i.e., the last spike in the spike train
                #that has been fully processed), adds to P the timing situated at the
                #middle of the interval between the currently processed spike and the
                #previously processed spike.
                t = (T1[i1] + T2[p2]) / 2.0
                #Since t is is at equal distance to the closest spikes in the two spike
                #trains, T1[i1] and T2[p2], we have d(t, T1)=d(t, T2) and phi(t)=0.
                P.append((t, 0))
            #Adds to P the currently processed spike.
            t = T1[i1]
            #We have d(t, T1) = 0. We have T2[p2] <= t and the spike at p2 ist the last one
            #in T2, and thus d(t, T2) = t - T2[p2].
            P.append((t, t - T2[p2]))
            #p1 = i1 #This could be added for completeness, but it is not used by this algorithm.
            i1 += 1
            p = 1
    while  i2 < n2:
            #Proceed analogously for the case that the train that has not been fully processed
            #is T2:
            if i2 > 0:
                t = (T2[i2] + T2[i2 - 1]) / 2.0
                P.append((t, abs((T2[i2] - T2[i2 - 1]) / 2.0 - d(t,T1,p1))))
            if p == 1:
                t = (T2[i2] + T1[p1]) / 2.0
                P.append((t, 0))
            t = T2[i2]
            P.append ((t, t - T1[p1]))
            #p2 = i2
            i2 += 1
            p = 2

    P.append ((b, abs(T1[n1 - 1] - T2[n2 - 1])))

    #sort P with regard to the first element
    P.sort()

    #perform the integration
    do = 0
    for i in range (1,len(P)):
        do += (P[i][0] - P[i-1][0]) * (P[i][1] + P[i-1][1]) / 2.0
    return do


def _all_dist_to_end(args):
    """
    Helper function for parallel pairwise distance calculations.
    """
    idx = args[0]
    spiketrains = args[1]
    start = args[2]
    end = args[3]
    num_spiketrains = len(spiketrains)
    distances = []
    for jdx in range(idx+1, num_spiketrains):
        dist = distance(spiketrains[idx], spiketrains[jdx], start, end)
        distances.append(dist)
    return distances


def pairwise_mp(spiketrains, start, end):
    """
    Calculates the average pairwise modulus distance between a set of
    spike trains.
    Uses Python's multiprocessing.Pool() to run each pairwise distance
    calculation in parallel.
    """
    # remove empty spike trains
    spiketrains = [sp for sp in spiketrains if len(sp)]
    count = len(spiketrains)
    distances = []
    idx_all = range(count - 1)
    pool = multiprocessing.Pool()
    distances_nested = pool.map(_all_dist_to_end,
                                zip(idx_all, itertools.repeat(spiketrains),
                                    itertools.repeat(start),
                                    itertools.repeat(end)))
    distances = []
    pool.close()
    pool.join()
    for dn in distances_nested:
        distances.extend(dn)
    return np.mean(distances)


def pairwise(spiketrains, start, end):
    """
    Calculates the average pairwise modulus distance between a set of
    spike trains.
    """
    # remove empty spike trains
    spiketrains = [sp for sp in spiketrains if len(sp)]
    count = len(spiketrains)
    distances = []
    for i in range(count - 1):
        for j in range(i+1, count):
            dist_ij = distance(spiketrains[i], spiketrains[j], start, end)
            distances.append(dist_ij)
    return np.mean(distances)


def interval(inputspikes, outputspikes, start, end, mp=True):
    """
    Calculates the mean pairwise modulus distance in intervals defined
    by a separate spike train. This function is used to calculate the distance
    between *input* spike trains based on the interspike intervals of the
    *output* spike train. The result is the distance between the input spikes
    that caused each response.

    inputspikes     A set of spike trains whose pairwise distances will
                    be calculated.

    outputspikes    A single spike train to be used to calculate the intervals.

    start           The start time of the spike trains.

    end             The end time of the spike trains.

    mp              Set to True to use the multiprocessing implementation of
                    the pairwise calculation function or False to use the
                    single process version (default: True)
    """
    modists = []
    pairwise_func = pairwise_mp if mp else pairwise
    for prv, nxt in zip(outputspikes[:-1], outputspikes[1:]):
        interval_inputs = []
        for insp in inputspikes:
            interval_inputs.append(insp[(prv < insp) & (insp <= nxt)])
        #mod = pairwise_func(interval_inputs, start, end)
        mod = pairwise_func(interval_inputs, prv, nxt)
        modists.append(mod)
    return modists



#Sample usage:
#a = 0; b = 20; T1 = [2, 5,10,11,20]; T2 = [3, 6, 14, 18]
#print str(modulus_metric(T1, T2, a, b))
