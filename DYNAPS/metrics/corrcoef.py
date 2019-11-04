'''
Correlation coefficient between binned spike trains.
'''
import numpy as np
from tools import times_to_bin_multi


def corrcoef_spiketrains(spikes, b=0.001, duration=None):
    '''
    Calculates the mean correlation coefficient between a set of spike trains
    after being binned with bin width `b`.

    NB: Empty spike trains are discarded.
    '''
    # remove empty spike trains
    spikes = [sp for sp in spikes if len(sp)]
    bintimes = times_to_bin_multi(spikes, b, duration)
    correlations = np.corrcoef(bintimes)
    return correlations


def interval(inputspikes, outputspikes, b=0.001, duration=None):
    '''
    Calculates the mean pairwise correlation coefficient in intervals defined
    by a separate spike train. This function is used to calculate the
    correlation between *input* spike trains based on the interspike intervals
    of the *output* spike train. The result is therefore the distance between
    the input spikes that caused each response.

    Parameters
    ----------
    inputspikes : A set of spike trains whose pairwise distance will be
        calculated

    outputspikes : A single spike train to be used to calculate the
        intervals

    b : Bin width

    duration : Duration of the simulation or spike train
        (defaults to last input or output spike)

    '''
    b = float(b)
    if duration is None:
        duration = max(max(t) for t in inputspikes if len(t))
        duration = max((duration, max(outputspikes)))
    corrs = []
    for prv, nxt in zip(outputspikes[:-1], outputspikes[1:]):
        interval_inputs = []
        for insp in inputspikes:
            interval_spikes = insp[(prv < insp) & (insp <= nxt)]-prv
            if len(interval_spikes):
                interval_inputs.append(interval_spikes)
        corrs_i = np.mean(corrcoef_spiketrains(interval_inputs, b, duration))
        corrs.append(corrs_i)
    return corrs

