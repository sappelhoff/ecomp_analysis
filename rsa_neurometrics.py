"""Calculate RSA neurometrics.

- import subj/stream wise rdm_times arrays
- for each subj/stream, average over time window: 9x9 RDM
- create an array of numberline RDMs with different parameters each: kappa, bias
- for each subj/stream/meantime RDM, correlate with all model RDMs --> grid
- plot mean over grids for each stream
- plot individual grid maxima
- plot mean grid maximum

"""
