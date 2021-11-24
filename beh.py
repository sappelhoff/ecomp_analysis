"""Beh analysis."""
# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import DATA_DIR_LOCAL, get_sourcedata

# %%
# Check how often participants "timed out" on their choices
for sub in range(1, 33):
    for stream in ["single", "dual"]:
        _, tsv = get_sourcedata(sub, stream, DATA_DIR_LOCAL)
        df = pd.read_csv(tsv, sep="\t")
        validity_sum = df["validity"].to_numpy().sum()
        if validity_sum != df.shape[0]:
            print(sub, stream, f" - n timeouts: {df.shape[0]-validity_sum}")

# %%
# Read data

sub = 5
stream = "single"
_, tsv = get_sourcedata(sub, stream, DATA_DIR_LOCAL)

# %%
df = pd.read_csv(tsv, sep="\t")
df
# %%
# Calculate weightings
#
# single: evidence of each number towards "higher"
#
# Go through trials and count for each number:
# how often did it occur in trials where the final choice was "higher"
# how often did it occur overall
# the weight is the former divided by the latter:
# It is "1", if occurrence of a number *always* led to the choice "higher",
# and it is "0", if occurrence of a number *never* led to the choice "higher",
# and it is "0.5" if it evenly led to either choice "higher" or "lower".
# Note that practically, the extreme weights are statistically unlikely to ever
# be reached, because a sample vector or [9, 9, 9, 9, 9, 9, 9, 9, 9, 1] will
# reasonably be followed by a choice "higher", which will count the sample "1"
# as having a weight >0.5 towards "higher", as opposed to the sensible weight
# being <0.5 (i.e., towards "lower").


n_number_higher_dict = {i: [0] for i in range(1, 10)}
n_number_overall_dict = {i: [0] for i in range(1, 10)}
for trial in np.unique(df["trial"]):

    valid = df.loc[df["trial"] == trial, "validity"][trial]
    choice = df.loc[df["trial"] == trial, "choice"][trial]
    samples = (
        df[df["trial"] == trial][[f"sample{i}" for i in range(1, 11)]]
        .to_numpy()
        .squeeze()
    )

    for number in samples:
        if choice == "higher":
            n_number_higher_dict[np.abs(number)][0] += 1
        n_number_overall_dict[np.abs(number)][0] += 1

tmp = pd.concat(
    [
        pd.DataFrame.from_dict(n_number_higher_dict),
        pd.DataFrame.from_dict(n_number_overall_dict),
    ]
).T
tmp = tmp.reset_index()
tmp.columns = ["number", "nhigher", "noverall"]
tmp["weight"] = tmp["nhigher"] / tmp["noverall"]

plt.plot(tmp["number"], tmp["weight"])

# %%
# verena approach "single"

numbers = df.loc[:, "sample1":"sample10"].to_numpy().reshape(-1)
numbers = np.abs(numbers)
finchoice = np.repeat(df["choice"].map({"lower": 0, "higher": 1}).to_numpy(), 10)
assert finchoice.shape == numbers.shape

weights = np.zeros(9)
for i, number in enumerate(range(1, 10)):

    weights[i] = np.mean(finchoice[numbers == number])
    print(
        number, finchoice[numbers == number].sum(), finchoice[numbers == number].shape
    )

plt.plot(np.arange(1, 10), weights)

# %%

# verena approach for "dual"
#
# finchoice vector: coded 1 for red, 0 for blue final choice
# repeat each element for number of samples
# for each red number, select same index from vector
# for each blue number, select same index from vector, but flip (blue=1, red=0)
# concatenate these two vectors and take the mean --> corresponds to:
# sum(red number and red choice, blue number and blue choice) / all numbers

numbers = df.loc[:, "sample1":"sample10"].to_numpy().reshape(-1)
catcols = (np.sign(numbers) + 1) / 2  # -=0, +=1
numbers = np.abs(numbers)
finchoice = np.repeat(df["choice"].map({"lower": 0, "higher": 1}).to_numpy(), 10)
assert finchoice.shape == numbers.shape

weights = np.zeros(9)
for i, number in enumerate(range(1, 10)):

    x = finchoice[np.logical_and(numbers == number, catcols == 1)]
    y = -finchoice[np.logical_and(numbers == number, catcols == 0)] + 1

    weights[i] = np.mean(np.hstack([x, y]))

plt.plot(np.arange(1, 10), weights)
# %%

# compute analogously to "SP" analysis:
# For each number:
# (
# number was shown in red and red was selected
# PLUS
# number was shown in blue and blue was selected
# )
# DIVIDED BY
# all times the number was shown

n_number_higher_dict = {i: [0] for i in range(1, 10)}
n_number_overall_dict = {i: [0] for i in range(1, 10)}
for trial in np.unique(df["trial"]):

    valid = df.loc[df["trial"] == trial, "validity"][trial]
    choice = df.loc[df["trial"] == trial, "choice"][trial]
    samples = (
        df[df["trial"] == trial][[f"sample{i}" for i in range(1, 11)]]
        .to_numpy()
        .squeeze()
    )

    for number in samples:
        if (choice == "higher" and number > 0) or (choice == "lower" and number < 0):
            n_number_higher_dict[np.abs(number)][0] += 1

        n_number_overall_dict[np.abs(number)][0] += 1

tmp = pd.concat(
    [
        pd.DataFrame.from_dict(n_number_higher_dict),
        pd.DataFrame.from_dict(n_number_overall_dict),
    ]
).T
tmp = tmp.reset_index()
tmp.columns = ["number", "nhigher", "noverall"]
tmp["weight"] = tmp["nhigher"] / tmp["noverall"]

plt.plot(tmp["number"], tmp["weight"])
# %%


def calc_nonp_weights(df):
    """Calculate non-parametric weights.

    Parameters
    ----------
    df : pandas.DataFrame
        The behavioral data of a subject.

    Returns
    -------
    weights : np.ndarray, shape(9,)
        The weight for each of the 9 numbers in ascending
        order (1 to 9).
    position_weights : np.ndarray, shape(9, 10)
        The weight for each of the 9 numbers in ascending
        order, calculated for each of the 10 sample positions.

    """
    # work on a copy of the data
    weights_df = df.copy()

    stream = np.unique(weights_df["stream"])[0]

    # remove NaN rows
    nan_row_idxs = np.nonzero(~weights_df["validity"].to_numpy())[0]
    for idx in nan_row_idxs:
        assert pd.isna(weights_df.loc[idx, "direction"])
    weights_df = weights_df.drop(nan_row_idxs)

    # prepare data
    nsamples = 10
    isamples = [f"sample{i}" for i in range(1, nsamples + 1)]  # samples 1 to nsamples
    samples_signed = weights_df.loc[:, isamples].to_numpy().reshape(-1)
    samples = np.abs(samples_signed)
    colors = (np.sign(samples_signed) + 1) / 2  # color1=0, color2=1
    positions = np.tile(
        np.arange(len(isamples)), len(weights_df["trial"])
    )  # sample positions
    assert positions.shape == samples.shape

    if stream == "single":
        # sanity check
        np.testing.assert_array_equal(
            np.unique(weights_df["choice"]), ["higher", "lower"]
        )

        # map lower/higher to 0/1 ... and repeat choice for each sample
        choices = np.repeat(
            weights_df["choice"].map({"lower": 0, "higher": 1}).to_numpy(),
            len(isamples),
        )
        assert choices.shape == samples.shape
    else:
        assert stream == "dual"
        # sanity check
        np.testing.assert_array_equal(np.unique(weights_df["choice"]), ["red", "blue"])

        # map red/blue to 0/1 ... and repeat choice for each sample
        choices = np.repeat(
            weights_df["choice"].map({"red": 0, "blue": 1}).to_numpy(), len(isamples)
        )
        assert choices.shape == samples.shape

    # Calculate overall weights and for each sample position
    numbers = np.arange(1, 10, dtype=int)  # numbers 1 to 9 were shown
    weights = np.zeros(len(numbers))
    position_weights = np.zeros((len(numbers), len(isamples)))
    for inumber, number in enumerate(numbers):

        if stream == "single":

            # overall weights
            weights[inumber] = np.mean(choices[samples == number])

            # weights for each sample position
            for pos in np.unique(positions):
                position_weights[inumber, pos] = np.mean(
                    choices[(samples == number) & (positions == pos)]
                )

        else:
            assert stream == "dual"

            # overall weights
            weights[inumber] = np.mean(
                np.hstack(
                    [
                        choices[(samples == number) & (colors == 1)],
                        -choices[(samples == number) & (colors == 0)] + 1,
                    ]
                )
            )

            # weights for each sample position
            for pos in np.unique(positions):
                position_weights[inumber, pos] = np.mean(
                    np.hstack(
                        [
                            choices[
                                (samples == number) & (colors == 1) & (positions == pos)
                            ],
                            -choices[
                                (samples == number) & (colors == 0) & (positions == pos)
                            ]
                            + 1,
                        ]
                    )
                )

    return weights, position_weights


weights, position_weights = calc_nonp_weights(df)

# %%
for i in range(10):
    plt.plot(np.arange(1, 10), position_weights[:, i])
# %%
