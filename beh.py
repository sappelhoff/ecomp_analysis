"""Beh analysis."""
# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import DATA_DIR_LOCAL, get_sourcedata

# %%
# Check how often participants "timed out" on their choices
for sub in range(1, 31):
    for stream in ["single", "dual"]:
        _, tsv = get_sourcedata(sub, stream, DATA_DIR_LOCAL)
        df = pd.read_csv(tsv, sep="\t")
        validity_sum = df["validity"].to_numpy().sum()
        if validity_sum != df.shape[0]:
            print(sub, stream, f" - n timeouts: {df.shape[0]-validity_sum}")

# %%
# Read data

sub = 1
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
