"""Use an automated method to pre-mark channels as bad.

Using the "pyprep" Python package, try to detect potentially bad channels
and save them. These can then be loaded when inspecting the raw data visually
to inform choices on whether or not particular channels are bad.

That is, the results from this step should be double checked and evaluated
by an expert in the "02_inspect_raw.py" script.

NOTE: This script takes about 10 minutes to run per participant on an
Intel® Core™ i7-8650U CPU @ 1.90GHz × 8 with 32GB RAM.

How to use the script?
----------------------
Either run in an interactive IPython session and have code cells rendered ("# %%")
by an editor such as VSCode, **or** run this from the command line, optionally
specifying settings as command line arguments:

```shell

python 01_find_bads.py --sub=1

```

"""
# %%
# Imports
import json
import pathlib
import sys

import click
import pyprep

from config import ANALYSIS_DIR_LOCAL, BAD_SUBJS, DATA_DIR_EXTERNAL
from utils import prepare_raw_from_source

# %%
# Settings
# Select the subject to work on here
sub = 1

# Select the data source and analysis directory here
data_dir = DATA_DIR_EXTERNAL
analysis_dir = ANALYSIS_DIR_LOCAL

# overwrite existing annotation data?
overwrite = False

# random number generator seed for the ICA
pyprep_rng = 42

# %%


# Potentially overwrite settings with command line arguments
@click.command()
@click.option("-s", "--sub", default=sub, type=int, help="Subject number")
@click.option("-d", "--data_dir", default=data_dir, type=str, help="Data location")
@click.option("-a", "--analysis_dir", default=data_dir, type=str, help="Analysis dir")
@click.option("-o", "--overwrite", default=overwrite, type=bool, help="Overwrite?")
@click.option("-r", "--pyprep_rng", default=pyprep_rng, type=int, help="PyPrep seed")
def get_inputs(sub, data_dir, analysis_dir, overwrite, pyprep_rng):
    """Parse inputs in case script is run from command line."""
    print("Overwriting settings from command line.\nUsing the following settings:")
    for name, opt in [
        ("sub", sub),
        ("data_dir", data_dir),
        ("analysis_dir", data_dir),
        ("overwrite", overwrite),
        ("pyprep_rng", pyprep_rng),
    ]:
        print(f"    > {name}: {opt}")

    data_dir = pathlib.Path(data_dir)
    analysis_dir = pathlib.Path(analysis_dir)
    return sub, data_dir, analysis_dir, overwrite, pyprep_rng


# only run this when not in an IPython session
# https://docs.python.org/3/library/sys.html#sys.ps1
if not hasattr(sys, "ps1"):
    sub, data_dir, analysis_dir, overwrite, pyprep_rng = get_inputs.main(
        standalone_mode=False
    )

# %%
# Prepare file paths
savedir = analysis_dir / "derived_data" / "annotations"
savedir.mkdir(parents=True, exist_ok=True)

fname_pyprep = savedir / f"sub-{sub:02}_bads_pyprep.json"

overwrite_msg = "\nfile exists and overwrite is False:\n\n>>> {}\n"

# %%
# Check overwrite
if fname_pyprep.exists() and not overwrite:
    raise RuntimeError(overwrite_msg.format(fname_pyprep))

# %%
# Prepare data
if sub in BAD_SUBJS:
    raise RuntimeError("No need to work on the bad subjs.")

raw = prepare_raw_from_source(sub, data_dir, analysis_dir)

# %%
# Run pyprep
nc = pyprep.NoisyChannels(raw, random_state=pyprep_rng)
nc.find_all_bads()
bads_dict = nc.get_bads(as_dict=True)

# %%
# Save the outputs
with open(fname_pyprep, "w") as fout:
    json.dump(bads_dict, fout, indent=4)

# %%