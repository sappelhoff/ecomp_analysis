# ecomp_analysis

This repository contains the analysis code for the eComp project.

The archived version can be found on Zenodo:

- doi: forthcoming

All important details are reported in the original paper for the project:

- preprint: forthcoming
- journal article: forthcoming

## Installation

To install a Python environment suitable to running this analysis code, please:

1. Download Miniconda for your system: https://docs.conda.io/en/latest/miniconda.html
   (this will provide you with the `conda` command)
1. Use `conda` to install `mamba`: `conda install mamba -n base -c conda-forge`
   (for more information, see: https://github.com/mamba-org/mamba)
1. Use the `environment.yml` file in this repository to create the `ecomp_analysis` environment:
   `mamba env create -f environment.yml`

You can now run `conda activate ecomp_analysis` and should have all required packages installed
for running the analyses.

## Obtaining the data

The code can run with the `mpib_ecomp_sourcedata` dataset:

- repository: https://gin.g-node.org/sappelhoff/mpib_ecomp_sourcedata
- doi: forthcoming

However, running the code the first time might take a long time
in order to produce all derivatives that are not shipped as part of this
(analysis code) repository.
Downloading the `mpib_ecomp_derivatives` dataset may speed this step up
(not required):

- repository: https://gin.g-node.org/sappelhoff/mpib_ecomp_derivatives
- doi: forthcoming

There is a Brain Imaging Data Structure (BIDS) version of the
data available:

- repository: https://gin.g-node.org/sappelhoff/mpib_ecomp_dataset
- doi: forthcoming

Finally, there is the experiment code that was used when the data was collected:

- repository: https://github.com/sappelhoff/ecomp_experiment/
- doi: forthcoming

## Configuration before running the code

Before running the code on your machine, make sure you have activated the `ecomp_analysis`
environment, and have downloaded all data (see steps above).

Next, you need to open the `config.py` file in this repository.
In there, you will find several important settings that can be kept stable.
However, you will need to adjust the path variables pointing to the data,
for example:

- `DATA_DIR_LOCAL`, pointing to the data stored locally
- `DATA_DIR_EXTERNAL`, pointing to the data stored on an external harddrive
- `ANALYSIS_DIR_LOCAL`, pointing to the analysis code repository (THIS repository) stored locally
- ...

Generally, the analyses scripts will import the paths defined in `config.py` and
use them to obtain the data.
For example:

```Python
from config import DATA_DIR_EXTERNAL, ANALYSIS_DIR_LOCAL

data_dir = DATA_DIR_EXTERNAL
analysis_dir = ANALYSIS_DIR_LOCAL

# ... use data_dir and analysis_dir
```

you can edit the analysis scripts themselves to get the data from local, external,
or remote sources that you have specified in `config.py`.

## Running the code

Some of the analysis scripts
(specifically those starting with two digits, e.g., `01_find_bads.py`)
accept command line options.
It is convenient to run them from the command line (e.g., with `bash`) like so:

```bash
for i in {1..32}
do
    python -u 01_find_bads.py \
        --sub=$i \
        --data_dir="/mnt/home/appelhoff/Projects/ARC-Studies/eeg_compression/ecomp_data/" \
        --analysis_dir="/mnt/home/appelhoff/Projects/ARC-Studies/eeg_compression/ecomp_data/code/ecomp_analysis/" \
        --overwrite=True
done
```

The above will run the `01_find_bads.py` script for each participant,
using the `overwrite` option,
and the data and analysis directory paths specified.

Note that many of the other scripts do not accept command line options.
We recommend running them interactively with an editor such as VSCode
(https://code.visualstudio.com/).
VSCode will pick up on the `# %%` comments, and render parts of the script
as code cells that can be individually executed.

## Explanation of files

### General "infrastructure" files

- `.flake8`
- `.gitignore`
- `.pre-commit-config.yaml`
- `.environment.yml`
- `pyproject.toml`
- `README.md`
- `LICENSE`

### Utility libraries

- `config.py`
- `utils.py`
- `clusterperm.py`
- `model_rdms.py`

### EEG preprocessing scripts

- `00_sanity_check.py`
- `01_find_bads.py`
- `02_inspect_raw.py`
- `03_run_ica.py`
- `04_inspect_ica.py`
- `05_make_epochs.py`

### Miscellaneous analysis scripts

- `beh_misc.py`
- `eeg_misc.py`

### EEG analysis scripts

- `rdms_mahalanobis.py`
- `rsa_neurometrics.py`
- `rsa_timecourse.py`
- `erp_numbers.py`

### Behavior analysis scripts

- `beh_accuracy.py`
- `beh_modeling.py`
- `beh_weightings.py`

### Figure creation scripts

- `plots_fig1.py`
- `plots_fig2.py`
- `plots_fig3.py`
- `plots_fig4.py`

### Analysis outputs (derivatives)

Generally contains outputs from the scripts above.
Importantly, the `annotations` subfolder contains important data from the
visual screening of the EEG data (cannot be automatically reproduced):

- `*_annotations.txt` saved MNE-Python annotations
- `*_bad-channels.txt` each non-empty row in the file describes a bad channel
- `*_bads_pyprep.json` outputs from the `01_find_bads.py` script
- `*_exclude_ica.json` which ICA components to exclude
- `*_faster_bad_epos.json` epoch indices rejected through the "FASTER" pipeline

### Analysis outputs (figures)

Generally contains outputs from the scripts above.
Exceptions are:

- `paradigm_figure.odg`, to create fig1a, using LibreOffice Draw
  (https://www.libreoffice.org/discover/draw/)
- `*.tex` files that are used to put panel letters onto finished figures:
    - `fig1.tex`
    - `fig2.tex`
    - `fig3.tex`
    - `fig4.tex`
