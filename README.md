[![DOI](https://zenodo.org/badge/423527863.svg)](https://zenodo.org/badge/latestdoi/423527863)

# ecomp_analysis

This repository contains the analysis code for the eComp project.

The archived version can be found on Zenodo:

- doi: [10.5281/zenodo.6411287](https://doi.org/10.5281/zenodo.6411287)

All important details are reported in the original paper for the project:

- preprint: [10.1101/2022.03.31.486560](https://doi.org/10.1101/2022.03.31.486560)
- journal article: [10.1371/journal.pcbi.1010747](https://doi.org/10.1371/journal.pcbi.1010747)

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
- doi: [10.12751/g-node.lir3qw](https://doi.org/10.12751/g-node.lir3qw)

However, running the code the first time might take a long time
in order to produce all derivatives that are not shipped as part of this
(analysis code) repository.
Downloading the `mpib_ecomp_derivatives` dataset may speed this step up
(not required):

- repository: https://gin.g-node.org/sappelhoff/mpib_ecomp_derivatives
- doi: [10.12751/g-node.9rtg6f](https://doi.org/10.12751/g-node.9rtg6f)

There is a Brain Imaging Data Structure (BIDS) version of the
data available:

- repository: https://gin.g-node.org/sappelhoff/mpib_ecomp_dataset
- doi: [10.12751/g-node.jtfg5d](https://doi.org/10.12751/g-node.jtfg5d)

Finally, there is the experiment code that was used when the data was collected:

- repository: https://github.com/sappelhoff/ecomp_experiment/
- doi: [10.5281/zenodo.6411313](https://doi.org/10.5281/zenodo.6411313)

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

Note that some of the other scripts do not accept command line options.
We recommend running them interactively with an editor such as VSCode
(https://code.visualstudio.com/).
VSCode will pick up on the `# %%` comments, and render parts of the script
as code cells that can be individually executed.

### General order to run code files

If you **don't** want to use the `mpib_ecomp_derivatives` dataset described above
to skip some of the computations, you will need to execute all code files.
The following list is a suggested order, because some code files will depend on others.
Note that the data in the `derived_data/annotations/` directory is crucial for
reproducing results because parts of the information therein cannot be automatically provided
(for example, which ICA components are rejected).

1. `beh_misc.py`
    1. two analyses will be skipped because they depend on outputs from later
       steps. Simply re-run this script after completing all other steps to
       get the respective results.
1. `00_sanity_check.py`
1. `01_find_bads.py`
1. `02_inspect_raw.py`
1. `03_run_ica.py`
1. `04_inspect_ica.py`
1. `05_make_epochs.py`
1. `eeg_misc.py`
1. `erp_numbers.py`
1. `rdms_mahalanobis.py`
1. `rsa_timecourse.py`
1. `rsa_neurometrics.py`
1. `beh_accuracy.py`
1. `beh_modeling.py`
    1. Run with different command line options: `--fit_scenario=X`, where
       `X` is (i) `"free"`, (ii) `"k_is_1"`, (iii) `"k_smaller_1"`, and
       (iv) `"k_bigger_1"`. Or set the `fit_scenario` variable in the script
       if you do not want to run this from the command line.
       (Keep `fit_position` set to `"all"`)
    1. Run with different command line options `fit_position=X`, where
       `X` is a string "1", "2", ..., "10" (So you need to run this
       ten times). Additionally, run it with `firsthalf` and `secondhalf`.
       As above, you may also simply set the `fit_position` variable
       in the script. (Keep `fit_scenario` set to `"free"`)
1. `beh_weightings.py`
    1. Run with different command line options: `--fit_scenario=X`, where
       `X` is (i) `free`, (ii) `k_is_1`. Or set the `fit_scenario` variable
       in the script if you do not want to run this from the command line.
1. `plots_fig1.py`
1. `plots_fig2.py`
1. `plots_fig3.py`
1. `plots_fig4.py`
1. Use LibreOffice Draw (https://www.libreoffice.org/discover/draw/) to open `figures/paradigm_figure.odg`
   and export `paradigm_figure.pdf`
1. Use `xelatex` (for example on a service like https://www.overleaf.com/) and `figures/fig*.tex` to finalize the figures.
   Note that the `*` stands for 1, 2, 3, or 4.

## Explanation of files

### General "infrastructure" files

- `.flake8` --> code style configuration (see also `pyproject.toml`)
- `.gitignore` --> which files to ignore during version control
- `.pre-commit-config.yaml` --> for enforcing proper coding style
- `.environment.yml` --> software dependencies, see "Installation"
- `pyproject.toml` --> code style configuration (see also `.flake8`)
- `README.md` --> general information about this repository
- `LICENSE` --> how to use and credit the resources

### Utility libraries

- `config.py` --> general configuration and variables that get re-used across scripts
- `utils.py` --> general functions that get re-used across scripts
- `clusterperm.py` --> functions for cluster-based permutation testing
- `model_rdms.py` --> script to generate model representational dissimilarity matrices

### EEG preprocessing scripts

Generel preprocessing flow of the EEG data:

- `00_sanity_check.py`
- `01_find_bads.py`
- `02_inspect_raw.py`
- `03_run_ica.py`
- `04_inspect_ica.py`
- `05_make_epochs.py`

### Miscellaneous analysis scripts

For example for checking for missing data or rejected EEG epochs:

- `beh_misc.py`
- `eeg_misc.py`

### EEG analysis scripts

For both RSA and ERP analyses:

- `rdms_mahalanobis.py`
- `rsa_neurometrics.py`
- `rsa_timecourse.py`
- `erp_numbers.py`

### Behavior analysis scripts

For general accuracy and descriptive behavior analyses, as well as modeling:

- `beh_accuracy.py`
- `beh_modeling.py`
- `beh_weightings.py`

### Figure creation scripts

Each producing the figure as found in the manuscript (but see `.tex` files mentioned below):

- `plots_fig1.py`
- `plots_fig2.py`
- `plots_fig3.py`
- `plots_fig4.py`

### Analysis outputs (`derived_data/` directory)

This directory generally contains outputs from the scripts above.
Importantly, the `annotations` subfolder contains important data from the
visual screening of the EEG data (cannot be automatically reproduced):

- `*_annotations.txt` --> saved MNE-Python annotations
- `*_bad-channels.txt` --> each non-empty row in the file describes a bad channel
- `*_bads_pyprep.json` --> outputs from the `01_find_bads.py` script
- `*_exclude_ica.json` --> which ICA components to exclude
- `*_faster_bad_epos.json` --> epoch indices rejected through the "FASTER" pipeline

### Analysis outputs (`figures/` directory)

This directory generally contains outputs from the scripts above.
Exceptions are:

- `paradigm_figure.odg` --> to create fig1a, using LibreOffice Draw
  (https://www.libreoffice.org/discover/draw/)
- `*.tex` --> files that are used to put panel letters onto finished figures:
    - `fig1.tex`
    - `fig2.tex`
    - `fig3.tex`
    - `fig4.tex`
