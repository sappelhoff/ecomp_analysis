"""Define and provide model RDMs.

NOTE: The order in MODELS determines the order of orthogonalization.
"""

import copy

import numpy as np
from scipy.spatial.distance import squareform

from config import NUMBERS
from utils import calc_rdm, eq1, spm_orth

MODELS = (
    "digit",
    "color",
    "parity",
    "numberline",
    "numXcat",
    "extremity",
)


def get_models_dict(rdm_size, modelnames, bias=None, kappa=None):
    """Get a dict of models and their orthogonalized versions.

    Parameters
    ----------
    rdm_size : {"9x9", "18x18"}
        The size of the RDMs.
    modelnames : list of str
        The names of the models to supply.
    bias, kappa : float | None
        The bias and kappa parameters, ignored if None. If specified,
        both must be specified, and numbers will be passed through
        `eq1` before RDMs are formed.

    Returns
    -------
    models_dict : dict
        The models in orthogonalized and non-orthogonalized format.
    """
    for model in modelnames:
        assert model in MODELS

    if (bias is not None) or (kappa is not None):
        numbers_rescaled = np.interp(
            NUMBERS, (NUMBERS.min(), NUMBERS.max()), (-1.0, 1.0)
        )
        numbers = eq1(numbers_rescaled, bias=bias, kappa=kappa)
    else:
        numbers = NUMBERS

    ndim = len(numbers)
    if rdm_size == "18x18":
        numbers = np.hstack((numbers, numbers))
    else:
        assert rdm_size == "9x9"

    # Define the models
    model_numberline = calc_rdm(numbers, normalize=True)

    model_extremity = calc_rdm(np.abs(numbers - 5), normalize=True)

    # following models are only 18x18
    model_color = np.repeat(
        np.repeat(np.abs(np.identity(2) - 1), ndim, axis=1), ndim, axis=0
    )

    model_digit = np.tile(np.abs(np.identity(ndim) - 1), (2, 2))

    model_parity = np.tile(np.tile(np.abs(np.identity(2) - 1), ndim).T, ndim)

    model_numXcat = np.vstack(
        (
            np.hstack(
                (
                    calc_rdm(numbers[:9], normalize=True),
                    np.fliplr(calc_rdm(numbers[:9], normalize=True)),
                )
            ),
            np.hstack(
                (
                    np.fliplr(calc_rdm(numbers[:9], normalize=True)),
                    calc_rdm(numbers[:9], normalize=True),
                )
            ),
        )
    )

    # first put all models in a dict, then remove the ones we don't need
    if rdm_size == "9x9":
        models_dict = {
            "no_orth": {"numberline": model_numberline, "extremity": model_extremity}
        }
    else:
        assert rdm_size == "18x18"
        models_dict = {
            "no_orth": {
                "digit": model_digit,
                "color": model_color,
                "parity": model_parity,
                "numberline": model_numberline,
                "numXcat": model_numXcat,
                "extremity": model_extremity,
            }
        }

    to_remove = []
    for key in models_dict["no_orth"]:
        if key not in modelnames:
            to_remove.append(key)
    for key in to_remove:
        del models_dict["no_orth"][key]

    nmodels = len(models_dict["no_orth"])

    # Orthogonalize models
    # First make a copy of non-orth models, this copy will be modified below
    models_dict["orth"] = copy.deepcopy(models_dict["no_orth"])

    # orthogonalize recursively using spm_orth
    # The last column vector in the output of the spm_orth function
    # is orthogonalized with respect to all previous column vectors.
    #
    # Iterate through models, for each model, obtain its orthed version
    # by placing it in the last column in a matrix X.
    # For example, for models 1, 2, 3, 4, 5 do the following column orders:
    #
    # 2 3 4 5 1
    # 1 3 4 5 2
    # 1 2 4 5 3
    # 1 2 3 5 4
    # 1 2 3 4 5
    #
    # ... to obtain the orthed models (5 calls to spm_orth needed).
    #
    #  NOTE: Each column must be mean-centered, and only the vector
    #        form of a square symmetric RDM should be passed to spm_orth
    #        That is, use "squareform", or extract the lower (or upper)
    #        triangle of the RDM, excluding the diagonal (but do not mix
    #        these approaches).
    model_arrs = np.stack(list(models_dict["orth"].values()), axis=-1)
    imodels = np.arange(nmodels)
    orthmodels = []
    for imodel in imodels:
        orth_col_order = np.hstack((imodels[imodels != imodel], imodel)).tolist()
        X = np.full((len(squareform(model_numberline)), nmodels), np.nan)
        for imod in range(nmodels):
            icol = orth_col_order.index(imod)
            vec = squareform(model_arrs[..., imod])  # convert to condensed vector
            X[..., icol] = vec - vec.mean()  # mean-center before orth
        X_orth = spm_orth(X)
        orth_model = squareform(
            X_orth[..., -1]
        )  # convert to square symmetric RDM again
        orthmodels.append((modelnames[imodel], orth_model))

    # update copied dict with final, orthogonalized models
    for modelname, orthmodel in orthmodels:
        models_dict["orth"][modelname] = orthmodel

    return models_dict
