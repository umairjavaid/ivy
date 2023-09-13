from collections import defaultdict

from .exceptions import MissingConverter
from ._utils import torch_installed

def _build_backend_map():
    """
    The set of supported backends is defined here.
    """
    backends = defaultdict(lambda: None)

    if torch_installed():
        #import torch
        import ivy.functional.frontends.torch as torch
        backends[torch.__name__] = torch.__name__
        backends["py" + torch.__name__] = torch.__name__  # For compatibility with earlier versions.

        backends[torch.jit.__name__] = torch.jit.__name__
        backends["torchscript"] = torch.jit.__name__  # For reference outside Hummingbird.

    return backends

backends = _build_backend_map()





def _build_sklearn_operator_list():
    """
    Put all supported Sklearn operators on a list.
    """
    if True:
        # Tree-based models
        from sklearn.ensemble import (
            RandomForestClassifier,
            RandomForestRegressor,
        )

        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

        
        # Preprocessing
        from sklearn.preprocessing import (
            Binarizer,
            KBinsDiscretizer,
            LabelEncoder,
            MaxAbsScaler,
            MinMaxScaler,
            Normalizer,
            OneHotEncoder,
            PolynomialFeatures,
            RobustScaler,
            StandardScaler,
        )

        try:
            from sklearn.preprocessing import Imputer
        except ImportError:
            # Imputer was deprecate in sklearn >= 0.22
            Imputer = None

        # Features
        from sklearn.feature_selection import SelectKBest, SelectPercentile, VarianceThreshold

        # Mixture models
        from sklearn.mixture import BayesianGaussianMixture

        supported_ops = [
            # Trees
            DecisionTreeClassifier,
            DecisionTreeRegressor,
            OneHotEncoder,
            RandomForestClassifier,
            RandomForestRegressor,

            # Preprocessing
            Binarizer,
            KBinsDiscretizer,
            LabelEncoder,
            MaxAbsScaler,
            MinMaxScaler,
            Normalizer,
            PolynomialFeatures,
            RobustScaler,
            StandardScaler,

            # Feature selection
            SelectKBest,
            SelectPercentile,
            VarianceThreshold,
            # Mixture models
            BayesianGaussianMixture,
        ]

        # Remove all deprecated operators given the sklearn version. E.g., Imputer for sklearn > 0.21.3.
        return [x for x in supported_ops if x is not None]

    return []

sklearn_operator_list = _build_sklearn_operator_list()

def _build_sklearn_api_operator_name_map():
    """
    Associate Sklearn with the operator class names.
    If two scikit-learn (API) models share a single name, it means they are equivalent in terms of conversion.
    """
    # Pipeline ops. These are ops injected by the parser not "real" sklearn operators.
    pipeline_operator_list = [
        "ArrayFeatureExtractor",
        "Concat",
        "Multiply",
        "Bagging",
    ]

    return {
        k: "Sklearn" + k.__name__ if hasattr(k, "__name__") else k
        for k in sklearn_operator_list
        + pipeline_operator_list
    }


sklearn_api_operator_name_map = _build_sklearn_api_operator_name_map()

def get_sklearn_api_operator_name(model_type):
    """
    Get the operator name for the input model type in *scikit-learn API* format.

    Args:
        model_type: A scikit-learn model object (e.g., RandomForestClassifier)
                    or an object with scikit-learn API (e.g., LightGBM)

    Returns:
        A string which stands for the type of the input model in the Hummingbird conversion framework
    """
    if model_type not in sklearn_api_operator_name_map:
        raise MissingConverter("Unable to find converter for model type {}.".format(model_type))
    return sklearn_api_operator_name_map[model_type]

TREE_IMPLEMENTATION = "tree_implementation"
"""Which tree implementation to use. Values can be: gemm, tree_trav, perf_tree_trav."""

TREE_OP_PRECISION_DTYPE = "tree_op_precision_dtype"
"""Which data type to be used for the threshold and leaf values of decision nodes. Values can be: float32 or float64."""

REMAINDER_SIZE = "remainder_size"
"""Determines the number of rows that an auxiliary remainder model can accept."""

INPUT_NAMES = "input_names"
"""Set the names of the inputs. Assume that the numbers of inputs_names is equal to the number of inputs."""

OUTPUT_NAMES = "output_names"
"""Set the names of the outputs."""

CONTAINER = "container"
"""Boolean used to chose whether to return the container for Sklearn API or just the model."""

BATCH_SIZE = "batch_size"
"""Select whether to partition the input dataset at inference time in N batch_size partitions."""

N_THREADS = "n_threads"
"""Select how many threads to use for scoring. This parameter will set the number of intra-op threads.
Inter-op threads are by default set to 1 in Hummingbird. Check `tests.test_extra_conf.py` for usage examples."""

MAX_STRING_LENGTH = "max_string_length"
"""Maximum expected length for string features. By default this value is set using the training information."""
