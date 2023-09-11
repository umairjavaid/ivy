# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Hummingbird main (converters) API.
"""
from copy import deepcopy
import psutil
import numpy as np

from .operator_converters import constants
from ._parse import parse_sklearn_api_model, parse_onnx_api_model, parse_sparkml_api_model
from ._topology import convert as topology_converter
from ._utils import (
    assert_torch_installed,
    assert_lightgbm_installed,
    assert_xgboost_installed,
    pandas_installed,
    sparkml_installed,
    is_pandas_dataframe,
    is_spark_dataframe,
    tvm_installed,
)
from .exceptions import MissingConverter, MissingBackend
from .supported import backends
from sklearn.utils import all_estimators
from sklearn.utils.validation import check_is_fitted

# Invoke the registration of all our converters.
from . import operator_converters  # noqa

# Set up the converter dispatcher.
from .supported import xgb_operator_list  # noqa
from .supported import lgbm_operator_list  # noqa


#backend: "torch"
#device: "cpu"
#extra_config: {'container': True, 'n_threads': 1}
#model: DecisionTreeClassifier(max_depth=3)
def _convert_sklearn(model, backend, test_input, device, extra_config={}):
    """
    This function converts the specified *scikit-learn* (API) model into its *backend* counterpart.
    The supported operators and backends can be found at `hummingbird.ml.supported`.
    """

    # Parse scikit-learn model as our internal data structure (i.e., Topology)
    # We modify the scikit learn model during translation.
    model = deepcopy(model)
    #what is topology? what happens in parse_sklearn_api_model?
    topology = parse_sklearn_api_model(model, extra_config)

    # Convert the Topology object into a PyTorch model.
    hb_model = topology_converter(topology, backend, test_input, device, extra_config=extra_config)
    return hb_model



#backend: "pytorch"
#device: "cpu"
#model: DecisionTreeClassifier(max_depth=3)
#extra_config: {}
def _convert_common(model, backend, test_input=None, device="cpu", extra_config={}):
    """
    A common function called by convert(...) and convert_batch(...) below.
    """
    assert model is not None

    # We destroy extra_config during conversion, we create a copy here.
    extra_config = deepcopy(extra_config) #(return) {container: True, n_threads: 1}

    # Set some default configurations.
    # Add test input as extra configuration for conversion.
    #this if was not used (debug-umair)
    if (
        test_input is not None
        and constants.TEST_INPUT not in extra_config
        and (is_spark_dataframe(test_input) or len(test_input) > 0)
    ):
        extra_config[constants.TEST_INPUT] = test_input
    # By default we return the converted model wrapped into a `hummingbird.ml._container.SklearnContainer` object.
    if constants.CONTAINER not in extra_config:
        extra_config[constants.CONTAINER] = True
    # By default we set num of intra-op parallelism to be the number of physical cores available
    if constants.N_THREADS not in extra_config:
        extra_config[constants.N_THREADS] = psutil.cpu_count(logical=False)

    # We do some normalization on backends.
    if not isinstance(backend, str):
        raise ValueError("Backend must be a string: {}".format(backend))
    backend_formatted = backend.lower()
    backend_formatted = backends[backend_formatted]

    return _convert_sklearn(model, backend_formatted, test_input, device, extra_config)


def convert(model, backend, test_input=None, device="cpu", extra_config={}):
    assert constants.REMAINDER_SIZE not in extra_config
    return _convert_common(model, backend, test_input, device, extra_config)

