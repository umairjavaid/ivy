import numpy as np
import torch
import ivy
from .onnxconverter_common.topology import Topology as ONNXTopology
from .onnxconverter_common.registration import get_converter
from ._executor import Executor
from .exceptions import MissingConverter
from .operator_converters import constants
from .containers import PyTorchSklearnContainerClassification


class Topology:
    def __init__(self, input_container):
        # returns topology object, abstract class containg decisiontree is passed to it
        self.onnxconverter_topology = ONNXTopology(input_container)
        # Declare an object to provide variables' and operators' naming mechanism.
        # One global scope is enough for parsing Hummingbird's supported input models.
        self.scope = self.onnxconverter_topology.declare_scope("__root__")

    @property
    def input_container(self):
        """Returns the input container wrapping the original input model."""
        return self.onnxconverter_topology.raw_model

    @property
    def variables(self):
        """Returns all the logical variables of the topology."""
        return self.scope.variables

    def declare_logical_variable(self, original_input_name, type=None):
        """
        This function creates a new logical variable within the topology.

        If original_input_name has been used to create other variables,
        the new variable will hide all other variables created using
        original_input_name.
        """
        return self.scope.declare_local_variable(original_input_name, type=type)

    def declare_logical_operator(self, alias, model=None):
        """This function is used to declare new logical operator."""
        return self.scope.declare_local_operator(alias, model)

    def topological_operator_iterator(self):
        """
        This is an iterator of all operators in the Topology object.

        Operators are returned in a topological order.
        """
        return self.onnxconverter_topology.topological_operator_iterator()


# --- Helpers --- #
# --------------- #


def _get_batch_size(batch):
    if isinstance(batch, tuple):
        return batch[0].shape[0]

    assert isinstance(batch, np.ndarray)
    return batch.shape[0]


# --- Main --- #
# ------------ #


def convert(topology, backend, test_input, device, extra_config={}):
    """
    This function is used to convert a `Topology` object into a *backend* model.

    Args:
        topology: The `Topology` object that will be converted into a backend model
        backend: Which backend the model should be run on
        test_input: Inputs for PyTorch model tracing
        device: Which device the translated model will be run on
        extra_config: Extra configurations to be used by individual operator converters

    Returns:
        A model implemented in the selected backend
    """
    assert topology is not None, "Cannot convert a Topology object of type None."
    assert backend is not None, "Cannot convert a Topology object into backend None."
    assert device is not None, "Cannot convert a Topology object into device None."

    operator_map = {}

    # if tvm_installed():
    #     import tvm

    #     tvm_backend = tvm.__name__

    for operator in topology.topological_operator_iterator():
        converter = get_converter(operator.type)
        if converter is None:
            raise MissingConverter(
                "Unable to find converter for {} type {} with extra config: {}.".format(
                    operator.type,
                    type(getattr(operator, "raw_model", None)),
                    extra_config,
                )
            )

        operator_map[operator.full_name] = converter(operator, device, extra_config)

    # Set the parameters for the model / container
    n_threads = (
        None
        if constants.N_THREADS not in extra_config
        else extra_config[constants.N_THREADS]
    )

    # We set the number of threads for torch here to avoid errors in case we JIT.
    # We set intra op concurrency while we force operators to run sequentially.
    # We can revise this later, but in general we don't have graphs requireing inter-op
    # parallelism.
    # if n_threads is not None:
    #     if torch.get_num_interop_threads() != 1:
    #         torch.set_num_interop_threads(1)
    #     torch.set_num_threads(n_threads)

    operators = list(topology.topological_operator_iterator())
    executor = Executor(
        topology.input_container.input_names,
        topology.input_container.output_names,
        operator_map,
        operators,
        extra_config,
    ).eval()

    if False:
        raise NotImplementedError
    elif False:
        raise NotImplementedError
    else:
        # Set the device for the model.
        if device != "cpu":
            if backend == torch.__name__ or torch.jit.__name__:
                executor = executor.to(device)

        # If the backend is tochscript, jit the model.
        # if backend == torch.jit.__name__:
        #     trace_input, _ = _get_trace_input_from_test_input(test_input,
        # remainder_size, extra_config)
        #     executor = _jit_trace(executor, trace_input, device, extra_config)
        #     torch.jit.optimized_execution(executor)

        hb_model = executor
    hb_model = ivy.unify(hb_model, source="torch")
    # Return if the container is not needed.
    if constants.CONTAINER in extra_config and not extra_config[constants.CONTAINER]:
        return hb_model

    # We scan the operators backwards until we find an operator with a defined type.
    # This is necessary because ONNX models can have arbitrary operators doing casting,
    # reshaping etc.
    idx = len(operators) - 1
    while (
        idx >= 0
        and not operator_map[operators[idx].full_name].regression
        and not operator_map[operators[idx].full_name].classification
        and not operator_map[operators[idx].full_name].anomaly_detection
        and not operator_map[operators[idx].full_name].transformer
    ):
        idx -= 1

    force_transformer = False
    if idx < 0:
        force_transformer = True

    # If is a transformer, we need to check whether there is
    # another operator type before.
    # E.g., normalization after classification.
    if not force_transformer:
        tmp_idx = idx
        if operator_map[operators[idx].full_name].transformer:
            while (
                idx >= 0
                and not operator_map[operators[idx].full_name].regression
                and not operator_map[operators[idx].full_name].classification
                and not operator_map[operators[idx].full_name].anomaly_detection
            ):
                idx -= 1
            if idx < 0:
                idx = tmp_idx

    # Get the proper container type.

    container = PyTorchSklearnContainerClassification

    n_threads = (
        None
        if constants.N_THREADS not in extra_config
        else extra_config[constants.N_THREADS]
    )
    batch_size = (
        None
        if constants.TEST_INPUT not in extra_config
        else _get_batch_size(test_input)
    )
    hb_container = container(hb_model, n_threads, batch_size, extra_config=extra_config)

    return hb_container
