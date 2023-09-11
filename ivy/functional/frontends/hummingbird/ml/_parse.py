
def _parse_sklearn_single_model(topology, model, inputs):
    """
    This function handles all sklearn objects composed by a single model.

    Args:
        topology: The ``hummingbitd.ml._topology.Topology`` object where the model will be added
        model: A scikit-learn model object
        inputs: A list of `onnxconverter_common.topology.Variable`s

    Returns:
        A list of output `onnxconverter_common.topology.Variable` which will be passed to next stage
    """
    if isinstance(model, str):
        raise RuntimeError("Parameter model must be an object not a " "string '{0}'.".format(model))

    alias = get_sklearn_api_operator_name(type(model))
    this_operator = topology.declare_logical_operator(alias, model)
    this_operator.inputs = inputs

    # We assume that all scikit-learn operators produce a single output.
    variable = topology.declare_logical_variable("variable")
    this_operator.outputs.append(variable)

    return this_operator.outputs


def _parse_sklearn_api(topology, model, inputs):
    """
    This is a delegate function adding the model to the input topology.
    It does nothing but invokes the correct parsing function according to the input model's type.

    Args:
        topology: The ``hummingbitd.ml._topology.Topology`` object where the model will be added
        model: A scikit-learn model object
        inputs: A list of `onnxconverter_common.topology.Variable`s

    Returns:
        The output `onnxconverter_common.topology.Variable`s produced by the input model
    """
    tmodel = type(model)
    if tmodel in sklearn_api_parsers_map:
        outputs = sklearn_api_parsers_map[tmodel](topology, model, inputs)
    else:
        outputs = _parse_sklearn_single_model(topology, model, inputs)

    return outputs

def parse_sklearn_api_model(model, extra_config={}):
    """
    Puts *scikit-learn* object into an abstract representation so that our framework can work seamlessly on models created
    with different machine learning tools.

    Args:
        model: A model object in scikit-learn format

    Returns:
        A `onnxconverter_common.topology.Topology` object representing the input model
    """
    assert model is not None, "Cannot convert a mode of type None."

    raw_model_container = CommonSklearnModelContainer(model) #onnxcoverter_common/container.py

    # Declare a computational graph. It will become a representation of
    # the input scikit-learn model after parsing.
    topology = Topology(raw_model_container)

    # Declare input variables.
    inputs = _declare_input_variables(topology, raw_model_container, extra_config)

    # Parse the input scikit-learn model into a topology object.
    # Get the outputs of the model.
    outputs = _parse_sklearn_api(topology, model, inputs)

    # Declare output variables.
    _declare_output_variables(raw_model_container, extra_config, outputs)

    return topology

def _declare_input_variables(topology, raw_model_container, extra_config):
    # Declare input variables.
    inputs = []
    n_inputs = extra_config[constants.N_INPUTS] if constants.N_INPUTS in extra_config else 1
    if constants.INPUT_NAMES in extra_config:
        assert n_inputs == len(extra_config[constants.INPUT_NAMES])
    if constants.TEST_INPUT in extra_config:
        from onnxconverter_common.data_types import (
            FloatTensorType,
            DoubleTensorType,
            Int32TensorType,
            Int64TensorType,
            StringTensorType,
        )

        test_input = extra_config[constants.TEST_INPUT] if n_inputs > 1 else [extra_config[constants.TEST_INPUT]]
        for i in range(n_inputs):
            input = test_input[i]
            input_name = (
                extra_config[constants.INPUT_NAMES][i] if constants.INPUT_NAMES in extra_config else "input_{}".format(i)
            )
            if input.dtype == np.float32:
                input_type = FloatTensorType(input.shape)
            elif input.dtype == np.float64:
                input_type = DoubleTensorType(input.shape)
            elif input.dtype == np.int32:
                input_type = Int32TensorType(input.shape)
            elif input.dtype == np.int64:
                input_type = Int64TensorType(input.shape)
            elif input.dtype.kind in constants.SUPPORTED_STRING_TYPES:
                input_type = StringTensorType(input.shape)
            else:
                raise NotImplementedError(
                    "Type {} not supported. Please fill an issue on https://github.com/microsoft/hummingbird/.".format(
                        input.dtype
                    )
                )
            inputs.append(topology.declare_logical_variable(input_name, type=input_type))
    else:
        # We have no information on the input. Sklearn/Spark-ML always gets as input a single dataframe,
        # therefore by default we start with a single `input` variable
        input_name = extra_config[constants.INPUT_NAMES][0] if constants.TEST_INPUT in extra_config else "input"
        var = topology.declare_logical_variable(input_name)
        inputs.append(var)

    # The object raw_model_container is a part of the topology we're going to return.
    # We use it to store the inputs of the Sklearn/Spark-ML's computational graph.
    for variable in inputs:
        raw_model_container.add_input(variable)

    return inputs

def _declare_output_variables(raw_model_container, extra_config, outputs):
    # Declare output variables.
    # Use the output names specified by the user, if any
    if constants.OUTPUT_NAMES in extra_config:
        assert len(extra_config[constants.OUTPUT_NAMES]) == len(outputs)
        for i in range(len(outputs)):
            outputs[i].raw_name = extra_config[constants.OUTPUT_NAMES][i]

    # The object raw_model_container is a part of the topology we're going to return.
    # We use it to store the outputs of the Sklearn/Spark-ML's computational graph.
    for variable in outputs:
        raw_model_container.add_output(variable)








