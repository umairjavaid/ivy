# This dictionary defines the shape calculators which can be invoked in the conversion framework defined in
# topology.py. A key in this dictionary is an operator's unique ID (e.g., string and type) while the associated value
# is the callable object used to infer the output shape(s) for the operator specified by the key.
_shape_calculator_pool = {}

def get_shape_calculator(operator_name):
    '''
    Given an Operator object (named operator) defined in topology.py, we can retrieve its shape calculation function.
    >>> from onnxmltools.convert.common._topology import Operator
    >>> operator = Operator('dummy_name', 'dummy_scope', 'dummy_operator_type', None)
    >>> get_shape_calculator(operator.type)  # Use 'dummy_operator_type' for dictionary looking-up

    :param operator_name: An operator ID
    :return: a shape calculation function for a specific Operator object
    '''
    if operator_name not in _shape_calculator_pool:
        raise ValueError('Unsupported shape calculation for operator %s' % operator_name)
    return _shape_calculator_pool[operator_name]