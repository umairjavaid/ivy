
_missing_converter = """
It usually means the pipeline being converted contains a
transformer or a predictor with no corresponding converter implemented.
Please fill an issue at https://github.com/microsoft/hummingbird.
"""


class MissingConverter(RuntimeError):
    """
    Raised when there is no registered converter for a machine learning operator.
    """

    def __init__(self, msg):
        super().__init__(msg + _missing_converter)