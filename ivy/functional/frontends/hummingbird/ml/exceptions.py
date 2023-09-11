class MissingConverter(RuntimeError):
    """
    Raised when there is no registered converter for a machine learning operator.
    """

    def __init__(self, msg):
        super().__init__(msg + _missing_converter)