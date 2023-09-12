import abc

class OperatorBase:
    __metaclass__ = abc.ABCMeta

    @property
    @abc.abstractmethod
    def full_name(self):
        """
        Return a globally unique operator ID
        """
        pass

    @property
    @abc.abstractmethod
    def input_full_names(self):
        """
        Return all input variables' names
        """
        pass

    @property
    @abc.abstractmethod
    def output_full_names(self):
        """
        Return all outpu variables' names
        """
        pass

    @property
    @abc.abstractmethod
    def original_operator(self):
        """
        Return the original operator/layer
        """
        pass


class ScopeBase:
    __metaclass__ = abc.ABCMeta

    pass