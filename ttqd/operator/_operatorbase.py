

from abc import abstractmethod, ABC



class BaseOperator(object):
    r"""
    Base Operator class
    """

    @abstractmethod
    def __init__(self, **kwargs):
        # self.label = kwargs.get("label", None)
        self._mat = None # matrix representation of the operator
