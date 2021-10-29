"""Implements the test cases from the report"""
from abc import abstractmethod


class MatrixOde:
    """d/dt = B @ A(t)"""

    @abstractmethod
    def A(self):
        pass

    def A_dot(self):
        pass
