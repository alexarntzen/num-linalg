import unittest
from test.case_matrix_ode import *


class TestMatrixOde(unittest.TestCase):
    def testSolutionMap(self):
        eq = HeatEquation(5, 5, 3)
        eq.A(3)



class MyTestCase(unittest.TestCase):
    def test_something(self):
        pass


if __name__ == '__main__':
    unittest.main()
