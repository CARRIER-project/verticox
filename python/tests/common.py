from sksurv.functions import StepFunction
import numpy as np

DEFAULT_DECIMAL = 4


def compare_stepfunctions(a: StepFunction, b: StepFunction, decimal=DEFAULT_DECIMAL):
    print('Comparing x')
    np.testing.assert_almost_equal(a.x, b.x, decimal=decimal)

    print('Comparing y')
    np.testing.assert_almost_equal(a(a.x), b(a.x), decimal=decimal)
