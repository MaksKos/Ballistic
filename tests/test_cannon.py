import unittest
import numpy as np
import numpy.testing as nptest
from modules.solver import Cannon
from pyballistics import ozvb_lagrange, get_db_powder, get_powder_names


class TestCannon(unittest.TestCase):

    def setUp(self):
        x = np.array([
            [-2, -1.5, -1, -0.4, 0],
            [-2, -1.3, -0.9, -0.5, 0.2],
            [-2, -1, -0.5, 0.6, 3],
            [-2, -0.3, 0.2, 1.8, 4],
        ])
        p = np.array([
            [2, 3.4, 4, 1, 0.5],
            [10, 15, 11, 9.5, 6],
            [6, 6.5, 7, 8, 7.5],
            [2, 3, 3.5, 1.9, 1],
        ])
        self.cannon = Cannon(0.125, x, p)

    def test_pressure_p(self):
        p = np.array([15, 15, 8, 1.9, 1])
        self.cannon.pressure_on_tube()
        self.assertIsNone(nptest.assert_allclose(p, self.cannon.p))
            

    def test_pressure_x(self):
        x = np.array([0, 0.7, 2.6, 3.8, 6])
        self.cannon.pressure_on_tube()
        self.assertIsNone(nptest.assert_allclose(x, self.cannon.x))

    def test_get_volume_error(self):
        with self.assertRaises(ValueError):
            self.cannon.get_volume()

    def test_get_volume(self):
        self.cannon.coordinate = np.array([0, 5, 10])
        self.cannon.r_outside = np.array([4, 3, 1])
        self.cannon.r_inside = np.array([1, 2, 0.5])
        self.cannon.get_volume()
        self.assertEqual(np.round(self.cannon.volume, 4), 197.6585)

    def test_get_mass(self):
        self.cannon.coordinate = np.array([0, 5, 10])
        self.cannon.r_outside = np.array([4, 3, 1])
        self.cannon.r_inside = np.array([1, 2, 0.5])
        self.cannon.get_mass()
        self.assertEqual(int(self.cannon.mass), int(197.6585*self.cannon.ro))


if __name__ == '__main__':
    unittest.main()