import unittest
import numpy as np
import numpy.testing as nptest
from modules.solver import Cannon


class TestCannon(unittest.TestCase):

    def setUp(self):
        x = np.array([
            [0, 0.25, 0.5, 0.75, 0.85, 1],
            [0, 0.5, 1, 1.25, 1.5, 2],
            [0, 1, 2, 3, 3.5, 4],
            [0, 1, 3, 4, 5, 6],
        ])
        p = np.array([
            [1, 1, 1, 1, 1, 1],
            [2, 2, 1.5, 1, 1, 1],
            [4, 10, 7, 4, 2, 2],
            [0.5, 1, 0.8, 6, 0.2, 0.2],
        ])
        lk = 2
        self.cannon = Cannon(0.125, x, p, lk)

    def test_inside_geometry(self):
        Cannon.n = 7
        self.cannon._Cannon__inside_geometry()
        r_coordinate = np.array([0, 1.153192, 1.221942, 1.534442, 1.546942, 5.546942])
        r_inside = np.array([0.078125, 0.07724, 0.06349, 0.062708, 0.0625, 0.0625])
        coordinate = np.array([0, 0.92449, 1.848981, 2.773471, 3.697961, 4.622451, 5.546942])
        self.assertIsNone(nptest.assert_allclose(r_coordinate, self.cannon.r_inside_coordinate, rtol=1e-5))
        self.assertIsNone(nptest.assert_allclose(r_inside, self.cannon.r_inside, rtol=1e-5))
        self.assertIsNone(nptest.assert_allclose(coordinate, self.cannon.coordinate, rtol=1e-5))
    
    def test_outside_geometry(self):
        Cannon.n = 3
        self.cannon.sigma_steel = 1000*1e6
        self.cannon.n_safety = 1.2
        self.cannon.coordinate = np.array([0, 1, 2])
        self.cannon.r_inside_coordinate = np.array([0, 1, 2])
        self.cannon.r_inside = np.array([0.13, 0.125, 0.125])/2
        self.cannon.pressure = np.array([6e8, 5e8, 3e8])
        self.cannon._Cannon__outside_geometry()
        radius = np.array([0.39538 , 0.165359, 0.096514])
        self.assertIsNone(nptest.assert_allclose(radius, self.cannon.r_outside, rtol=1e-5))

    def test_pressure_on_tube(self):
        self.cannon.coordinate = np.array([0, 1, 2, 3, 4, 5, 6])
        Cannon.n = 7
        presure = np.array([4, 10, 7, 4, 6, 0.2, 0.2])
        self.cannon._Cannon__pressure_on_tube()
        self.assertIsNone(nptest.assert_allclose(presure, self.cannon.pressure))


    def test_get_volume(self):
        self.cannon.coordinate = np.array([0, 5, 10])
        self.cannon.r_inside_coordinate = np.array([0, 5, 10])
        self.cannon.r_outside = np.array([4, 3, 1])
        self.cannon.r_inside = np.array([1, 2, 0.5])
        self.assertEqual(np.round(self.cannon.get_volume(), 4), 197.6585)

    def test_get_mass(self):
        self.cannon.coordinate = np.array([0, 5, 10])
        self.cannon.r_inside_coordinate = self.cannon.coordinate
        self.cannon.r_outside = np.array([4, 3, 1])
        self.cannon.r_inside = np.array([1, 2, 0.5])
        self.assertEqual(int(self.cannon.get_mass()), int(197.6585*self.cannon.ro))


if __name__ == '__main__':
    unittest.main()