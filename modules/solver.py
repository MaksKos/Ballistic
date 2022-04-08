""" 
version 1.0

This module solves inverse problem of internal ballistics.

"""
import numpy as np
from pyballistics import ozvb_lagrange, get_db_powder, get_powder_names
from scipy.optimize import minimize, OptimizeResult

class Solver():


    def __init__(self, initial_parametrs, *args, **kwargs) -> None:

        if  initial_parametrs['powders'][0]['dbname'] is None:
            raise TypeError('dbname is None')
        if  initial_parametrs['powders'][0]['omega'] is None:
            raise TypeError('omega is None')  
        if  initial_parametrs['init_conditions']['W_0'] is None:
            raise TypeError('W_0 is None')     

        self._min_criterion = None
        self.result = None
        self.initial_parametrs = initial_parametrs
        return None

    def solution(self) -> dict:
        """

        The main function that find optimal solve (by min criterian).
        In this realisation min criterian is mass of cannon.

        """
        wq_0 = self.initial_parametrs['powders'][0]['omega']/self.initial_parametrs['init_conditions']['q']
        ro_0 = self.initial_parametrs['powders'][0]['omega']/self.initial_parametrs['init_conditions']['W_0']

        initial_guess = np.array([[wq_0, ro_0]])
        result = minimize(self.__minimum_mass, initial_guess, method='nelder-mead',
                         options={'xtol': 1e-4, 'disp': True, 'return_all': True}) 
        self.result = OptimizeResult(result)
        return self.result.success

    def get_solution(self) -> dict:

        if self.result is None:
            print ("No result")
            return {None}
        
        if not self.result.success:
            print("No success in result")
            return {None}

        return {
            'name': self.initial_parametrs['powders'][0]['dbname'],
            'wq': self.result.x[0],
            'ro': self.result.x[1],
            "mass": self._min_criterion,
        }

    @staticmethod 
    def check_ozvb(result_dict: dict) -> bool:
        if result_dict['stop_reason'] == 'error':
            return False
        if result_dict['stop_reason'] == 'step_max':
            return False
        return True
    
    def __minimum_mass(self, initial_guess):
        """

        Penalty function for Nelder Mead otimization method.
        The creterion is minimum of cannon's mass.

        """
        #factor scaling
        a = np.array([1e-2, 1])
        w = initial_guess[0]*self.initial_parametrs['init_conditions']['q']
        ro = initial_guess[1]
        self.initial_parametrs['powders'][0]['omega'] = w
        self.initial_parametrs['init_conditions']['W_0'] = w/ro
 
        result_dict = ozvb_lagrange(self.initial_parametrs)
      
        if not self.check_ozvb(result_dict):
            return 2e10

        self._min_criterion = self.cannon_mass(result_dict)

        diff_velocity = self.initial_parametrs['stop_conditions']['v_p'] - result_dict['layers'][-1]['u'][-1]
        diff_lenght = result_dict['layers'][-1]['x'][-1] - self.initial_parametrs['stop_conditions']['x_p']

        heviside = np.heaviside([diff_velocity, diff_lenght], 0)
        del_array = np.abs(np.array([diff_velocity, diff_lenght])) 
        
        return self._min_criterion + np.sum(1e4*(a*del_array)*heviside)

    @staticmethod
    def make_matrix(result_dict, label):
        """
        Method make matrix with dimension [layers x cells] 
        result_dict: <dict> from ozvb
        label: name of parametr ('p', 'x', etc.)
        """
        if not label in ['x', 'p', 'T', 'u']:
            raise TypeError('undefine "label": ', label)
        matrix = list()
        for layer in result_dict['layers']:
            matrix.append(layer[label])
        return np.array(matrix)


    def cannon_mass(self, result_dict=None):
        
        if result_dict is None:
            raise TypeError("result_dict is empty")

        diametr = self.initial_parametrs['init_conditions']['d']
        matrix_x = self.make_matrix(result_dict, 'x')
        matrix_p = self.make_matrix(result_dict, 'p')

        cannon = Cannon(diametr, matrix_x, matrix_p)
        cannon.cannon_geometry()
        cannon.get_mass()
        return cannon.mass


class Cannon():

    cone_k1 = 1/75
    cone_k2 = 1/2.5
    cone_k6 = 1/200
    cone_k7 = 1/30
    hi = 1.5
    bottle_capacity = 1.2
    n_safety = 1
    sigma_steel = 1e9
    ro = 7800
    W_sn = 0 

    def __init__(self, diametr, coordinate, pressure) -> None:
        self.diametr = diametr
        self.__matrix_x = coordinate
        self.__matrix_p = pressure
        self.x = None
        self.p = None
        self.r_inside = None
        self.r_outside = None
        self.r_inside_coordinate = None
        self.r_outside_coordinate = None
        self.coordinate = None
        self.volume = None
        self.mass = None

    def __inside_geometry(self):
        """
        Geometry of combustion chamber and tube
        """
        diametr = self.diametr
        W_0 = diametr**2*np.math.pi/4 * np.abs(self.__matrix_x[0,0])
        l_2 = 0.55*diametr if np.sqrt(self.hi) <= 1.25 else 0.9*diametr
        l_6 = 2.5*diametr
        l_7 = 0.1*diametr
        d_4 = diametr + self.cone_k7*l_7
        d_3 = d_4 + self.cone_k6*l_6 
        d_2 = d_3 + self.cone_k2*l_2
        d_k = self.bottle_capacity*diametr
        W_6 = np.math.pi/12*l_6*(d_3**2 + d_3*d_4 + d_4**2)
        W_2 = np.math.pi/12*l_2*(d_3**2 + d_3*d_2 + d_2**2)
        W_7 = np.math.pi/12*l_7*(d_4**2 + d_4*diametr + diametr**2)
        W_1 = 1.1*W_0-W_2-W_6-W_7+self.W_sn
        l_1 = W_1*12/np.math.pi / (d_k**2 + d_k*d_2 + d_2**2)

        self.r_inside_coordinate = np.cumsum([0, l_1, l_2, l_6, l_7, self.__matrix_x[-1][-1]])
        self.r_inside = np.array([d_k, d_4, d_3, d_2, diametr, diametr]) / 2

    def __outside_geometry(self):
        """
        Geometry of outer shell
        """
        if self.r_inside_coordinate is None:
            raise ValueError("empty inside coordinate")
        if self.r_inside is None:
            raise ValueError("empty inside radius")

        l_k = self.r_inside_coordinate[4]
        self.r_outside_coordinate = np.array([0, 1.2*l_k, 1.6*l_k, l_k + self.__matrix_x[-1,-1]])

        if self.pressure_on_tube():
            pass
        if self.p.max() > self.sigma_steel:
            raise ValueError("pressure in tube more than sigma (steel)")

        pressure_inside = np.interp(self.r_outside_coordinate, self.x, self.p)
        sqr = (3*self.sigma_steel + 2*pressure_inside) / (3*self.sigma_steel - 4*pressure_inside)
        
        if min(sqr) < 0:
            raise ValueError("pressure in tube destroy cannon")
        radius_inside = np.interp(self.r_outside_coordinate, self.r_inside_coordinate, self.r_inside)
        radius_outside = radius_inside*np.sqrt(sqr)
        self.r_outside = np.array([max(radius_outside[i], 1.075*radius_inside[i]) for i in range(len(radius_outside))])

    def cannon_geometry(self):
        """
        Method construct cannon
        """   

        self.__inside_geometry()
        self.__outside_geometry()
        
        self.coordinate = np.unique(np.concatenate((self.r_inside_coordinate, self.r_outside_coordinate), axis=None))
        r1 = np.interp(self.coordinate, self.r_outside_coordinate, self.r_outside)
        r2 = np.interp(self.coordinate, self.r_inside_coordinate, self.r_inside)
        a_21 = r2/r1
        pressure_real = 3/2*self.sigma_steel*(a_21**2 - 1)/(2*a_21**2 + 1)
        pressure_coordinate = np.interp(self.coordinate, self.x, self.p)
        n_real = pressure_real/pressure_coordinate
        if min(n_real) < 1:
            TypeError("check of real pressure fail")
        self.r_inside = r2
        self.r_outside = r1
        del self.r_inside_coordinate, self.r_outside_coordinate

    def get_volume(self):
        """
        Calculate volume of cannon 
        """
        if self.coordinate is None:
            raise ValueError("empty coordinate")
        x = self.coordinate
        r1 = self.r_outside
        r2 = self.r_inside
        volume_1 = 1/3*np.math.pi*(x[1:]-x[:-1])*(r1[:-1]**2 + r1[:-1]*r1[1:] + r1[1:]**2) 
        volume_2 = 1/3*np.math.pi*(x[1:]-x[:-1])*(r2[:-1]**2 + r2[:-1]*r2[1:] + r2[1:]**2) 
        self.volume = np.sum(volume_1-volume_2)

    def get_mass(self):
        if self.volume is None:
            self.get_volume()
        self.mass = self.volume*self.ro

    def pressure_on_tube(self):
        """
        Method for calculate pressure distribution along the cannon's tube
        """

        index = np.argmax(self.__matrix_p)
        border = index%self.__matrix_p.shape[1]
        layer = index//self.__matrix_p.shape[1]
        pressure_max = self.__matrix_p[layer, border]

        pressure = list([pressure_max, pressure_max])
        coordinate = list([self.__matrix_x[0,0], self.__matrix_x[layer, border]])
        previous_position = coordinate[-1]
        current_position = coordinate[-1]

        for i in range(layer+1, self.__matrix_x.shape[0]):

            ind = np.argmax(self.__matrix_p[i][border:])
            current_position = self.__matrix_x[i][ind+border]
            if current_position-previous_position >= 0:
                pressure.append(self.__matrix_p[i][ind+border])
                coordinate.append(self.__matrix_x[i][ind+border])
                previous_position = current_position
                border += ind
            else:
                pressure.append(self.__matrix_p[i][ind+border])
                coordinate.append(self.__matrix_x[i][-1])
        pressure.append(self.__matrix_p[-1][-1])
        coordinate.append(self.__matrix_x[-1][-1])
        self.x = np.array(coordinate)
        self.x += np.abs(self.x[0])
        self.p = np.array(pressure)