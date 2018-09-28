import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize


'''
To add a new fit function you need to provide a function that has the x and fit params listed
and returns the value f(x). The you need to add to the dictionary a string which describes the equation
and the number of fit parameters as an int.
'''

'''
This dictionary provides the equation string and the number of fit parameters.
'''
fit_dict = {
            'linear':('f(x) = a*x + b',2),
            'quadratic':('f(x) = a*x**2 + b*x + c',3),
            'cubic':('f(x) = a*x**3 + b*x**2 + c*x + d',4),
            'exponential':('f(x) = a*exp(b*x)',2),
            'flipped_exponential':('f(x) = a*(1 - exp(b*x))'),
            'sin_cos':('f(x) = asin(bx)+bcos(cx)+d',4)
           }


'''
Polynomial functions
'''
def linear(x,a,b):
    return a*x + b

def quadratic(x,a,b,c):
    return a*x**2 + b*x + c

def cubic(x,a,b,c,d):
    return a*x**3 + b*x**2 + c*x + d

'''
Exponential functions
'''
def exponential(x,a,b):
    return a*np.exp(b*x)

def flipped_exponential(x,a,b):
    return a*(1-np.exp(b*x))

'''
sin wave fitting is incredibly sensitive to phase c so use the following form
'''
def sin_cos(x,a,b,c,d):
    return a * np.sin(c * x) + b * np.cos(c * x) + d

def sin_const_convert(params,long=True):
    '''
    There are two equivalent forms 1) Asin(CX + B) + D and 2) asin(cx) +  bcos(cx) + d. 
    This converts the constants between 1 and 2 if long == False and 2 and 1 if long == True  
    Maths is here:
    http://www.ugrad.math.ubc.ca/coursedoc/math100/notes/trig/phase.html
    c and d or C and D remain unchanged
    '''
    if long == True:
        print('Changing: a sin(cx) + b cos(cx) + d')
        print('to : A sin (CX + B) + D')
        a = params[0]
        b = params[1]
        
        params[1] = np.arctan2(b,a) #B
        params[0] = (a**2 + b**2)**0.5 #A
        
    else:
        print('Changin : A sin (CX + B) + D')
        print('to : a sin(cx) + b cos(cx) + d')
        
        A = params[0]
        B = params[1]
        
        params[1] = A*np.cos(B)
        params[0] = A*np.sin(B)
    
    return params


    

    
class Fit:
    '''
    Generic fit object. Allows simple fitting of various functional forms together with viewing data
    and selecting appropriate values. Also provides simple statistics.
    
    Inputs:
    fit_type = function name that specifies the type of fit of interest. These are specified in the
                fit_dict and have a matching named function. The corresponding function with the same name 
                returns the value of the function for a specific set of parameter values.
                x,y = 1d numpy array data to be fitted of the form y = f(x)
                series = pandas series can be supplied in place of x,y data. It will plot data against index values
    
    guess = starting values for the fit -tuple with length = number of parameters
    lower = lower bounds on fit (optional)
    upper = upper bounds on fit (optional)
            These maybe a value, np.inf, -np.inf, None, None sets to +/- infinity, Fixed sets the values to the guess values +/- 0.001%
    logic = a numpy array of length = x and y with True or False depending on whether data is to be included
    
    methods:
    __init__()  -     initialises with fit_type and optionally with data
    add_fit_data()  - can be used to add data or update data. reset_filter will
                      remove any logical filters that have been added. If the data is different
                      length this will autoreset regardless printing a message.
    add_params()  -   add fit parameters. lower and upper bounds are optional and will be set to +/- np.inf
                      if not supplied.
    _replace_none_fixed - Internal method to replace None and Fixed values in fit limits with appropriate substitutes
    add_filter()  -       Takes a logical numpy array with True or False to indicate if data should be included in fit
    fit()             Fits the data and returns (fit_params, fit_residuals, fit_y)
    plot_fit()        Plots fit to screen or file depending on settings
    stats()           provides simple statistics on the data and filtered data sets.
    
    Outputs:
    fit_params        A list of the parameters used to fit the data. Definitions are provided in the dict type fit_dict['*Fit Type*']
    fit_residuals     np array of differences between data and fit
    fit_y             Returns the values of the fit at each point
    '''
    
    def __init__(self,fit_type,x=None,y=None,series=None):
        self.x=0
        self.y=0
        self.fit_type = fit_type
        self.fit_string,self._num_fit_params = fit_dict[fit_type]
        if type(series) == type(pd.Series()):
            x = series.index.values
            y = series.values
        
        self.add_fit_data(x,y)
    
    def add_fit_data(self,x=None,y=None,series=None,reset_filter=True):
        if type(series) == type(pd.Series()):
            x = series.index
            y = series.values
        if np.shape(x)[0] != np.shape(y)[0]:
            print('x','y')
            raise DataLengthException(x,y)
        if np.shape(x) != np.shape(self.x):
            reset_filter = True
            print('Data different length to current data, resetting logical filter')
        self.x = x
        self.y = y
        if reset_filter == True:
            self.add_filter(np.ones(np.shape(x),dtype=bool))
    
    def add_params(self,guess,lower=None,upper=None):
        _num_params = np.shape(guess)[0]
        if _num_params != self._num_fit_params:
            raise ParamNumberException(guess)
        self._params = guess
        self._lower=lower
        self._upper=upper
        self._replace_none_fixed()
    
    def _replace_none_fixed(self,nudge = 0.001):
        if self._lower == None:
            self._lower = [-np.inf]*self._num_fit_params
        if self._upper == None:
            self._upper = [np.inf]*self._num_fit_params
    
        for index,item in enumerate(self._lower):
            if item == None:
                self._lower[index] = -np.inf
            elif (item == 'Fixed') and (self._params[index] < 0):
                self._lower[index] = self._params[index]*(1.0 + nudge)
            elif (item == 'Fixed') and (self._params[index] > 0):
                self._lower[index] = self._params[index]*(1.0 - nudge)
    
            for index,item in enumerate(self._upper):
                if item == None:
                    self._upper[index] = np.inf
                elif (item == 'Fixed') and (self._params[index] >= 0):
                    self._upper[index] = self._params[index]*(1.0 + nudge)
                elif (item == 'Fixed') and (self._params[index] < 0):
                    self._upper[index] = self._params[index]*(1.0 - nudge)
    
    def add_filter(self,logic):
        if type(logic) == type(pd.Series()):
            logic = logic.values
        len_logic=np.shape(logic)[0]
        if len_logic != np.shape(self.x)[0]:
            print('x','logic')
            raise DataLengthException(x,logic)
        self.logic = logic
        self.fx = self.x[logic]
        self.fy = self.y[logic]    
    
    def fit(self,interpolate=False,factor=0.001):
        fit_output = optimize.curve_fit(globals()[self.fit_type], self.fx,self.fy,p0=self._params,bounds=(self._lower,self._upper))
        self.fit_params = fit_output[0]
        self.fit_residuals = fit_output[1]
        if interpolate == True:
            self.stats()
            original_step = (self.xdata_max - self.xdata_min) / self.xdata_length
            interpolation_step_size = factor * original_step
            self.fit_x = np.arange(self.xdata_min,self.xdata_max,interpolation_step_size)
        else:
            self.fit_x = self.x
        self.fit_y = globals()[self.fit_type](self.fit_x,*self.fit_params)
        print('\nFit : ',fit_dict[self.fit_type])
        print('Fit params : ')
        letters = [chr(c) for c in range(ord('a'),ord('z')+1)]
        for index,param in enumerate(self.fit_params):
            print(letters[index],': (', param, self._lower[index], self._upper[index],')')
        
        return (self.fit_params,self.fit_residuals,self.fit_y)
    
    def plot_fit(self,filename=None,show=False,save=False):
        if filename == None:
            filename = ' '
        plt.figure(filename)
        plt.plot(self.x,self.y,'rx')
        plt.plot(self.fx,self.fy,'bx')
        plt.plot(self.fit_x,self.fit_y,'g-')
        if save == True:
            plt.savefig(filename)
        if show == True:
            plt.show()            
            
    def stats(self,show_stats=False):
        self.ydata_mean = np.mean(self.y)
        self.ydata_std = np.std(self.y)
        self.ydata_median = np.median(self.y)
        self.ydata_max = np.max(self.y)
        self.ydata_min = np.min(self.y)
        self.xdata_max = np.max(self.x)
        self.xdata_min = np.min(self.x)
        self.xdata_length = np.shape(self.x)[0]
        
        self.fydata_mean = np.mean(self.fy)
        self.fydata_std = np.std(self.fy)
        self.fydata_median = np.median(self.fy)
        self.fydata_max = np.max(self.fy)
        self.fydata_min = np.min(self.fy)
        self.fxdata_max = np.max(self.fx)
        self.fxdata_min = np.min(self.fx)
        self.fdata_length = np.shape(self.fx)[0]
        
        if show_stats:
            print('ydata:')
            print('mean - ',self.ydata_mean)
            print('std - ',self.ydata_std)
            print('median - ',self.ydata_median)
            print('min - ',self.ydata_min)
            print('max - ',self.ydata_max)
            print('xdata:')
            print('min - ',self.xdata_min)
            print('max - ',self.xdata_max)
            print('data length - ', self.xdata_length)
            print('')
            print('ydata filtered:')
            print('mean - ',self.fydata_mean)
            print('std - ',self.fydata_std)
            print('median - ',self.fydata_median)
            print('min - ',self.fydata_min)
            print('max - ',self.fydata_max)
            print('xdata filtered:')
            print('min - ',self.fxdata_min)
            print('max - ',self.fxdata_max)
            print('data length - ', self.fxdata_length)
            
'''
Exception defintions
'''
class DataLengthException(Exception):
    def __init__(self,x,y):
        len_x = np.shape(x)[0]
        len_y = np.shape(y)[0]
        if len_x != len_y:
            print('The arrays are different lengths')
            print(len_x)
            print(len_y)
                         
class ParamNumberException(Exception):
    def __init__(self,fit_num_params,guess):
        len_guess = np.shape(guess)[0]
        print('fit_num_params is ',fit_num_params)
        print('params guess has only ', len_guess)
        
        
if __name__ ==  '__main__':
    '''This contains all the unit tests'''
    xdata = np.arange(1000)
    
    '''sin or cos'''
    a = 10
    b = 0.5
    c = 0.5
    d = 5
    y_sin = sin_cos(xdata,a,b,c,d) + 2.0*np.random.rand(1000)
    
    pd_y_sin = pd.Series(y_sin,index=xdata)
    
    guess = [a+0.1,b*0.9,c+0*np.pi,d]
    lower_bcs = [None,0,0,'Fixed']
    upper_bcs = [None,np.inf,np.pi,'Fixed']
    fit_obj = Fit('sin_cos',x=xdata,y=y_sin)#series=pd_y_sin)
    fit_obj.add_params(guess,lower=lower_bcs,upper=upper_bcs)
    logic = xdata < 500
    fit_obj.add_filter(logic)
    fit_obj.stats(show_stats=True)
    fit_obj.fit()
    fit_obj.plot_fit(show=True)
    
    
    '''polynomials'''
    a = 1
    b = 2
    c = 3
    d = 4
    y_linear = linear(xdata,a,b)
    y_quadratic = quadratic(xdata,a,b,c)
    y_cubic = cubic(xdata,a,-b,-c,d)
    
    linear_fit = Fit('linear',x=xdata,y=y_linear)
    linear_fit.add_params([4,0])
    linear_fit.fit()
    linear_fit.plot_fit(show=True)
    quadratic_fit = Fit('quadratic',x=xdata,y=y_quadratic)
    quadratic_fit.add_params([3,2,1])
    quadratic_fit.fit()
    quadratic_fit.plot_fit(show=True)
    cubic_fit = Fit('cubic',x=xdata,y=y_cubic)
    cubic_fit.add_params([2,3,2,6])
    cubic_fit.fit()
    cubic_fit.plot_fit(show=True)
                
                
    
                         
    
    
    
                
                
                
    
                         
                         
 