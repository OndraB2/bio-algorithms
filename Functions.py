from math import cos, sin, pi, exp, sqrt, e
import numpy as np
class Funcs:
    min_val = 99999999999999999
    counter = 0
    max_calls = 3030
    dimensions = 30

    @staticmethod
    def initValues():
        Funcs.min_val = 9999999999999999
        Funcs.counter = 0    

    @staticmethod
    def functionWithCounter(func, params):
        Funcs.counter += 1
        end = False
        if Funcs.counter > Funcs.max_calls:
            end = True
        result = func(params)

        if result < Funcs.min_val:
            Funcs.min_val = result
        return result, Funcs.min_val, end
    
    
    @staticmethod
    def sphere(params = []):
        sum = 0

        for item in params:
            sum += item**2

        # result = (1 / 899) * (sum)
        result = sum
        return result

    # @staticmethod
    # def ackley(params = []):
    #     sum = 0
    #     sum_cos = 0

    #     for item in params:
    #         sum += item**2
    #         sum_cos += cos(2 * pi * item)

    #     return -20.0 * exp(-0.2 * sqrt(0.5 * sum))-exp(0.5 * sum_cos) + e + 20

    @staticmethod
    def ackley(d = []):
        res = 0
        res2 = 0
        i = 0
        a = 20
        b = 0.2
        c = 2 * np.pi
        for val in d:
            i+=1
            res += pow(val,2)
            res2 += np.cos(c * val)
        return -a * np.exp(-b * np.sqrt((1/(len(d)+1)) * res)) - np.exp((1/(len(d)+1)) * res2) + a + np.exp(1)

    @staticmethod
    def rastrigin(params = []):
        return 10 * Funcs.dimensions + sum([(x**2 - 10 * cos(2 * pi * x)) for x in params])

    @staticmethod
    def rosenbrock(params = []):
        sum = 0
        for i in range(Funcs.dimensions-2):
            sum += 100 * ( params[i+1] - params[i]**2 )**2 + ( params[i] - 1 )**2
        return sum

    @staticmethod
    def griewank(params = []):
        sum = 0
        product = 0

        for i, item in enumerate(params):
            sum += (item**2) / 4000
            product *= cos(item / sqrt(i+1)) + 1

        return sum - product

    @staticmethod
    def schwefel(params = []):
        sum = 0

        for item in params:
            sum += item * sin(sqrt(abs(item)))

        return 418.9829 * Funcs.dimensions - sum

    # @staticmethod
    # def levy(params = []):
    #     sum = 0
    #     w1 =  1 + (params[0] - 1) / 4 
    #     wd = 1 + (params[ Funcs.dimensions - 2 ] - 1) / 4 
    #     for item in params:
    #         w = 1 + (item - 1) / 4 
    #         sum += (item * -1)**2 * (1 + 10 * sin(pi * w + 1)**2) + (wd - 1)**2 * (1 + sin(2 * pi * wd)**2)

    #     return sin(pi * w1)**2 + sum


    def levyW(x):
        return 1 + ((x - 1)/4)

    @staticmethod
    def levy(d = []):
        res = 0
        for i in range(0, len(d) - 1):
            res += pow(Funcs.levyW(d[i]) - 1, 2) * (1+ 10*pow(np.sin(np.pi * Funcs.levyW(d[i] + 1)),2)) + pow((Funcs.levyW(d[-1]) - 1),2) * (1+pow(np.sin(2*np.pi*Funcs.levyW(d[-1])), 2))
        return pow(np.sin(np.pi * Funcs.levyW(d[0])),2) + res

    @staticmethod
    def michalewicz(params = []):
        sum = 0
        m = 10
        for i, item in enumerate(params):
            sum += sin(item) * sin((i+1) * item**2 / pi)**(2 * m)

        return - sum

    @staticmethod
    def zakharov(params = []):
        sum1 = 0
        sum2 = 0

        for i, item in enumerate(params):
            sum1 += item**2
            sum2 += 0.5 * (i+1) * item

        return sum1 + sum2**2 + sum2**4