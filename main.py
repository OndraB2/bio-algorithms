from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm
import numpy as np
from Functions import Funcs
import random
import math

class Bia:
    def __init__(self):

        self.min = ()
        self.max = ()
        self.x, Y, Z = [], [], []
        self.last_best = ()
        self.point = ()
        self.scat = None
        self.func = None
        self.tempX = []
        self.tempY = []
        self.best_points = []
        self.generation_i = 0
        self.i = 1
        self.first_point = True
        self.sigma = 1
        self.xmin = 0
        self.xmax = 0
        self.ymin = 0
        self.ymax = 0
        self.step = 0

        self.T_0 = 100
        self.T_min = 0.5
        self.aplha = 0.95
        self.points_to_remove = []

        self.T = self.T_0

        self.generation_count = 50
        self.NP=20
        self.F=0.5
        self.CR=0.5
        self.dimension = 3
        self.pop = []
        self.paths = None
        self.name = ""


        self.pop_size = 15
        self.M_max = 50
        self.m = 0
        self.c1 = 2.0
        self.c2 = 2.0
        self.v_mini = -2
        self.v_maxi = 2
        self.swarm = []
        self.velocity = []
        self.pBest = []
        self.ws = 0.9
        self.we = 0.4
        self.w = 0
        self.gBest = ()

        # SOMA
        self.soma_pop_size=20
        self.PRT=0.4
        self.path_length=3.0
        self.soma_step = 0.11
        self.soma_M_max=10
        self.leader_i = 0
        self.leader = ()

        # firefly
        self.firefly_pop_size=20
        self.firefly_M_max=100
        self.leader_i = 0
        self.leader = ()

        # firefly
        self.NP=20
        self.teaching_G_max=100
        self.teacher_i = 0
        self.teacher = ()

        self.dimension = 30
        self.NP = 30

    def make_surface(self, func, xmin, xmax, ymin, ymax, step, sigma, v_mini, v_maxi):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.step = step
        self.sigma = sigma

        self.v_mini = v_mini
        self.v_maxi = v_maxi

        # self.X = np.arange(xmin, xmax, step)
        # self.Y = np.arange(ymin, ymax, step)

        # self.tempX = self.X
        # self.tempY = self.Y

        self.func = func["func"]
        self.name = func["name"]
        # self.Z = []

        # x = random.uniform(self.xmin, self.xmax)
        # y = random.uniform(self.ymin, self.ymax)
        # z = self.func([x, y])

        # self.last_best = (x, y, z)
        # self.best_points.append(self.last_best)

        # for x in self.X:
        #     temparr = []
        #     for y in self.Y:
        #         z = self.func([x, y])
        #         temparr.append(z)
        #     self.Z.append(np.array(temparr))
        # self.X, self.Y = np.meshgrid(self.X, self.Y)
        # self.Z = np.array (self.Z)

        # return (self.Y , self.X, self.Z)

    def normalizeCords(self, cords):
        res_cord = list(cords)
        for dim_i in range(self.dimension-1):
            if res_cord[dim_i] < self.xmin:
                res_cord[dim_i] = self.xmin
            if res_cord[dim_i] > self.xmax:
                res_cord[dim_i] = self.xmax\

        return res_cord

    def differentialEvolution(self):
        for i in range(self.NP):
            l_points = []
            for dim_i in range(self.dimension):
                l_points.append(random.uniform(self.xmin, self.xmax))
            self.pop.append(tuple(l_points))      

        while True:
            new_pop = self.pop.copy() # new generation
            for i, x in enumerate(self.pop): # x is also denoted as a target vector
                r = random.sample(range(self.NP), 3)
                while i not in r:
                    r = random.sample(range(self.NP), 3)

                temp = tuple(np.multiply(tuple(np.subtract(self.pop[r[0]], self.pop[r[1]])), self.F))
                v = tuple(np.add(temp, self.pop[r[2]])) # mutation vector. TAKE CARE FOR BOUNDARIES!   

                v = self.normalizeCords(v)

                u = np.zeros(self.dimension) # trial vector
                j_rnd = np.random.randint(0, self.dimension)

                for j in range(self.dimension):
                    if np.random.uniform() < self.CR or j == j_rnd:
                        u[j] = v[j] # at least 1 parameter should be from a mutation vector v
                    else:
                        u[j] = x[j]

                f_u, min_val, end = Funcs.functionWithCounter(self.func, [*u[:-1]])
                if end:
                    return min_val

                if f_u <= x[-1]: # We always accept a solution with the same fitness as a target vector

                    new_x = (*u[:-1], f_u)
                    new_pop[i] = new_x
                self.pop = new_pop

    def normalizeVelocity(self, vel):
        res_vel = list(vel)
        for dim_i in range(self.dimension-1):
            if abs(res_vel[dim_i]) > self.v_maxi:
                if res_vel[dim_i] < 0:
                    res_vel[dim_i] = self.v_maxi * -1
                else:
                    res_vel[dim_i] = self.v_maxi
            if abs(res_vel[dim_i]) < self.v_mini:
                if res_vel[dim_i] < 0:
                    res_vel[dim_i] = self.v_mini * -1
                else:
                    res_vel[dim_i] = self.v_mini  
        return tuple(res_vel)

    def particleSwarm(self):
        for i in range(self.NP):
            l_points = []
            for dim_i in range(self.dimension):
                l_points.append(random.uniform(self.xmin, self.xmax))

            f_u, min_val, end = Funcs.functionWithCounter(self.func, [*l_points[:-1]])
            if end:
                return min_val
            l_points[-1] = f_u

            self.swarm.append(tuple(l_points))  
            signs = []
            for _ in range(self.dimension):  # generovani znamenek pro velocity, jelikoz je generuju z abolutni hodnoty
                signs.append(-1 if np.random.uniform() > 0.5 else 1)

            l_vel = []
            for dim_i in range(self.dimension):
                l_vel.append(signs[dim_i] * random.uniform(self.v_mini, self.v_maxi))

            self.velocity.append(tuple(l_vel))
            self.pBest.append(self.swarm[i])
        
        min_i = 0
        min_z = self.swarm[0][-1]

        for i, x in enumerate(self.swarm):     
            if x[-1] < min_z:
                min_z = x[-1]
                min_i = i
        self.gBest = self.swarm[min_i]

        while True:

            for i, x in enumerate(self.swarm): 
                self.w = self.ws - ((self.ws - self.we) * i) / self.M_max
                r1 = np.random.uniform()
                temp_pBest_mul = tuple(np.multiply(r1 * self.c1, tuple(np.subtract(self.pBest[i], x))))
                temp_gBest_mul = tuple(np.multiply(r1 * self.c2, tuple(np.subtract(self.gBest, x))))
                temp_mul_add = tuple(np.add(temp_gBest_mul, temp_pBest_mul))
                v_new = tuple(np.add(tuple(np.multiply(self.velocity[i], self.w)), temp_mul_add))


                v_new = self.normalizeVelocity([*v_new])
                x_new = tuple(np.add(x, v_new))

                x_new = self.normalizeCords(x_new)

                f_u, min_val, end = Funcs.functionWithCounter(self.func, [*x_new[:-1]])
                if end:
                    return min_val
                temp_x_new = list(x_new)
                temp_x_new[-1] = f_u

                self.swarm[i] = tuple(temp_x_new)

                if self.swarm[i][-1] < self.pBest[i][-1]:
                    self.pBest[i] = self.swarm[i]
                    if self.pBest[i][-1] < self.gBest[-1]:
                        self.gBest = self.pBest[i]    

    def chooseBest(self, l: list):
        min_i = 0
        min_z = l[0][-1]

        for i, x in enumerate(l):
            if x[-1] < min_z:
                min_z = x[-1]
                min_i = i
        return min_i

    def SOMA(self):
        for i in range(self.NP):
            l_points = []
            for dim_i in range(self.dimension):
                l_points.append(random.uniform(self.xmin, self.xmax))

            f_u, min_val, end = Funcs.functionWithCounter(self.func, l_points[:-1])
            # print(f'eval: {f_u}')
            if end:
                return min_val
            l_points[-1] = f_u
            self.pop.append(tuple(l_points)) 

        new_pop = self.pop.copy()

        while True:
            self.leader_i = self.chooseBest(self.pop)
            self.leader = self.pop[self.leader_i]

            for i, x in enumerate(self.pop):
                if i != self.leader_i:
                    t = 0
                    x_path_points = []
                    while t <= self.path_length:
                        PRT_vector = ()
                        temp_vector = []
                        for _ in range(self.dimension):
                            temp_vector.append(1 if np.random.uniform() < self.PRT else 0)
                        PRT_vector = tuple(temp_vector)
                        temp_mul = tuple(np.multiply(t, PRT_vector))
                        temp_sub_and_mul = tuple(np.multiply(tuple(np.subtract(self.pop[self.leader_i], x)), temp_mul))
                        x_new = np.add(x, temp_sub_and_mul)

                        x_new = self.normalizeCords(x_new)

                        f_u, min_val, end = Funcs.functionWithCounter(self.func, x_new[:-1])
                        if end:
                            return min_val
                        temp_x_new = list(x_new)
                        temp_x_new[-1] = f_u

                        x_path_points.append(tuple(temp_x_new))

                        t += self.soma_step

                    temp_i = self.chooseBest(x_path_points)
                    best_point = x_path_points[temp_i] if x_path_points[temp_i][-1] <= x[-1] else x
                    new_pop[i] = best_point

    def distance(self, a, b):
        return np.linalg.norm(b-a)


    def firefly(self):
        alpha = 0.3
        beta_0 = 1 
        gama = 0.5

        for i in range(self.NP):
            l_points = []
            for dim_i in range(self.dimension):
                l_points.append(random.uniform(self.xmin, self.xmax))

            f_u, min_val, end = Funcs.functionWithCounter(self.func, [*l_points[:-1]])
            if end:
                return min_val
            l_points[-1] = f_u
            self.pop.append(tuple(l_points)) 

        while True:
            self.leader_i = self.chooseBest(self.pop)
            self.leader = self.pop[self.leader_i]

            for i in range(self.NP):
                for j in range(self.NP):
                    self.leader_i = self.chooseBest(self.pop)
                    self.leader = self.pop[self.leader_i]
                    if i != j and i != self.leader_i:
                        # print(self.pop[i][:-1], self.pop[j])
                        r = self.distance(np.array(self.pop[i][:-1]), np.array(self.pop[j][:-1]))
                        i_intesity = self.pop[i][-1] * np.e**(-gama * r)
                        j_intesity = self.pop[j][-1] * np.e**(-gama * r)
                        x_new = self.pop[i]

                        if j_intesity < i_intesity:
                            beta = beta_0 / (1 + r)
                            e = []
                            for k in range(self.dimension):
                                e.append(np.random.normal(0, 1))
                            beta_part = np.multiply(beta, np.subtract(self.pop[j], self.pop[i]))
                            alpha_part = np.multiply(alpha, tuple(e))
                            beta_plus_alpha_parts = np.add(beta_part, alpha_part)
                            self.pop[i] = tuple(np.add(self.pop[i], beta_plus_alpha_parts))
                            x_new = self.normalizeCords(self.pop[i])

                        f_u, min_val, end = Funcs.functionWithCounter(self.func, [*x_new[:-1]])
                        if end:
                            return min_val
                        temp_x_new = list(x_new)
                        temp_x_new[-1] = f_u

                        self.pop[i] = tuple(temp_x_new)
            
            # Best firefly
            e = []
            for k in range(self.dimension):
                e.append(np.random.normal(0, 1))
            self.leader = tuple(np.add(self.leader, np.add(alpha, tuple(e))))

            x_new = self.normalizeCords(self.leader)

            f_u, min_val, end = Funcs.functionWithCounter(self.func, [*x_new[:-1]])
            if end:
                return min_val
            temp_x_new = list(x_new)
            temp_x_new[-1] = f_u

            self.leader = tuple(temp_x_new)

            if self.leader[-1] < self.pop[self.leader_i][-1]:
                self.pop[self.leader_i] = self.leader

    # def calculateMean(self, population):
    #     sum = [0 for _ in range(self.dimension)]
    #     for i in range(self.dimension):
    #         for j in range(self.NP):
    #             sum[i] += population[j][i]

    #     for i in range(self.dimension):
    #         sum[i] /= self.NP

    #     return sum        

    def calculateMean(self, pop):
        _sum = np.zeros(self.dimension)
        for i in range(self.NP):
            for dim_i in range(self.dimension):
                _sum[dim_i] += self.pop[i][dim_i]
        for dim_i in range(self.dimension):
            _sum[dim_i] /= self.NP
            
        return tuple(_sum)

    # def teachingLearning(self):
    #     for i in range(self.NP):
    #         l_points = []
    #         for dim_i in range(self.dimension):
    #             l_points.append(random.uniform(self.xmin, self.xmax))

    #         f_u, min_val, end = Funcs.functionWithCounter(self.func, [*l_points[:-1]])
    #         if end:
    #             return min_val
    #         l_points[-1] = f_u
    #         self.pop.append(tuple(l_points)) 
    #         # print(l_points)

    #     while True:
    #         # teacher phase
    #         m = self.calculateMean(self.pop)

    #         self.teacher_i = self.chooseBest(self.pop)
    #         self.teacher = self.pop[self.teacher_i]

    #         tf = np.random.randint(1, 2)
    #         r = np.random.uniform()

    #         diff = np.multiply(r, np.subtract(self.teacher, np.multiply(tf, m)))

    #         teacher_new = tuple(np.add(self.teacher, diff))

    #         x_new = self.normalizeCords(teacher_new)

    #         f_u, min_val, end = Funcs.functionWithCounter(self.func, [*x_new[:-1]])
    #         if end:
    #             return min_val
    #         temp_x_new = list(x_new)
    #         temp_x_new[-1] = f_u

    #         teacher_new = tuple(temp_x_new)

    #         if teacher_new[-1] < self.teacher[-1]:
    #             self.teacher = teacher_new
    #             self.pop[self.teacher_i] = teacher_new
                
    #         # learner phase
    #         for i in range(self.NP):
    #             if i != self.teacher_i:
    #                 items = [x for x in range(self.NP) if x not in [i, self.teacher_i]]
    #                 j = random.choice(items) 
    #                 another_learner = self.pop[j]  
    #                 x_new = ()
    #                 if self.pop[i][-1] < another_learner[-1]:
    #                     x_new = np.add(self.pop[i], np.multiply(r, np.subtract(self.pop[i], another_learner)))
    #                 else:
    #                     x_new = np.add(self.pop[i], np.multiply(r, np.subtract(another_learner, self.pop[i])))  

    #                 x_new = self.normalizeCords(x_new)

    #                 f_u, min_val, end = Funcs.functionWithCounter(self.func, [*x_new[:-1]])
    #                 if end:
    #                     return min_val
    #                 temp_x_new = list(x_new)
    #                 temp_x_new[-1] = f_u

    #                 x_new = tuple(temp_x_new)

    #                 if x_new[-1] < self.pop[i][-1]:
    #                     self.pop[i] = x_new

    def teachingLearning(self):
        alpha = 0.3
        beta_0 = 1 
        gama = 0.5
    
        for i in range(self.NP):
            l_points = []
            for dim_i in range(self.dimension):
                l_points.append(random.uniform(self.xmin, self.xmax))
            f_u, min_val, end = Funcs.functionWithCounter(self.func, l_points[:-1])
            if end:
                return min_val
            l_points[-1] = f_u
            self.pop.append(tuple(l_points)) 

        while True:
            #teacher
            mean = self.calculateMean(self.pop)
            self.teacher_i = self.chooseBest(self.pop)
            self.teacher = self.pop[self.teacher_i]
            tf = random.randint(1,2)
            r = np.random.uniform()
            diff = tuple(np.multiply(r, np.subtract(self.teacher, np.multiply(tf, mean))))
            teacher_new = tuple(np.add(self.teacher, diff)) 
            l_points = self.normalizeCords(teacher_new)
            f_u, min_val, end = Funcs.functionWithCounter(self.func, l_points[:-1])
            if end:
                return min_val
            l_points[-1] = f_u
            
            teacher_new = tuple(l_points)
            if teacher_new[-1] < self.teacher[-1]:
                self.teacher = teacher_new
                self.pop[self.teacher_i] = teacher_new
            # learner
            for i in range(self.NP):
                if i != self.teacher_i:
                    learner_j = random.choice([x for x in range(self.NP) if x != i and x != self.teacher_i])
                    x_new = ()
                    if self.pop[i][-1] < self.pop[learner_j][-1]:
                        x_new = np.add(self.pop[i], np.multiply(r, np.subtract(self.pop[i], self.pop[learner_j])))
                    else:
                        x_new = np.add(self.pop[i], np.multiply(r, np.subtract(self.pop[learner_j], self.pop[i])))
                    
                    l_points = self.normalizeCords(x_new)
                    f_u, min_val, end = Funcs.functionWithCounter(self.func, l_points[:-1])
                    if end:
                        return min_val
                    l_points[-1] = f_u
                    
                    x_new = tuple(l_points)
                    if x_new[-1] < self.pop[i][-1]:
                        self.pop[i] = x_new 


functs = [
    { "name" : "Sphere", "func" : Funcs.sphere, "xmin" : -10, "xmax" : 10, "ymin" : -10, "ymax" : 10,  "step" : 0.4, "sigma" : 1, "v_mini" : 0.1, "v_maxi" : 0.5},
    { "name" : "Ackley", "func" : Funcs.ackley, "xmin" : -40, "xmax" : 40, "ymin" : -40, "ymax" : 40,  "step" : 0.5, "sigma" : 3, "v_mini" : 0.5, "v_maxi" : 2},
    { "name" : "Rastrigin", "func" : Funcs.rastrigin, "xmin" : -5, "xmax" : 5, "ymin" : -5, "ymax" : 5,  "step" : 0.4, "sigma" : 0.5, "v_mini" : 0.05, "v_maxi" : 0.25},
    { "name" : "Rosenbrock", "func" : Funcs.rosenbrock, "xmin" : -10, "xmax" : 10, "ymin" : -10, "ymax" : 10,  "step" : 0.4, "sigma" : 1, "v_mini" : 0.1, "v_maxi" : 0.5},
    { "name" : "Griewank", "func" : Funcs.griewank, "xmin" : -50, "xmax" : 50, "ymin" : -50, "ymax" : 50, "step" : 1, "sigma" : 5, "v_mini" : 0.5, "v_maxi" : 2},
    { "name" : "Schwefel", "func" : Funcs.schwefel, "xmin" : -500, "xmax" : 500, "ymin" : -500, "ymax" : 500, "step" : 5, "sigma" : 10, "v_mini" : 5, "v_maxi" : 20},
    { "name" : "Levy", "func" : Funcs.levy, "xmin" : -10, "xmax" : 10, "ymin" : -10, "ymax" : 10,  "step" : 0.4, "sigma" : 1, "v_mini" : 0.1, "v_maxi" : 0.5},
    { "name" : "Michalewicz", "func" : Funcs.michalewicz, "xmin" : 0, "xmax" : 4, "ymin" : 0, "ymax" : 4, "step" : 0.1, "sigma" : 0.2, "v_mini" : 0.5, "v_maxi" : 0.2},
    { "name" : "Zakharov", "func" : Funcs.zakharov, "xmin" : -10, "xmax" : 10, "ymin" : -10, "ymax" : 10,  "step" : 0.4, "sigma" : 1, "v_mini" : 0.1, "v_maxi" : 0.5},
]
for f in functs:
    print(f'{f["name"]}')
    for i in range(30):
        bia = Bia()
        bia.make_surface(func=f, xmin=f['xmin'], xmax=f['xmax'], ymin=f['ymin'], ymax=f['ymax'], step=f["step"], sigma=f["sigma"], v_mini=f["v_mini"], v_maxi=f["v_maxi"])
        Funcs.initValues()
        de_best = bia.differentialEvolution()
        Funcs.initValues()
        pso_best = bia.particleSwarm()
        Funcs.initValues()
        soma_best = bia.SOMA()
        Funcs.initValues()
        fa_best = bia.firefly()
        Funcs.initValues()
        tlbo_best = bia.teachingLearning()
        # print(f'{round(de_best, 4)},{round(pso_best, 4)},{round(soma_best, 4)},{round(fa_best, 4)},{round(tlbo_best, 4)}')
        print(f'{de_best},{pso_best},{soma_best},{fa_best},{tlbo_best}')
    print('\n\n')

