import numpy as np
import matplotlib.pyplot as plt


class DE(object):
    def __init__(self, param_de):
        self.NP = param_de['NP']
        self.size = param_de['size']
        self.xMin = param_de['xMin']
        self.xMax = param_de['xMax']
        self.F = param_de['F']
        self.CR = param_de['CR']
        self.pop_mut = None
        self.pop_cro = None
        self.best_archive = []

    def pop_init(self):
        pop_de = self.xMin + np.random.uniform(0, 1, (self.NP, self.size)) * (self.xMin - self.xMax)
        return pop_de

    @staticmethod
    def calfitness(pop_de):
        res = fan(pop_de)
        return res

    def mutation(self, pop_de, f_de):
        self.pop_mut = np.zeros((self.NP, self.size))
        for i in range(self.NP):
            index = np.arange(self.NP)
            index = np.delete(index, i)
            r1, r2, r3 = pop_de[np.random.choice(index, 3, replace=False)]
            self.pop_mut[i] = r1 + f_de * (r2 - r3)
        return self.pop_mut

    def crossover(self, pop_de):
        self.pop_cro = pop_de.copy()
        for i in range(self.NP):
            rn = np.random.randint(0, self.size, 1)
            rand = np.random.uniform(0, 1, (1, self.size))
            self.pop_cro[i, rn] = self.pop_mut[i, rn]
            condition = (rand <= self.CR)
            np.where(condition, self.pop_mut, self.pop_cro)
        return self.pop_cro

    def check_bound(self):
        for i in range(self.NP):
            for j in range(self.size):
                condition = (self.pop_cro[i, j] < self.xMin) or (self.pop_cro[i, j] > self.xMax)
                if condition:
                    self.pop_cro[i, j] = self.xMin + np.random.uniform(0, 1, 1) * (self.xMax - self.xMin)
        return self.pop_cro

    def selection(self, pop_de, fit):
        pop_c = self.check_bound()
        fit_c = self.calfitness(self.pop_cro)
        condition = fit_c < fit
        fit[condition] = fit_c[condition]
        pop_de[condition] = pop_c[condition]
        index = np.argsort(fit.flatten())
        best_de = fit[index[0]]
        self.best_archive.append(best_de)
        return pop_de, fit, best_de


def fan(pop_f):
    res = np.sum(pop_f**2, axis=1)
    return res


if __name__ == '__main__':
    param = {
        'NP': 100,
        'size': 10,
        'xMin': -10,
        'xMax': 10,
        'F': 0.5,
        'CR': 0.8,
        'G': 200
    }
    G = param['G']
    F = param['F']
    de = DE(param)
    pop = de.pop_init()
    fitness = de.calfitness(pop)
    for gen in range(G):
        lamda = np.exp(1 - G / (G + 1 - gen))
        f = F * np.power(2, lamda)
        de.mutation(pop, f)
        de.crossover(pop)
        pop, fitness, best = de.selection(pop, fitness)
        print(f"第{gen}代最优值", best)
    plt.figure()
    plot_x = np.arange(G)
    plot_y = de.best_archive
    plt.plot(plot_x, plot_y)
    plt.show()
