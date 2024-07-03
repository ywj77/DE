import numpy as np
import matplotlib.pyplot as plt

NP = 100
size = 10
xMin = -10
xMax = 10
F = 0.5
CR = 0.8
G = 200


def fan(pop_v):
    res = np.sum(pop_v**2, axis=1)
    return res


def calfitness(pop_v):
    res = fan(pop_v)
    return res


def mutation(pop_v, f):
    m, n = pop_v.shape
    new_pop = np.zeros((m, n))
    for i in range(m):
        indices = np.arange(m)
        indices = np.delete(indices, i)
        r1, r2, r3 = pop_v[np.random.choice(indices, 3, replace=False)]
        new_pop[i] = r1 + f * (r2 - r3)
    return new_pop


def crossover(pop_v, pop_m, cr):
    m, n = pop_v.shape
    new_pop = pop_v.copy()
    for i in range(m):
        rn = np.random.randint(0, n, 1)
        rand = np.random.uniform(0, 1, (1, n))
        new_pop[i, rn] = pop_m[i, rn]
        condition = (rand <= cr)
        np.where(condition, pop_m, new_pop)
    return new_pop


def check_bound(pop_v):
    m, n = pop_v.shape
    for i in range(m):
        for j in range(n):
            condition = (pop_v[i, j] < xMin) or (pop_v[i, j] > xMax)
            if condition:
                pop_v[i, j] = xMin + np.random.uniform(0, 1, 1) * (xMax - xMin)
    return pop_v


def selection(pop_v, pop_c, fit):
    pop_c = check_bound(pop_c)
    fit_m = calfitness(pop_c)
    ind = fit_m < fit
    fit[ind] = fit_m[ind]
    pop_v[ind] = pop_c[ind]
    return pop_v, fit


pop = xMin + np.random.uniform(0, 1, (NP, size))*(xMax - xMin)
fitness = calfitness(pop)
best_archive = []
for gen in range(G):
    lamda = np.exp(1 - G/(G + 1 - gen))
    f = F*np.power(2, lamda)
    pop_mutation = mutation(pop, f)
    pop_crossover = crossover(pop, pop_mutation, CR)
    pop, fitness = selection(pop, pop_crossover, fitness)
    index = np.argsort(fitness.flatten())
    best = fitness[index[0]]
    best_archive.append(best)
    print(f"第{gen}代最优值", best)

plt.figure()
plot_x = np.arange(G)
plot_y = best_archive
plt.plot(plot_x, plot_y)
plt.show()
