import systems_fun as sf
import SystOsscills as sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

#Задаем область поиска локальных минимумов для численного метода, а также более точную область поиска.

bounds = [(-0.1,+2*np.pi+0.1), (-0.1,+2*np.pi+0.1)]
bordersEq = [(-1e-15,+2*np.pi+1e-15), (-1e-15,+2*np.pi+1e-15)]

# Разбиваем значения параметров на сетку

N = 3 # Количество разбиений параметра альфа
M = 3 # Количество разбиений параметра бета

alphas = np.linspace(0, 2 * np.pi, N)
betas = np.linspace(0,2*np.pi,M)

ps = sf.STD_PRECISION

#Строим карту расстояний
colorGridHeterIndMap = np.zeros((M, N))
colorGridDistMap = np.zeros((M, N))
for i,a in enumerate(alphas):
    for j,b in enumerate(betas):
        # Устанавливаем значения параметров системы
        ud = [0.5,a,b,1]
        rhs=sys.FourBiharmonicPhaseOscillators(ud[0],ud[1],ud[2],ud[3])
        # Находим состояния равновесия
        eqList = sf.findEquilibria(rhs.getRestriction,rhs.getRestrictionJac,bounds,bordersEq,sf.ShgoEqFinder(300,30,1e-10), ps)
        # Из всех состояний равновесия отбираем нужные нам конфигурации
        gfe = sf.getTresserPairs(eqList, rhs, ps)
        if (len(gfe)>0 ):
            colorGridHeterIndMap[j][i] = 1
            for SfSd in gfe:
                dist = sf.heterCheck(SfSd[0],SfSd[1],rhs,1000, ps) # Находим расстояние сепаратрисы седло-фокуса до седла
                if (dist<1e-5):
                    colorGridDistMap[j][i] = 1 # Если расстояние достаточно мало, то фиксируем это

plt.pcolormesh(alphas, betas, colorGridDistMap, cmap=plt.cm.get_cmap('RdBu'))
plt.colorbar()
plt.savefig('DistanceMap11x11')

plt.pcolormesh(alphas, betas, colorGridHeterIndMap, cmap=plt.cm.get_cmap('RdBu'))
plt.colorbar()
plt.savefig('HeterIndMap11x11')

# Нужная пара состояний равновесия на плоскости
ud = [0.5,1,1,1]
rhs=sys.FourBiharmonicPhaseOscillators(ud[0],ud[1],ud[2],ud[3])
eqList = sf.findEquilibria(rhs.getRestriction,rhs.getRestrictionJac,bounds,bordersEq,sf.ShgoEqFinder(300,30,1e-10), ps)
gfe = sf.getTresserPairs(eqList, rhs, ps)

xs = ys = np.linspace(0, +2 * np.pi, 1001)
res = np.zeros([len(xs), len(xs)])
for i, y in enumerate(ys):
    for j, x in enumerate(xs):
        res[i][j] = np.log10(np.dot(rhs.getRestriction([x, y]), rhs.getRestriction([x, y])) + 1e-10)

matplotlib.rcParams['figure.figsize'] = 10, 10

plt.pcolormesh(xs, ys, res, cmap=plt.cm.get_cmap('RdBu'))
plt.xlim([0, +2 * np.pi])
plt.ylim([0, +2 * np.pi])
plt.xlabel('$\gamma_3$')
plt.ylabel('$\gamma_4$')
plt.axes().set_aspect('equal', adjustable='box')
plt.scatter(gfe[0][0].coordinates[1], gfe[0][0].coordinates[2], c='green', s=40)  # sad foc
plt.scatter(gfe[0][1][0].coordinates[1], gfe[0][1][0].coordinates[2], c='red', s=40)  # saddle

ValParams = [[1.6929693744345, 1.2391837689159737],
 [1.6929693744345, 1.256637061435917],
 [1.6929693744345, 1.3089969389957468],
 [1.6929693744345, 1.32645023151569],
 [1.6929693744345, 1.3439035240356334]
]

for i,params in enumerate(ValParams):
    a,b = params
    ud = [0.5,a,b,1]
    rhs=sys.FourBiharmonicPhaseOscillators(ud[0],ud[1],ud[2],ud[3])
    eqList = sf.findEquilibria(rhs.getRestriction,rhs.getRestrictionJac,bounds,bordersEq,sf.ShgoEqFinder(300,30,1e-10), ps)
    gfe = sf.getTresserPairs(eqList, rhs, ps)
    if (len(gfe)>0 ):
        for SfSd in gfe:
            dist = sf.heterCheck(SfSd[0],SfSd[1],rhs,1000, ps)
            if (dist<1e-5):
                pair = SfSd
    sep1 = sf.computeSeparatrices(pair[0], rhs, sf.isInCIR, ps, 1000)
    x,y,z = zip(*sep1)
    fig, axs = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(30,10))

    axs[0].scatter(x[0], y[0], s=40,c='g', label='Начало')
    axs[0].scatter(x[-1], y[-1], s=40,c='r', label='Конец')
    axs[0].set_xlim(0,2*np.pi)
    axs[0].set_ylim(0,2*np.pi)
    axs[0].plot(x, y)
    axs[0].set_xlabel(r'$\phi_1$')
    axs[0].set_ylabel(r'$\phi_2$')

    axs[1].scatter(x[0], z[0], s=40,c='g', label='Начало')
    axs[1].scatter(x[-1], z[-1], s=40,c='r', label='Конец')
    axs[1].set_xlim(0,2*np.pi)
    axs[1].set_ylim(0,2*np.pi)
    axs[1].plot(x, z)
    axs[1].set_xlabel(r'$\phi_1$')
    axs[1].set_ylabel(r'$\phi_3$')

    axs[2].scatter(y[0], z[0], s=40,c='g', label='Начало')
    axs[2].scatter(y[-1], z[-1], s=40,c='r', label='Конец')
    axs[2].set_xlim(0,2*np.pi)
    axs[2].set_ylim(0,2*np.pi)
    axs[2].plot(y, z)
    axs[2].set_xlabel(r'$\phi_2$')
    axs[2].set_ylabel(r'$\phi_3$')
    plt.savefig('TrajProec_{}.pdf'.format(i))