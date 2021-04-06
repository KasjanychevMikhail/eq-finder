import multiprocessing as mp
import numpy as np
import systems_fun as sf
import SystOsscills as a4d
import findTHeteroclinic as fth
import itertools as itls
import time
import plotFun as pf

f = open('./output_files/HeteroclinicFiles/LaunchParameters.txt', 'r')
d=eval(f.read())

# Задаем область поиска локальных минимумов для численного метода, а также более точную область поиска.

bounds = [(-0.1, +2 * np.pi + 0.1), (-0.1, +2 * np.pi + 0.1)]
bordersEq = [(-1e-15, +2 * np.pi + 1e-15), (-1e-15, +2 * np.pi + 1e-15)]

# Разбиваем значения параметров на сетку

N = d['Parameters']['a_N']  # Количество разбиений параметра альфа
M = d['Parameters']['b_N']  # Количество разбиений параметра бета

alphas = np.linspace(d['Parameters']['a_min'], d['Parameters']['a_max'], N)
betas = np.linspace(d['Parameters']['b_min'], d['Parameters']['b_max'], M)

ps = sf.PrecisionSettings(zeroImagPartEps=d['NumericTolerance']['zeroImagPartEps'],
                                  zeroRealPartEps=d['NumericTolerance']['zeroRealPartEps'],
                                  clustDistThreshold=d['NumericTolerance']['clustDistThreshold'],
                                  separatrixShift=d['SeparatrixComputing']['separatrixShift'],
                                  separatrix_rTol=d['SeparatrixComputing']['separatrix_rTol'],
                                  separatrix_aTol=d['SeparatrixComputing']['separatrix_aTol'],
                                  sdlSinkPrxty=d['ConnectionProximity']['sdlSinkPrxty'],
                                  sfocSddlPrxty=d['ConnectionProximity']['sfocSddlPrxty'],
                                  marginBorder=d['NumericTolerance']['marginBorder']
                                  )

def workerChectTargHet(params):
    (i, a), (j, b) = params
    ud = [0.5, a, b, 1]
    osc = a4d.FourBiharmonicPhaseOscillators(ud[0], ud[1], ud[2], ud[3])
    eqf = sf.ShgoEqFinder(300, 30, 1e-10)
    result = fth.checkTargetHeteroclinic(osc, bordersEq, bounds, eqf, ps, 1000.)
    return i, j, a, b, result

if __name__ == "__main__":
    pool = mp.Pool(mp.cpu_count())
    start = time.time()
    ret = pool.map(workerChectTargHet, itls.product(enumerate(alphas), enumerate(betas)))
    end = time.time()
    pool.close()
    pf.plotHeteroclinicsData(pf.prepareHeteroclinicsData(ret), alphas, betas, './output_files/HeteroclinicFiles/'
                             , "Heteroclinic{}x{}".format(N,M))
    pf.saveHeteroclinicsDataAsTxt(pf.prepareHeteroclinicsData(ret),'./output_files/HeteroclinicFiles/'
                                  , "Heteroclinic{}x{}".format(N,M))
