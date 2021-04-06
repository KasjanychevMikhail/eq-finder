import multiprocessing as mp
import numpy as np
import systems_fun as sf
import SystOsscills as a4d
import findTHeteroclinic as fth
import itertools as itls
import time
import plotFun as pf

f = open('./output_files/TrajProec/LaunchParameters.txt', 'r')
d=eval(f.read())

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

DtFile = "{}{}.txt".format(d['DataFile']['PathToFile'],d['DataFile']['NameOfFile'])
pars = np.loadtxt(DtFile, usecols=(2,3))
startPts = np.loadtxt(DtFile, usecols=(5,6,7))

def workerChectTargHet(params):
    (i, (a, b)), startPt = params
    ud = [0.5, a, b, 1]
    osc = a4d.FourBiharmonicPhaseOscillators(ud[0], ud[1], ud[2], ud[3])
    pf.plotTrajProec(osc.getReducedSystem, startPt, ps, 1000, './output_files/TrajProec/', "TrajProec_{}_{}".format(i,d['DataFile']['NameOfFile']),a,b)

if __name__ == "__main__":
    pool = mp.Pool(mp.cpu_count())
    start = time.time()
    ret = pool.map(workerChectTargHet, itls.product(enumerate(pars), startPts))
    end = time.time()
    pool.close()