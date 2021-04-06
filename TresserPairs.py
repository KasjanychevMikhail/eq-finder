import numpy as np
import systems_fun as sf
import SystOsscills as a4d
import plotFun as pf

f = open('./output_files/TresserPairs/LaunchParameters.txt', 'r')
d=eval(f.read())

# Задаем область поиска локальных минимумов для численного метода, а также более точную область поиска.

bounds = [(-0.1, +2 * np.pi + 0.1), (-0.1, +2 * np.pi + 0.1)]
bordersEq = [(-1e-15, +2 * np.pi + 1e-15), (-1e-15, +2 * np.pi + 1e-15)]

a = d['Parameters']['aval']
b = d['Parameters']['bval']

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


ud = [0.5, a, b, 1]
osc = a4d.FourBiharmonicPhaseOscillators(ud[0], ud[1], ud[2], ud[3])
pf.plotTresserPairs(osc, bounds, bordersEq, ps, './output_files/TresserPairs/', "testTresser")
