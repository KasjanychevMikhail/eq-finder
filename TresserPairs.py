import numpy as np
import SystOsscills as a4d
import plotFun as pf
import scriptUtils as su
import sys

sys.argv
f = open("{}{}.txt".format(sys.argv[1],sys.argv[2]), 'r')
d = eval(f.read())
# Задаем область поиска локальных минимумов для численного метода, а также более точную область поиска.

bounds = [(-0.1, +2 * np.pi + 0.1), (-0.1, +2 * np.pi + 0.1)]
bordersEq = [(-1e-15, +2 * np.pi + 1e-15), (-1e-15, +2 * np.pi + 1e-15)]

a = d['Parameters']['aval']
b = d['Parameters']['bval']
ps = su.getPrecisionSettings(d)

ud = [0.5, a, b, 1]
osc = a4d.FourBiharmonicPhaseOscillators(ud[0], ud[1], ud[2], ud[3])
pf.plotTresserPairs(osc, bounds, bordersEq, ps, './output_files/TresserPairs/', "testTresser")
