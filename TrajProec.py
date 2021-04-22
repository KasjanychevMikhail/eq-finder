import multiprocessing as mp
import os.path

import numpy as np
import systems_fun as sf
import SystOsscills as a4d
import itertools as itls
import time
import plotFun as pf
import sys
import scriptUtils as su
import datetime

from functools import partial

def workerPlotTrajProec(params, pset: sf.PrecisionSettings, pathToDir, nameOfFile):
    (i, (a, b)), startPt = params
    ud = [0.5, a, b, 1]
    osc = a4d.FourBiharmonicPhaseOscillators(ud[0], ud[1], ud[2], ud[3])
    pf.plotTrajProec(osc.getReducedSystem, startPt, pset, 1000, pathToDir, "{}_{}".format(i, nameOfFile),a,b)

if __name__ == "__main__":
    if '-h' in sys.argv or '--help' in sys.argv:
        print("Usage: python TargetHeteroclinic.py <pathToConfig> <outputMask> <outputDir>"
              "\n    pathToConfig: full path to configuration file (e.g., \"./cfg.txt\")"
              "\n    outputMask: unique name that will be used for saving output"
              "\n    outputDir: directory to which the results are saved")
        sys.exit()
    assert os.path.isfile(sys.argv[1]), "Configuration file does not exist!"
    assert os.path.isfile(sys.argv[2]), "File does not exist!"
    assert os.path.isdir(sys.argv[4]), "Output directory does not exist!"

    configFile = open("{}".format(sys.argv[1]), 'r')
    configDict = eval(configFile.read())

    timeOfRun = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')

    ps = su.getPrecisionSettings(configDict)

    fullPathToDataFile = sys.argv[2]
    nameOutputFile = sys.argv[3]
    pathToOutputDir = sys.argv[4]

    outputFileMask = "{}_{}".format(nameOutputFile, timeOfRun)

    pars = np.loadtxt(fullPathToDataFile, usecols=(2, 3))
    startPts = np.loadtxt(fullPathToDataFile, usecols=(5, 6, 7))
    pool = mp.Pool(mp.cpu_count())
    start = time.time()
    ret = pool.map(partial(workerPlotTrajProec, pset = ps, pathToDir = pathToOutputDir, nameOfFile = outputFileMask), itls.product(enumerate(pars), startPts))
    end = time.time()

    pool.close()