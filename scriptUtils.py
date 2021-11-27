import numpy as np
import systems_fun as sf

def getGridR(dictConfig):
    N = dictConfig['Parameters']['r_N']  # Количество разбиений параметра r

    rs = np.linspace(dictConfig['Parameters']['r_min'], dictConfig['Parameters']['r_max'], N)

    a = dictConfig['Parameters']['aval']
    b = dictConfig['Parameters']['bval']
    return (N, a, b, rs)

def getGrid(dictConfig):
    N = dictConfig['Parameters']['x1_N']  # Количество разбиений параметра X1
    M = dictConfig['Parameters']['x2_N']  # Количество разбиений параметра X2

    alphas = np.linspace(dictConfig['Parameters']['x1_min'], dictConfig['Parameters']['x1_max'], N)
    betas = np.linspace(dictConfig['Parameters']['x2_min'], dictConfig['Parameters']['x2_max'], M)

    r = dictConfig['Parameters']['x4val']
    return ( N, M, alphas, betas, r)

def getParamsSHGO(dictConfig):

    numOfSamp = dictConfig['ParametersShgoEqFinder']['numOfSamp']
    numOfIters = dictConfig['ParametersShgoEqFinder']['numOfIters']
    valToCompareWith = dictConfig['ParametersShgoEqFinder']['compareWith']

    return(numOfSamp, numOfIters, valToCompareWith)

def getPrecisionSettings(dictConfig):
    ps = sf.PrecisionSettings(zeroImagPartEps=dictConfig['NumericTolerance']['zeroImagPartEps'],
                              zeroRealPartEps=dictConfig['NumericTolerance']['zeroRealPartEps'],
                              clustDistThreshold=dictConfig['NumericTolerance']['clustDistThreshold'],
                              separatrixShift=dictConfig['SeparatrixComputing']['separatrixShift'],
                              separatrix_rTol=dictConfig['SeparatrixComputing']['separatrix_rTol'],
                              separatrix_aTol=dictConfig['SeparatrixComputing']['separatrix_aTol'],
                              marginBorder=dictConfig['NumericTolerance']['marginBorder']
                              )
    return ps

def getProximitySettings(dictConfig):
    prox = sf.ProximitySettings(toSinkPrxtyEv = dictConfig['ConnectionProximity']['toSinkPrxtyEv'],
                                toSddlPrxtyEv = dictConfig['ConnectionProximity']['toSddlPrxtyEv'],
                                toTargetSinkPrxtyEv=dictConfig['ConnectionProximity']['toTargetSinkPrxtyEv'],
                                toTargetSddlPrxtyEv=dictConfig['ConnectionProximity']['toTargetSddlPrxtyEv'],
                                toSinkPrxty = dictConfig['ConnectionProximity']['toSinkPrxty'],
                                toSddlPrxty = dictConfig['ConnectionProximity']['toSddlPrxty'])
    return prox