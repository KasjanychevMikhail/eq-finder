import os
import subprocess
from collections import namedtuple
import matplotlib.pyplot as plt
from scipy import optimize
from numpy import linalg as LA
import numpy as np
import scipy
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import AgglomerativeClustering

MapParameters = namedtuple('MapParameters', ['rhs', 'rhsJac', 'valueFirstParam',
                                             'valueSecondParam', 'constParam', 'bounds', 'optMethodParams',
                                             'bordersEq'])


class EnvironmentParameters:
    """
    Class that stores information about where
    to get source, where to put compiled version
    and where to save output files
    """

    def __init__(self, pathToOutputDirectory, outputStamp, imageStamp):
        assert (os.path.isdir(pathToOutputDirectory)), 'Output directory does not exist!'
        self.pathToOutputDirectory = os.path.join(os.path.normpath(pathToOutputDirectory), '')
        self.outputStamp = outputStamp
        self.imageStamp = imageStamp

    @property
    def clearAllInOutputDirectory(self):
        return os.path.join(self.pathToOutputDirectory, '*')

    @property
    def fullExecName(self):
        return os.path.join(self.pathToOutputDirectory, self.outputExecutableName)


def prepareEnvironment(envParams):
    """
    Clears output directory and copies executable
    """

    assert isinstance(envParams, EnvironmentParameters)
    clearOutputCommand = 'rm {env.clearAllInOutputDirectory}'.format(env=envParams)
    subprocess.call(clearOutputCommand, shell=True)


def isComplex(Z):
    return (abs(Z.imag) > 1e-14)


def describeEqType(eigvals):
    eigvalsS = eigvals[np.real(eigvals) < -1e-14]
    eigvalsU = eigvals[np.real(eigvals) > +1e-14]
    nS = len(eigvalsS)
    nU = len(eigvalsU)
    nC = len(eigvals) - nS - nU
    issc = 1 if nS > 0 and isComplex(eigvalsS[-1]) else 0
    isuc = 1 if nU > 0 and isComplex(eigvalsU[0]) else 0
    return (nS, nC, nU, issc, isuc)


def describePortrType(dataEqSignatures):
    phSpaceDim = int(sum(dataEqSignatures[0]))
    eqTypes = {(i, phSpaceDim-i):0 for i in range(phSpaceDim+1)}
    nonRough = 0
    for eqSign in dataEqSignatures:
        nS, nC, nU = eqSign
        if nC == 0:
            eqTypes[(nU, nS)] += 1
        else:
            nonRough += 1
    # nSinksn, nSources, nSaddles,  nNonRough
    portrType = tuple([eqTypes[(i, phSpaceDim-i)] for i in range(phSpaceDim+1)] + [nonRough])
    return portrType

class ShgoEqFinder:
    def __init__(self, nSamples, nIters):
        self.nSamples = nSamples
        self.nIters = nIters
    def __call__(self, rhs, rhsSq, rhsJac, boundaries, borders):
        optResult = scipy.optimize.shgo(rhsSq, boundaries, n=self.nSamples, iters=self.nIters, sampling_method='sobol');
        allEquilibria = [x for x, val in zip(optResult.xl, optResult.funl) if
                         abs(val) < 1e-15 and inBounds(x, borders)];
        return allEquilibria

class NewtonEqFinder:
    def __init__(self, xGridSize, yGridSize):
        self.xGridSize = xGridSize
        self.yGridSize = yGridSize
    def __call__(self, rhs, rhsSq, rhsJac, boundaries, borders):
        Result = []
        for i, x in enumerate(np.linspace(boundaries[0][0], boundaries[0][1], self.xGridSize)):
            for j, y in enumerate(np.linspace(boundaries[1][0], boundaries[1][1], self.yGridSize)):
                Result.append(optimize.root(rhs, [x, y], method='broyden1', jac=rhsJac).x)
        allEquilibria = [x for x in Result if abs(rhsSq(x)) < 1e-15 and inBounds(x, borders)];
        return allEquilibria

def createEqList (allEquilibria, rhsJac, ud):
    allEquilibria = sorted(allEquilibria, key=lambda ar: tuple(ar))
    result = np.zeros([len(allEquilibria), 9])
    for k, eqCoords in enumerate(allEquilibria):
        eqJacMatrix = rhsJac(eqCoords, ud)
        eigvals, _ = LA.eig(eqJacMatrix)
        eqTypeData = describeEqType(eigvals)
        eigvals = sorted(eigvals, key=lambda eigvals: eigvals.real)
        result[k] = list(eqCoords) + list(eqTypeData) + list(eigvals)
    return result

def findEquilibria(rhs, rhsJac, boundaries, ud, borders, method):
    def rhsSq(x):
        xArr = np.array(x)
        vec = rhs(xArr)
        return np.dot(vec, vec)

    allEquilibria = method(rhs, rhsSq, rhsJac, boundaries, borders)

    return createEqList(allEquilibria,rhsJac, ud)


def inBounds(X, boundaries):
    x, y = X
    Xb, Yb = boundaries
    b1, b2 = Xb
    b3, b4 = Yb
    return ((x > b1) and (x < b2) and (y > b3) and (y < b4))


def createFileTopologStructPhasePort(envParams, mapParams, i, j):
    ud = [mapParams.valueFirstParam, mapParams.valueSecondParam] + mapParams.constParam;
    headerStr = ('gamma = {par[0]}\n' +
                 'd = {par[1]}\n' +
                 'X  Y  nS  nC  nU  isSComplex  isUComplex  eigval1  eigval2\n' +
                 '0  1  2   3   4   5           6           7        8').format(par=ud)
    fmtList = ['%+18.15f',
               '%+18.15f',
               '%2u',
               '%2u',
               '%2u',
               '%2u',
               '%2u',
               '%+18.15f',
               '%+18.15f', ]
    rhsCurrent = lambda X: mapParams.rhs(X, ud)
    sol = findEquilibria(rhsCurrent, mapParams.rhsJac, mapParams.bounds, ud, mapParams.bordersEq, mapParams.method,
                         mapParams.optMethodParams)
    X = sol[:, 0:2]
    if list(X) and len(list(X)) != 1:
        clustering = AgglomerativeClustering(n_clusters=None, affinity='euclidean', linkage='single',
                                             distance_threshold=(5 * 1e-4))
        clustering.fit(X)
        data = sol[:, 2:5]
        trueStr = mergePoints(clustering.labels_, data)
        np.savetxt("{env.pathToOutputDirectory}{:0>5}_{:0>5}.txt".format(i, j, env=envParams), sol[list(trueStr), :],
                   header=headerStr,fmt=fmtList)
    else:
        np.savetxt("{env.pathToOutputDirectory}{:0>5}_{:0>5}.txt".format(i, j, env=envParams), sol, header=headerStr,fmt=fmtList)


def createBifurcationDiag(envParams, numberValuesParam1, numberValuesParam2, arrFirstParam, arrSecondParam):
    N, M = numberValuesParam1, numberValuesParam2
    colorGrid = np.zeros((M, N)) * np.NaN
    diffTypes = {}
    curTypeNumber = 0
    for i in range(M):
        for j in range(N):
            data = np.loadtxt('{}{:0>5}_{:0>5}.txt'.format(envParams.pathToOutputDirectory, i, j), usecols=(2, 3, 4));
            curPhPortrType = describePortrType(data.tolist())
            if curPhPortrType not in diffTypes:
                diffTypes[curPhPortrType] = curTypeNumber
                curTypeNumber += 1.
            colorGrid[i][j] = diffTypes[curPhPortrType]
    plt.pcolormesh(arrFirstParam, arrSecondParam, colorGrid, cmap=plt.cm.get_cmap('RdBu'))
    plt.colorbar()
    plt.savefig('{}{}.pdf'.format(envParams.pathToOutputDirectory, envParams.imageStamp))


def createDistMatrix(coordinatesPoins):
    X = coordinatesPoins[:, 0]
    Y = coordinatesPoins[:, 1]
    len(X)
    Matrix = np.zeros((len(X), len(X))) * np.NaN
    for i in range(len(X)):
        for j in range(len(X)):
            Matrix[i][j] = np.sqrt((X[i] - X[j]) ** 2 + (Y[i] - Y[j]) ** 2)
    return Matrix


def work(distMatrix, distThreshold):
    ######################
    currmatrix = np.array(distMatrix)
    adjMatrix = (currmatrix <= distThreshold) * 1.0
    nComps, labels = connected_components(adjMatrix, directed=False)
    allData = [l for l in labels]
    ######################
    return allData


def mergePoints(connectedPoints, nSnCnU):
    arrDiffPoints = {}
    for i in range(len(connectedPoints)):
        pointParams = np.append(nSnCnU[i], connectedPoints[i])
        if tuple(pointParams) not in arrDiffPoints:
            arrDiffPoints[tuple(pointParams)] = i
    return arrDiffPoints.values()


class FourBiharmonicPhaseOscillators:
    def __init__(self, paramW, paramA, paramB, paramR):
        self.paramW = paramW
        self.paramA = paramA
        self.paramB = paramB
        self.paramR = paramR

    def funG(self, fi):
        return -np.sin(fi + self.paramA) + self.paramR * np.sin(2 * fi + self.self.paramB)

    def getFullSystem(self, phis):
        rhsPhis = [0,0,0,0]
        for i in range(4):
            elem = self.paramW
            for j in range(4):
                elem += 0.25 * self.funG(phis[i]-phis[j])
            rhsPhis[i] = elem
        return rhsPhis

    def getReducedSystem(self, gammas):
        phis = [0] + gammas
        rhsPhi = self.getFullSystem(phis)
        rhsGamma = [0,0,0,0]
        for i in range(4):
            rhsGamma[i] = rhsPhi[i]-rhsPhi[1]
        return rhsGamma[1:]

    def getRestriction(self,psi):
        gammas = [0] + psi
        rhsPsi = self.getReducedSystem(gammas)
        return rhsPsi[1:]
