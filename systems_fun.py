import os
import subprocess
import scipy
import matplotlib.pyplot as plt
import numpy as np

from collections import namedtuple
from scipy import optimize
from numpy import linalg as LA
from sklearn.cluster import AgglomerativeClustering
from scipy.integrate import solve_ivp

MapParameters = namedtuple('MapParameters', ['rhs', 'rhsJac', 'param', 'bounds',
                                             'borders','optMethod'])


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

class Equilibrium:
    def __init__(self, coordinates, eigenvalues, eigvectors = None):
        self.coordinates = list(coordinates)
        self.eigenvalues = sorted(eigenvalues, key=lambda eigenvalues: eigenvalues.real)
        self.eqType = describeEqType(np.array(self.eigenvalues))
        self.eigvectors = eigvectors
        if len(eigenvalues) != len(coordinates):
            raise ValueError('Vector of coordinates and vector of eigenvalues must have the same size!')

    def __repr__(self):
        return "Equilibrium\nCoordinates: {}\nEigenvalues: {}\nType: {}".format(self.coordinates, self.eigenvalues, self.eqType)

    def __str__(self):
        return ' '.join([str(it) for it in self.coordinates + self.eqType + self.eigenvalues])

    def strToFile(self):
        eigs = []
        for val in self.eigenvalues:
            eigs += [val.real, val.imag]
        return self.coordinates + self.eqType + eigs




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
    return [nS, nC, nU, issc, isuc]


def describePortrType(arrEqSignatures):
    phSpaceDim = int(sum(arrEqSignatures[0]))
    eqTypes = {(i, phSpaceDim-i):0 for i in range(phSpaceDim+1)}
    nonRough = 0
    for eqSign in arrEqSignatures:
        nS, nC, nU = eqSign
        if nC == 0:
            eqTypes[(nU, nS)] += 1
        else:
            nonRough += 1
    # nSinks, nSaddles, nSources,  nNonRough
    portrType = tuple([eqTypes[(i, phSpaceDim-i)] for i in range(phSpaceDim+1)] + [nonRough])
    return portrType

class ShgoEqFinder:
    def __init__(self, nSamples, nIters, eps):
        self.nSamples = nSamples
        self.nIters = nIters
        self.eps = eps
    def __call__(self, rhs, rhsSq, rhsJac, boundaries, borders):
        optResult = scipy.optimize.shgo(rhsSq, boundaries, n=self.nSamples, iters=self.nIters, sampling_method='sobol');
        allEquilibria = [x for x, val in zip(optResult.xl, optResult.funl) if
                         abs(val) < self.eps and inBounds(x, borders)];
        return allEquilibria

class NewtonEqFinder:
    def __init__(self, xGridSize, yGridSize, eps):
        self.xGridSize = xGridSize
        self.yGridSize = yGridSize
        self.eps = eps
    def __call__(self, rhs, rhsSq, rhsJac, boundaries, borders):
        Result = []
        for i, x in enumerate(np.linspace(boundaries[0][0], boundaries[0][1], self.xGridSize)):
            for j, y in enumerate(np.linspace(boundaries[1][0], boundaries[1][1], self.yGridSize)):
                Result.append(optimize.root(rhs, [x, y], method='broyden1', jac=rhsJac).x)
        allEquilibria = [x for x in Result if abs(rhsSq(x)) < self.eps and inBounds(x, borders)];
        return allEquilibria


class NewtonEqFinderUp:
    def __init__(self, xGridSize, yGridSize, eps):
        self.xGridSize = xGridSize
        self.yGridSize = yGridSize
        self.eps = eps
    def test(self, rhs,x,y,step):
        res = 0
        if rhs((x, y))[0] * rhs((x, y + step))[0] < 0:
            res = 1
        elif rhs((x, y + step))[0] * rhs((x + step, y + step))[0] < 0:
            res = 1
        elif rhs((x + step, y + step))[0] * rhs((x + step, y))[0] < 0:
            res = 1
        elif rhs((x + step, y))[0] * rhs((x, y))[0] < 0:
            res = 1
        if res:
            if rhs((x, y))[1] * rhs((x, y + step))[1] < 0:
                res = 1
            elif rhs((x, y + step))[1] * rhs((x + step, y + step))[1] < 0:
                res = 1
            elif rhs((x + step, y + step))[1] * rhs((x + step, y))[1] < 0:
                res = 1
            elif rhs((x + step, y))[1] * rhs((x, y))[1] < 0:
                res = 1
        return res
    def __call__(self, rhs, rhsSq, rhsJac, boundaries, borders):
        rectangles = np.zeros((self.xGridSize - 1, self.yGridSize - 1))

        x = boundaries[0][0]
        step =  (boundaries[0][1]-boundaries[0][0])/ (self.yGridSize - 1)
        for i in range (self.xGridSize - 1):
            y = boundaries[1][0]
            for j in range (self.yGridSize - 1):
                if self.test(rhs,x,y,step):
                        rectangles[self.xGridSize - i - 2][self.yGridSize - j - 2] = 1
                y += step
            x += step

        Result = []
        for i in range (self.xGridSize):
            for j in range (self.yGridSize):
                if rectangles[self.xGridSize - i - 2][self.yGridSize - j - 2]:
                    Result.append(optimize.root(rhs, [boundaries[0][0] + i * step, boundaries[1][0] + j * step],
                                        method='broyden1', jac=rhsJac).x)
        allEquilibria = [x for x in Result if abs(rhsSq(x)) < self.eps and inBounds(x, borders)]
        return allEquilibria
def createEqList (allEquilibria, rhsJac):
    allEquilibria = sorted(allEquilibria, key=lambda ar: tuple(ar))
    EqList = []
    for k, eqCoords in enumerate(allEquilibria):
        eqJacMatrix = rhsJac(eqCoords)
        eigvals, _ = LA.eig(eqJacMatrix)
        EqList.append(Equilibrium(eqCoords, eigvals))
    if len(EqList) > 1:
        trueStr = filterEq(EqList)
        trueEqList = [EqList[i] for i in list(trueStr)]
        return trueEqList

    return EqList

def findEquilibria(mapParams):
    def rhsSq(x):
        xArr = np.array(x)
        vec = mapParams.rhs(xArr)
        return np.dot(vec, vec)
    
    allEquilibria = mapParams.optMethod(mapParams.rhs, rhsSq, mapParams.rhsJac, mapParams.bounds, mapParams.borders)
    return createEqList(allEquilibria,mapParams.rhsJac)


def inBounds(X, boundaries):
    x, y = X
    Xb, Yb = boundaries
    b1, b2 = Xb
    b3, b4 = Yb
    return ((x > b1) and (x < b2) and (y > b3) and (y < b4))

def filterEq(listEquilibria):
    X = []
    data = []
    for eq in listEquilibria:
        X.append(eq.coordinates)
        data.append(eq.eqType)
    clustering = AgglomerativeClustering(n_clusters=None, affinity='euclidean', linkage='single',
                                         distance_threshold=(1e-5))
    clustering.fit(X)
    return indicesUniqueEq(clustering.labels_, data)

def writeToFileEqList(envParams, EqList,params, nameOfFile):
    sol = []
    for eq in EqList:
        sol.append(eq.strToFile())
    headerStr = ('gamma = {par[0]}\n' +
                 'd = {par[1]}\n' +
                 'X  Y  nS  nC  nU  isSComplex  isUComplex  Re(eigval1)  Im(eigval1)  Re(eigval2)  Im(eigval2)\n' +
                 '0  1  2   3   4   5           6           7            8            9            10').format(par=params)
    fmtList = ['%+18.15f',
               '%+18.15f',
               '%2u',
               '%2u',
               '%2u',
               '%2u',
               '%2u',
               '%+18.15f',
               '%+18.15f',
               '%+18.15f',
               '%+18.15f', ]
    np.savetxt("{env.pathToOutputDirectory}{}.txt".format(nameOfFile, env=envParams), sol, header=headerStr,fmt=fmtList)

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
            colorGrid[j][i] = diffTypes[curPhPortrType]
    plt.pcolormesh(arrFirstParam, arrSecondParam, colorGrid, cmap=plt.cm.get_cmap('RdBu'))
    plt.colorbar()
    plt.savefig('{}{}.pdf'.format(envParams.pathToOutputDirectory, envParams.imageStamp))



def indicesUniqueEq(connectedPoints, nSnCnU):
    arrDiffPoints = {}
    for i in range(len(connectedPoints)):
        pointParams = np.append(nSnCnU[i], connectedPoints[i])
        if tuple(pointParams) not in arrDiffPoints:
            arrDiffPoints[tuple(pointParams)] = i
    return arrDiffPoints.values()

def valP (saddlEq,sdlFocEq):
    saddlEigs = saddlEq.eigvals
    sdlFocEigs = sdlFocEq.eigvals
    # assume that sdlEigs is sorted
    sdlLeadingS = max([se for se in saddlEigs if se.real < -1e-14])
    sdlLeadingU = min([se for se in saddlEigs if se.real > +1e-14])
    # assume that sdlFocEigs is sorted
    sdlFocLeadS = max([se for se in sdlFocEigs if se.real < -1e-14])
    sdlFocLeadU = min([se for se in sdlFocEigs if se.real > +1e-14])
    p = (-sdlLeadingU / sdlLeadingS) * (-sdlFocLeadU / sdlFocLeadS)
    return p

def goodConfEqList(EqList, rhs):
    sadFocs=[]
    saddles=[]
    for i,eq in enumerate(EqList):
        X=eq.coordinates
        eqJacMatrix = rhs.getReducedSystemJac([0]+X)
        eigvals, eigvecs = LA.eig(eqJacMatrix)
        vecs = []
        for i in enumerate(eigvals):
            vecs.append(eigvecs[:, i])
        xs,ys =X
        if(xs<=ys):
            if (np.array_equal(eq.eqType ,(2,0,0,1,0)) and np.array_equal(describeEqType(eigvals),(2,0,1,1,0))):
                indices = sorted(eigvals, key=lambda eigvals: eigvals.real)
                sadFocs.append(Equilibrium([0]+X, [eigvals[i] for i in indices],[vecs[i] for i in indices]))
            elif (np.array_equal(eq.eqType , (1,0,1,0,0)) and np.array_equal(describeEqType(eigvals),(2,0,1,0,0))):
                eigvals = sorted(eigvals, key=lambda eigvals: eigvals.real)
                saddles.append(Equilibrium([0]+X, eigvals))
    return [[sf, sd] for sf in sadFocs for sd in saddles if valP(sf, sd) > 1.]


def createListSymmSaddles(coordsSaddle):
    cs = coordsSaddle
    for i in range(3):
        cs.append([cs[i][1] - cs[i][0], cs[i][2] - cs[i][0], 2 * np.pi - cs[i][0]])
    return cs

def minDistToSaddle(lastPointTraj,coordsSaddle):
    x,y,z = lastPointTraj
    minDist = 10
    cs = coordsSaddle
    for  coordSad in (cs):
        dist = ((coordSad[0] - x) ** 2 + (coordSad[1] - y) ** 2 + (coordSad[2] - z) ** 2) ** 0.5
        if (minDist > dist):
            minDist = dist
    return minDist

def heterСheck(pairSfSd, rhs,maxTime):
    sf,sd = pairSfSd
    coordSymmSadds = createListSymmSaddles(sd.coordinates)
    x0 = sf.coordinates
    vec = sf.eigvectors[2]
    if (vec[0] < 0):
        vec = -1*vec
    x0 = np.add(np.array(x0), (vec.real) * 1e-5)
    rhs_vec = lambda t, X: rhs.getReducedSystem(X)
    sol = solve_ivp(rhs_vec, [0, maxTime], x0, rtol=1e-11, atol=1e-11, dense_output=True)
    x, y, z = sol.y
    return minDistToSaddle((x[-1],y[-1],z[-1]),coordSymmSadds)


