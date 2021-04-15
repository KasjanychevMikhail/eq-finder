import systems_fun as sf
from collections import defaultdict
import itertools as itls
import SystOsscills as a4d
from scipy.spatial import distance


def checkSeparatrixConnection(pairsToCheck, ps: sf.PrecisionSettings, rhs, rhsJac, phSpaceTransformer, sepCondition, eqTransformer, sepNumCondition, sepProximity, maxTime, withEvents = False, listEqCoords3D = None):
    """
    Accepts pairsToCheck — a list of pairs of Equilibria — and checks if there is
    an approximate connection between them. First equilibrium of pair
    must be a saddle with one-dimensional unstable manifold. The precision of
    connection is given by :param sepProximity.
    """
    grpByAlphaEq = defaultdict(list)
    for alphaEq, omegaEq in pairsToCheck:
        grpByAlphaEq[alphaEq].append(omegaEq)

    outputInfo = []

    events = None

    for alphaEq, omegaEqs in grpByAlphaEq.items():
        alphaEqTr = phSpaceTransformer(alphaEq, rhsJac)
        omegaEqsTr = [phSpaceTransformer(oEq, rhsJac) for oEq in omegaEqs]
        fullOmegaEqsTr = itls.chain.from_iterable([eqTransformer(oEq, rhsJac) for oEq in omegaEqsTr])
        if withEvents:
            events = sf.createListOfEvents(alphaEqTr, listEqCoords3D, ps)
        separatrices = sf.computeSeparatrices(alphaEqTr, rhs, ps, maxTime, sepCondition, events)

        if not sepNumCondition(separatrices):
            raise ValueError('Assumption on the number of separatrices is not satisfied')

        for omegaEqTr in fullOmegaEqsTr:
            for separatrix in separatrices:
                dist = distance.cdist(separatrix, [omegaEqTr.coordinates]).min()
                if dist < sepProximity:
                    info = {}
                    # TODO: what exactly to output
                    info['alpha'] = alphaEqTr
                    info['omega'] = omegaEqTr
                    info['stPt']  = separatrix[0]
                    info['dist'] = dist
                    outputInfo.append(info)

    return outputInfo

def idTransform(X, rhsJac):
    """
    Accepts an Equilibrium and returns it as is
    """
    return X

def idListTransform(X, rhsJac):
    """
    Accepts an Equilibrium and returns it wrapped in list
    """
    return [X]

def hasExactly(num):
    return lambda seps: len(seps) == num

def anyNumber(seps):
    return True

def embedBackTransform(X: sf.Equilibrium, rhsJac):
    """
    Takes an Equilbrium from invariant plane
    and reinterprets it as an Equilibrium
    of reduced system
    """
    xNew = sf.embedPointBack(X.coordinates)
    return sf.getEquilibriumInfo(xNew, rhsJac)

def cirTransform(eq: sf.Equilibrium, rhsJac):
    coords = sf.generateSymmetricPoints(eq.coordinates)
    return [sf.getEquilibriumInfo(cd, rhsJac) for cd in coords]

def checkTargetHeteroclinic(osc: a4d.FourBiharmonicPhaseOscillators, borders, bounds, eqFinder, ps: sf.PrecisionSettings, maxTime):
    rhsInvPlane = osc.getRestriction
    jacInvPlane = osc.getRestrictionJac
    rhsReduced = osc.getReducedSystem
    jacReduced = osc.getReducedSystemJac

    planeEqCoords = sf.findEquilibria(rhsInvPlane, jacInvPlane, bounds, borders, eqFinder, ps)
    eqCoords3D = sf.listEqOnInvPlaneTo3D(planeEqCoords, osc)
    tresserPairs = sf.getTresserPairs(planeEqCoords, osc, ps)

    cnctInfo = checkSeparatrixConnection(tresserPairs, ps, rhsInvPlane, jacInvPlane, idTransform, sf.pickBothSeparatrices, idListTransform, anyNumber, ps.sdlSinkPrxty, maxTime)
    newPairs = {(it['omega'], it['alpha']) for it in cnctInfo}
    finalInfo = checkSeparatrixConnection(newPairs, ps, rhsReduced, jacReduced, embedBackTransform, sf.pickCirSeparatrix, cirTransform, hasExactly(1), ps.sfocSddlPrxty, maxTime, withEvents = True, listEqCoords3D = eqCoords3D)
    return finalInfo

def checkTargetHeteroclinicInInterval(osc: a4d.FourBiharmonicPhaseOscillators, borders, bounds, eqFinder, ps: sf.PrecisionSettings, maxTime, lowerLimit):
    info = checkTargetHeteroclinic(osc, borders, bounds, eqFinder, ps, maxTime)
    finalInfo = []
    for dic in info:
        if dic['dist'] > lowerLimit:
            finalInfo.append(dic)
    return finalInfo

def getStartPtsForLyapVals(osc: a4d.FourBiharmonicPhaseOscillators, borders, bounds, eqFinder, ps: sf.PrecisionSettings, OnlySadFoci):
    rhsInvPlane = osc.getRestriction
    jacInvPlane = osc.getRestrictionJac
    planeEqCoords = sf.findEquilibria(rhsInvPlane, jacInvPlane, bounds, borders, eqFinder, ps)
    ListEqs1dU = sf.get1dUnstEqs(planeEqCoords, osc, ps, OnlySadFoci)
    outputInfo = []

    for eq in ListEqs1dU:
        ptOnInvPlane = eq.coordinates
        ptOnPlaneIn3D = sf.embedPointBack(ptOnInvPlane)
        eqOnPlaneIn3D = sf.getEquilibriumInfo(ptOnPlaneIn3D, osc.getReducedSystemJac)
        startPts = sf.getInitPointsOnUnstable1DSeparatrix(eqOnPlaneIn3D,sf.pickCirSeparatrix, ps)[0]

        outputInfo.append(startPts)
    return outputInfo