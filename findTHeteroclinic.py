import systems_fun as sf
from collections import defaultdict
import itertools as itls
import SystOsscills as a4d
from scipy.spatial import distance


def checkConnection(pairsToCheck, ps: sf.PrecisionSettings, rhs, rhsJac, phSpaceTransformer, sepCondition, eqTransformer, sepNumCondition, sepProximity, maxTime):
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

    for alphaEq, omegaEqs in grpByAlphaEq.items():
        alphaEqTr = phSpaceTransformer(alphaEq, rhsJac)
        omegaEqsTr = [phSpaceTransformer(oEq, rhsJac) for oEq in omegaEqs]
        fullOmegaEqsTr = itls.chain.from_iterable([eqTransformer(oEq, rhsJac) for oEq in omegaEqsTr])
        separatrices = sf.computeSeparatrices(alphaEqTr, rhs, ps, maxTime, sepCondition)

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

def hasExactly(num, seps):
    return len(seps) == num

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

def checkTargetHeteroclinic(osc: a4d.FourBiharmonicPhaseOscillators, borders, bounds, eqFinder, ps: sf.PrecisionSettings):
    rhsInvPlane = osc.getRestriction
    jacInvPlane = osc.getRestrictionJac
    rhsReduced = osc.getReducedSystem
    jacReduced = osc.getReducedSystemJac

    planeEqCoords = sf.findEquilibria(rhsInvPlane, jacInvPlane, bounds, borders, eqFinder, ps)
    tresserPairs = sf.getTresserPairs(planeEqCoords, osc, ps)
    cnctInfo = checkConnection(tresserPairs, ps, rhsInvPlane, jacInvPlane, idTransform, sf.pickBothSeparatrices, idListTransform, anyNumber, ps.sdlSinkPrxty, 1000)
    newPairs = {(it['omega'], it['alpha']) for it in cnctInfo}
    finalInfo = checkConnection(newPairs, ps, rhsReduced, jacReduced, embedBackTransform, sf.pickCirSeparatrix, cirTransform, lambda X: hasExactly(1, X), ps.sfocSddlPrxty, 1000)
    return finalInfo
