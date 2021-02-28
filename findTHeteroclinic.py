import systems_fun as sf
from collections import defaultdict
import itertools as itls
import SystOsscills as a4d
from scipy.spatial import distance


def checkConnection(pairsToCheck, ps: sf.PrecisionSettings, rhs, phSpaceTransformer, sepCondition, eqTransformer, sepNumCondition, sepProximity):
    # pairsToCheck: (Fst, Snd) : if separatrix from Fst goes close to Snd
    grpByAlphaEq = defaultdict(list)
    for alphaEq, omegaEq in pairsToCheck:
        grpByAlphaEq[alphaEq].append(omegaEq)

    for alphaEq, omegaEqs in grpByAlphaEq:
        alphaEqTr = phSpaceTransformer(alphaEq)
        omegaEqsTr = [phSpaceTransformer(oEq) for oEq in omegaEqs]
        fullOmegaEqsTr = itls.chain.from_iterable([eqTransformer(oEq) for oEq in omegaEqsTr])
        separatrices = sf.computeSeparatrices(alphaEqTr, rhs, ps, 1000, sepCondition)

        if sepNumCondition(separatrices):
            raise ('ValueError', 'Assumption on the number of separatrices is not satisfied')

        outputInfo = []
        for omegaEqTr in fullOmegaEqsTr:
            for separatrix in separatrices:
                dist = distance.cdist(separatrix, [omegaEqTr]).min()
                if dist < sepProximity:
                    info = {}
                    # TODO: what exactly to output
                    info['alpha'] = alphaEqTr
                    info['omega'] = omegaEqTr
                    info['stPt']  = separatrix[0]
                    info['dist'] = dist
                    outputInfo.append(info)
        return outputInfo

def idTransform(X):
    return X

def idListTransform(X):
    return [X]

def hasExactly(num, seps):
    return len(seps)==num

def anyNumber(seps):
    return True

def checkTargetHeteroclinic(osc: a4d.FourBiharmonicPhaseOscillators, borders, bounds, eqFinder, ps: sf.PrecisionSettings):
    rhsInvPlane = osc.getRestriction
    jacInvPlane = osc.getRestrictionJac
    rhsReduced = osc.getReducedSystem
    planeEqCoords = sf.findEquilibria(rhsInvPlane,jacInvPlane, bounds, borders, eqFinder, ps)
    tresserPairs = sf.getTresserPairs(planeEqCoords, osc, ps)
    cnctInfo = checkConnection(tresserPairs,ps, rhsInvPlane, idTransform, sf.pickBothSeparatrices, idListTransform, anyNumber, ps.sdlSinkPrxty)
    print(cnctInfo)
    newPairs = {(it['omega'], it['alpha']) for it in cnctInfo}
    finalInfo = checkConnection(newPairs, ps, rhsReduced, sf.embedPointBack, sf.pickCirSeparatrix, sf.T, lambda X: hasExactly(1, X), ps.sfocSddlPrxty)
    return finalInfo
