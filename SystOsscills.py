import numpy as np

class FourBiharmonicPhaseOscillators:
    def __init__(self, paramW, paramA, paramB, paramR):
        self.paramW = paramW
        self.paramA = paramA
        self.paramB = paramB
        self.paramR = paramR

    def funG(self, fi):
        return -np.sin(fi + self.paramA) + self.paramR * np.sin(2 * fi + self.paramB)

    def getFullSystem(self, phis):
        rhsPhis = [0,0,0,0]
        for i in range(4):
            elem = self.paramW
            for j in range(4):
                elem += 0.25 * self.funG(phis[i]-phis[j])
            rhsPhis[i] = elem
        return rhsPhis

    def getReducedSystem(self, gammas):
        gammas = list(gammas)
        phis = [0] + gammas
        rhsPhi = self.getFullSystem(phis)
        rhsGamma = [0,0,0,0]
        for i in range(4):
            rhsGamma[i] = rhsPhi[i]-rhsPhi[0]
        return rhsGamma[1:]

    def getRestriction(self,psi):
        psi = list(psi)
        gammas = [0] + psi
        rhsPsi = self.getReducedSystem(gammas)
        return rhsPsi[1:]