import numpy as np
import sympy as sp

class FivePhaseOscillators:
    def __init__(self, paramE1, paramE2, paramE3, paramE4, paramE5, paramX1, paramX2, paramX3, paramX4, paramX5,
                 paramEps):
        self.paramE1 = paramE1
        self.paramE2 = paramE2
        self.paramE3 = paramE3
        self.paramE4 = paramE4
        self.paramE5 = paramE5
        self.paramX1 = paramX1
        self.paramX2 = paramX2
        self.paramX3 = paramX3
        self.paramX4 = paramX4
        self.paramX5 = paramX5
        self.paramEps = paramEps

    def funcG2(self, phi):
        return self.paramE1 * np.cos(phi + self.paramX1) + self.paramE2 * np.cos(2 * phi + self.paramX2)

    def funcG3(self, phi):
        return self.paramE3 * np.cos(phi + self.paramX3)

    def funcG4(self, phi):
        return self.paramE4 * np.cos(phi + self.paramX4)

    def funcG5(self, phi):
        return self.paramE5 * np.cos(phi + self.paramX5)

    def sum1(self, psis, j):
        tmp = 0
        for k in range(4):
            tmp += self.funcG2(psis[k] - psis[j]) - self.funcG2(psis[k])
        return tmp

    def sum2(self, psis, j):
        tmp = 0
        for k in range(4):
            for l in range(4):
                tmp += self.funcG3(psis[k] + psis[l] - 2 * psis[j]) - self.funcG3(psis[k] + psis[l])
        return tmp

    def sum3(self, psis, j):
        tmp = 0
        for k in range(4):
            for l in range(4):
                tmp += self.funcG4(2 * psis[k] - psis[l] - psis[j]) - self.funcG4(2 * psis[k] - psis[l])
        return tmp

    def sum4(self, psis, j):
        tmp = 0
        for k in range(4):
            for l in range(4):
                for m in range(4):
                    tmp += self.funcG5(psis[k] + psis[l] - psis[m] - psis[j]) - self.funcG5(psis[k] + psis[l] - psis[m])
        return tmp

    def getReducedSystem(self, psis):
        psis = [0] + list(psis)
        res = psis
        for i in range(4):
            if i == 0:
                res[i] = 0
            else:
                res[i] = self.paramEps / 4 * self.sum1(psis, i) + self.paramEps / 16 * self.sum2(psis, i) + \
                         self.paramEps / 16 * self.sum3(psis, i) + self.paramEps / 64 * self.sum4(psis, i)
        return res[1:]

    def getRestriction(self, psi):
        psi = list(psi)      
        gammas = [0] + psi
        rhsPsi = self.getReducedSystem(gammas)
        return rhsPsi[1:]

    def f1x1(self, psis):
        return self.paramEps*(-self.paramE1*np.sin(psis[1] - self.paramX1) +
                              self.paramE1*np.sin(psis[1] + self.paramX1) +
                              self.paramE1*np.sin(-psis[1] + psis[2] + self.paramX1) +
                              self.paramE1*np.sin(-psis[1] + psis[3] + self.paramX1) -
                              2*self.paramE2*np.sin(2*psis[1] - self.paramX2) +
                              2*self.paramE2*np.sin(2*psis[1] + self.paramX2) +
                              2*self.paramE2*np.sin(-2*psis[1] + 2*psis[2] + self.paramX2) +
                              2*self.paramE2*np.sin(-2*psis[1] + 2*psis[3] + self.paramX2))/4 + \
               self.paramEps*(-2*self.paramE3*np.sin(psis[1] - self.paramX3) +
                              2*self.paramE3*np.sin(psis[1] + self.paramX3) -
                              2*self.paramE3*np.sin(2*psis[1] - self.paramX3) +
                              2*self.paramE3*np.sin(2*psis[1] + self.paramX3) +
                              4*self.paramE3*np.sin(-2*psis[1] + psis[2] + self.paramX3) +
                              2*self.paramE3*np.sin(-2*psis[1] + 2*psis[2] + self.paramX3) +
                              4*self.paramE3*np.sin(-2*psis[1] + psis[3] + self.paramX3) +
                              2*self.paramE3*np.sin(-2*psis[1] + 2*psis[3] + self.paramX3) +
                              2*self.paramE3*np.sin(-psis[1] + psis[2] + self.paramX3) +
                              2*self.paramE3*np.sin(-psis[1] + psis[3] + self.paramX3) +
                              2*self.paramE3*np.sin(psis[1] + psis[2] + self.paramX3) +
                              2*self.paramE3*np.sin(psis[1] + psis[3] + self.paramX3) +
                              4*self.paramE3*np.sin(-2*psis[1] + psis[2] + psis[3] + self.paramX3))/16 + \
               self.paramEps*(-2*self.paramE4*np.sin(2*psis[1] - self.paramX4) +
                              2*self.paramE4*np.sin(2*psis[1] + self.paramX4) +
                              2*self.paramE4*np.sin(-2*psis[1] + 2*psis[2] + self.paramX4) +
                              2*self.paramE4*np.sin(-2*psis[1] + 2*psis[3] + self.paramX4) +
                              self.paramE4*np.sin(-psis[1] + psis[2] + self.paramX4) +
                              self.paramE4*np.sin(-psis[1] + psis[3] + self.paramX4) -
                              self.paramE4*np.sin(psis[1] - psis[2] + self.paramX4) -
                              self.paramE4*np.sin(psis[1] + psis[2] - self.paramX4) -
                              self.paramE4*np.sin(psis[1] - psis[3] + self.paramX4) -
                              self.paramE4*np.sin(psis[1] + psis[3] - self.paramX4) +
                              2*self.paramE4*np.sin(2*psis[1] - psis[2] + self.paramX4) +
                              2*self.paramE4*np.sin(2*psis[1] - psis[3] + self.paramX4) -
                              self.paramE4*np.sin(psis[1] - 2*psis[2] + psis[3] - self.paramX4) -
                              self.paramE4*np.sin(psis[1] + psis[2] - 2*psis[3] - self.paramX4))/16 + \
               self.paramEps*(-6*self.paramE5*np.sin(psis[1] - self.paramX5) +
                              6*self.paramE5*np.sin(psis[1] + self.paramX5) -
                              2*self.paramE5*np.sin(2*psis[1] - self.paramX5) +
                              2*self.paramE5*np.sin(2*psis[1] + self.paramX5) +
                              4*self.paramE5*np.sin(-2*psis[1] + psis[2] + self.paramX5) +
                              2*self.paramE5*np.sin(-2*psis[1] + 2*psis[2] + self.paramX5) +
                              4*self.paramE5*np.sin(-2*psis[1] + psis[3] + self.paramX5) +
                              2*self.paramE5*np.sin(-2*psis[1] + 2*psis[3] + self.paramX5) +
                              5*self.paramE5*np.sin(-psis[1] + psis[2] + self.paramX5) +
                              5*self.paramE5*np.sin(-psis[1] + psis[3] + self.paramX5) +
                              self.paramE5*np.sin(psis[1] - psis[2] + self.paramX5) -
                              self.paramE5*np.sin(psis[1] + psis[2] - self.paramX5) +
                              2*self.paramE5*np.sin(psis[1] + psis[2] + self.paramX5) +
                              self.paramE5*np.sin(psis[1] - psis[3] + self.paramX5) -
                              self.paramE5*np.sin(psis[1] + psis[3] - self.paramX5) +
                              2*self.paramE5*np.sin(psis[1] + psis[3] + self.paramX5) +
                              2*self.paramE5*np.sin(2*psis[1] - psis[2] + self.paramX5) +
                              2*self.paramE5*np.sin(2*psis[1] - psis[3] + self.paramX5) +
                              4*self.paramE5*np.sin(-2*psis[1] + psis[2] + psis[3] + self.paramX5) -
                              self.paramE5*np.sin(psis[1] - 2*psis[2] + psis[3] - self.paramX5) -
                              2*self.paramE5*np.sin(psis[1] - psis[2] + psis[3] - self.paramX5) +
                              2*self.paramE5*np.sin(psis[1] - psis[2] + psis[3] + self.paramX5) -
                              self.paramE5*np.sin(psis[1] + psis[2] - 2*psis[3] - self.paramX5) -
                              2*self.paramE5*np.sin(psis[1] + psis[2] - psis[3] - self.paramX5) +
                              2*self.paramE5*np.sin(psis[1] + psis[2] - psis[3] + self.paramX5))/64

    def f1x2(self, psis):
        return self.paramEps*(self.paramE1*np.sin(psis[2] + self.paramX1) -
                              self.paramE1*np.sin(-psis[1] + psis[2] + self.paramX1) +
                              2*self.paramE2*np.sin(2*psis[2] + self.paramX2) -
                              2*self.paramE2*np.sin(-2*psis[1] + 2*psis[2] + self.paramX2))/4 + \
               self.paramEps*(2*self.paramE3*np.sin(psis[2] + self.paramX3) +
                              2*self.paramE3*np.sin(2*psis[2] + self.paramX3) -
                              2*self.paramE3*np.sin(-2*psis[1] + psis[2] + self.paramX3) -
                              2*self.paramE3*np.sin(-2*psis[1] + 2*psis[2] + self.paramX3) -
                              2*self.paramE3*np.sin(-psis[1] + psis[2] + self.paramX3) +
                              2*self.paramE3*np.sin(psis[1] + psis[2] + self.paramX3) +
                              2*self.paramE3*np.sin(psis[2] + psis[3] + self.paramX3) -
                              2*self.paramE3*np.sin(-2*psis[1] + psis[2] + psis[3] + self.paramX3))/16 + \
               self.paramEps*(self.paramE4*np.sin(psis[2] - self.paramX4) +
                              self.paramE4*np.sin(psis[2] + self.paramX4) +
                              2*self.paramE4*np.sin(2*psis[2] + self.paramX4) -
                              2*self.paramE4*np.sin(-2*psis[1] + 2*psis[2] + self.paramX4) -
                              self.paramE4*np.sin(-psis[1] + psis[2] + self.paramX4) +
                              self.paramE4*np.sin(psis[1] - psis[2] + self.paramX4) -
                              self.paramE4*np.sin(psis[1] + psis[2] - self.paramX4) -
                              self.paramE4*np.sin(2*psis[1] - psis[2] + self.paramX4) -
                              self.paramE4*np.sin(-psis[2] + 2*psis[3] + self.paramX4) +
                              2*self.paramE4*np.sin(2*psis[2] - psis[3] + self.paramX4) +
                              2*self.paramE4*np.sin(psis[1] - 2*psis[2] + psis[3] - self.paramX4) -
                              self.paramE4*np.sin(psis[1] + psis[2] - 2*psis[3] - self.paramX4))/16 + \
               self.paramEps*(-self.paramE5*np.sin(psis[2] - self.paramX5) +
                              5*self.paramE5*np.sin(psis[2] + self.paramX5) +
                              2*self.paramE5*np.sin(2*psis[2] + self.paramX5) -
                              2*self.paramE5*np.sin(-2*psis[1] + psis[2] + self.paramX5) -
                              2*self.paramE5*np.sin(-2*psis[1] + 2*psis[2] + self.paramX5) -
                              5*self.paramE5*np.sin(-psis[1] + psis[2] + self.paramX5) -
                              self.paramE5*np.sin(psis[1] - psis[2] + self.paramX5) -
                              self.paramE5*np.sin(psis[1] + psis[2] - self.paramX5) +
                              2*self.paramE5*np.sin(psis[1] + psis[2] + self.paramX5) -
                              self.paramE5*np.sin(2*psis[1] - psis[2] + self.paramX5) -
                              self.paramE5*np.sin(-psis[2] + 2*psis[3] + self.paramX5) +
                              2*self.paramE5*np.sin(psis[2] + psis[3] + self.paramX5) +
                              2*self.paramE5*np.sin(2*psis[2] - psis[3] + self.paramX5) -
                              2*self.paramE5*np.sin(-2*psis[1] + psis[2] + psis[3] + self.paramX5) +
                              2*self.paramE5*np.sin(psis[1] - 2*psis[2] + psis[3] - self.paramX5) +
                              2*self.paramE5*np.sin(psis[1] - psis[2] + psis[3] - self.paramX5) -
                              2*self.paramE5*np.sin(psis[1] - psis[2] + psis[3] + self.paramX5) -
                              self.paramE5*np.sin(psis[1] + psis[2] - 2*psis[3] - self.paramX5) -
                              2*self.paramE5*np.sin(psis[1] + psis[2] - psis[3] - self.paramX5) +
                              2*self.paramE5*np.sin(psis[1] + psis[2] - psis[3] + self.paramX5))/64

    def f1x3(self, psis):
        return self.paramEps*(self.paramE1*np.sin(psis[3] + self.paramX1) -
                              self.paramE1*np.sin(-psis[1] + psis[3] + self.paramX1) +
                              2*self.paramE2*np.sin(2*psis[3] + self.paramX2) -
                              2*self.paramE2*np.sin(-2*psis[1] + 2*psis[3] + self.paramX2))/4 + \
               self.paramEps*(2*self.paramE3*np.sin(psis[3] + self.paramX3) +
                              2*self.paramE3*np.sin(2*psis[3] + self.paramX3) -
                              2*self.paramE3*np.sin(-2*psis[1] + psis[3] + self.paramX3) -
                              2*self.paramE3*np.sin(-2*psis[1] + 2*psis[3] + self.paramX3) -
                              2*self.paramE3*np.sin(-psis[1] + psis[3] + self.paramX3) +
                              2*self.paramE3*np.sin(psis[1] + psis[3] + self.paramX3) +
                              2*self.paramE3*np.sin(psis[2] + psis[3] + self.paramX3) -
                              2*self.paramE3*np.sin(-2*psis[1] + psis[2] + psis[3] + self.paramX3))/16 + \
               self.paramEps*(self.paramE4*np.sin(psis[3] - self.paramX4) +
                              self.paramE4*np.sin(psis[3] + self.paramX4) +
                              2*self.paramE4*np.sin(2*psis[3] + self.paramX4) -
                              2*self.paramE4*np.sin(-2*psis[1] + 2*psis[3] + self.paramX4) -
                              self.paramE4*np.sin(-psis[1] + psis[3] + self.paramX4) +
                              self.paramE4*np.sin(psis[1] - psis[3] + self.paramX4) -
                              self.paramE4*np.sin(psis[1] + psis[3] - self.paramX4) -
                              self.paramE4*np.sin(2*psis[1] - psis[3] + self.paramX4) +
                              2*self.paramE4*np.sin(-psis[2] + 2*psis[3] + self.paramX4) -
                              self.paramE4*np.sin(2*psis[2] - psis[3] + self.paramX4) -
                              self.paramE4*np.sin(psis[1] - 2*psis[2] + psis[3] - self.paramX4) +
                              2*self.paramE4*np.sin(psis[1] + psis[2] - 2*psis[3] - self.paramX4))/16 + \
               self.paramEps*(-self.paramE5*np.sin(psis[3] - self.paramX5) +
                              5*self.paramE5*np.sin(psis[3] + self.paramX5) +
                              2*self.paramE5*np.sin(2*psis[3] + self.paramX5) -
                              2*self.paramE5*np.sin(-2*psis[1] + psis[3] + self.paramX5) -
                              2*self.paramE5*np.sin(-2*psis[1] + 2*psis[3] + self.paramX5) -
                              5*self.paramE5*np.sin(-psis[1] + psis[3] + self.paramX5) -
                              self.paramE5*np.sin(psis[1] - psis[3] + self.paramX5) -
                              self.paramE5*np.sin(psis[1] + psis[3] - self.paramX5) +
                              2*self.paramE5*np.sin(psis[1] + psis[3] + self.paramX5) -
                              self.paramE5*np.sin(2*psis[1] - psis[3] + self.paramX5) +
                              2*self.paramE5*np.sin(-psis[2] + 2*psis[3] + self.paramX5) +
                              2*self.paramE5*np.sin(psis[2] + psis[3] + self.paramX5) -
                              self.paramE5*np.sin(2*psis[2] - psis[3] + self.paramX5) -
                              2*self.paramE5*np.sin(-2*psis[1] + psis[2] + psis[3] + self.paramX5) -
                              self.paramE5*np.sin(psis[1] - 2*psis[2] + psis[3] - self.paramX5) -
                              2*self.paramE5*np.sin(psis[1] - psis[2] + psis[3] - self.paramX5) +
                              2*self.paramE5*np.sin(psis[1] - psis[2] + psis[3] + self.paramX5) +
                              2*self.paramE5*np.sin(psis[1] + psis[2] - 2*psis[3] - self.paramX5) +
                              2*self.paramE5*np.sin(psis[1] + psis[2] - psis[3] - self.paramX5) -
                              2*self.paramE5*np.sin(psis[1] + psis[2] - psis[3] + self.paramX5))/64

    def f2x1(self, psis):
        return self.paramEps*(self.paramE1*np.sin(psis[1] + self.paramX1) -
                              self.paramE1*np.sin(psis[1] - psis[2] + self.paramX1) +
                              2*self.paramE2*np.sin(2*psis[1] + self.paramX2) -
                              2*self.paramE2*np.sin(2*psis[1] - 2*psis[2] + self.paramX2))/4 + \
               self.paramEps*(2*self.paramE3*np.sin(psis[1] + self.paramX3) +
                              2*self.paramE3*np.sin(2*psis[1] + self.paramX3) -
                              2*self.paramE3*np.sin(psis[1] - 2*psis[2] + self.paramX3) -
                              2*self.paramE3*np.sin(psis[1] - psis[2] + self.paramX3) +
                              2*self.paramE3*np.sin(psis[1] + psis[2] + self.paramX3) +
                              2*self.paramE3*np.sin(psis[1] + psis[3] + self.paramX3) -
                              2*self.paramE3*np.sin(2*psis[1] - 2*psis[2] + self.paramX3) -
                              2*self.paramE3*np.sin(psis[1] - 2*psis[2] + psis[3] + self.paramX3))/16 + \
               self.paramEps*(self.paramE4*np.sin(psis[1] - self.paramX4) +
                              self.paramE4*np.sin(psis[1] + self.paramX4) +
                              2*self.paramE4*np.sin(2*psis[1] + self.paramX4) +
                              self.paramE4*np.sin(-psis[1] + psis[2] + self.paramX4) -
                              self.paramE4*np.sin(-psis[1] + 2*psis[2] + self.paramX4) -
                              self.paramE4*np.sin(-psis[1] + 2*psis[3] + self.paramX4) -
                              self.paramE4*np.sin(psis[1] - psis[2] + self.paramX4) -
                              self.paramE4*np.sin(psis[1] + psis[2] - self.paramX4) -
                              2*self.paramE4*np.sin(2*psis[1] - 2*psis[2] + self.paramX4) +
                              2*self.paramE4*np.sin(2*psis[1] - psis[3] + self.paramX4) -
                              self.paramE4*np.sin(psis[1] + psis[2] - 2*psis[3] - self.paramX4) -
                              2*self.paramE4*np.sin(2*psis[1] - psis[2] - psis[3] + self.paramX4))/16 + \
               self.paramEps*(-self.paramE5*np.sin(psis[1] - self.paramX5) +
                              5*self.paramE5*np.sin(psis[1] + self.paramX5) +
                              2*self.paramE5*np.sin(2*psis[1] + self.paramX5) -
                              self.paramE5*np.sin(-psis[1] + psis[2] + self.paramX5) -
                              self.paramE5*np.sin(-psis[1] + 2*psis[2] + self.paramX5) -
                              self.paramE5*np.sin(-psis[1] + 2*psis[3] + self.paramX5) -
                              2*self.paramE5*np.sin(psis[1] - 2*psis[2] + self.paramX5) -
                              5*self.paramE5*np.sin(psis[1] - psis[2] + self.paramX5) -
                              self.paramE5*np.sin(psis[1] + psis[2] - self.paramX5) +
                              2*self.paramE5*np.sin(psis[1] + psis[2] + self.paramX5) +
                              2*self.paramE5*np.sin(psis[1] + psis[3] + self.paramX5) -
                              2*self.paramE5*np.sin(2*psis[1] - 2*psis[2] + self.paramX5) +
                              2*self.paramE5*np.sin(2*psis[1] - psis[3] + self.paramX5) -
                              2*self.paramE5*np.sin(-psis[1] + psis[2] + psis[3] + self.paramX5) -
                              2*self.paramE5*np.sin(psis[1] - 2*psis[2] + psis[3] + self.paramX5) -
                              2*self.paramE5*np.sin(psis[1] - psis[2] - psis[3] + self.paramX5) -
                              self.paramE5*np.sin(psis[1] + psis[2] - 2*psis[3] - self.paramX5) -
                              2*self.paramE5*np.sin(psis[1] + psis[2] - psis[3] - self.paramX5) +
                              2*self.paramE5*np.sin(psis[1] + psis[2] - psis[3] + self.paramX5) -
                              2*self.paramE5*np.sin(2*psis[1] - psis[2] - psis[3] + self.paramX5))/64

    def f2x2(self, psis):
        return self.paramEps*(-self.paramE1*np.sin(psis[2] - self.paramX1) +
                              self.paramE1*np.sin(psis[2] + self.paramX1) +
                              self.paramE1*np.sin(psis[1] - psis[2] + self.paramX1) +
                              self.paramE1*np.sin(-psis[2] + psis[3] + self.paramX1) -
                              2*self.paramE2*np.sin(2*psis[2] - self.paramX2) +
                              2*self.paramE2*np.sin(2*psis[2] + self.paramX2) +
                              2*self.paramE2*np.sin(2*psis[1] - 2*psis[2] + self.paramX2) +
                              2*self.paramE2*np.sin(-2*psis[2] + 2*psis[3] + self.paramX2))/4 + \
               self.paramEps*(-2*self.paramE3*np.sin(psis[2] - self.paramX3) +
                              2*self.paramE3*np.sin(psis[2] + self.paramX3) -
                              2*self.paramE3*np.sin(2*psis[2] - self.paramX3) +
                              2*self.paramE3*np.sin(2*psis[2] + self.paramX3) +
                              4*self.paramE3*np.sin(psis[1] - 2*psis[2] + self.paramX3) +
                              2*self.paramE3*np.sin(psis[1] - psis[2] + self.paramX3) +
                              2*self.paramE3*np.sin(psis[1] + psis[2] + self.paramX3) +
                              2*self.paramE3*np.sin(2*psis[1] - 2*psis[2] + self.paramX3) +
                              4*self.paramE3*np.sin(-2*psis[2] + psis[3] + self.paramX3) +
                              2*self.paramE3*np.sin(-2*psis[2] + 2*psis[3] + self.paramX3) +
                              2*self.paramE3*np.sin(-psis[2] + psis[3] + self.paramX3) +
                              2*self.paramE3*np.sin(psis[2] + psis[3] + self.paramX3) +
                              4*self.paramE3*np.sin(psis[1] - 2*psis[2] + psis[3] + self.paramX3))/16 + \
               self.paramEps*(-2*self.paramE4*np.sin(2*psis[2] - self.paramX4) +
                              2*self.paramE4*np.sin(2*psis[2] + self.paramX4) -
                              self.paramE4*np.sin(-psis[1] + psis[2] + self.paramX4) +
                              2*self.paramE4*np.sin(-psis[1] + 2*psis[2] + self.paramX4) +
                              self.paramE4*np.sin(psis[1] - psis[2] + self.paramX4) -
                              self.paramE4*np.sin(psis[1] + psis[2] - self.paramX4) +
                              2*self.paramE4*np.sin(2*psis[1] - 2*psis[2] + self.paramX4) +
                              2*self.paramE4*np.sin(-2*psis[2] + 2*psis[3] + self.paramX4) +
                              self.paramE4*np.sin(-psis[2] + psis[3] + self.paramX4) -
                              self.paramE4*np.sin(psis[2] - psis[3] + self.paramX4) -
                              self.paramE4*np.sin(psis[2] + psis[3] - self.paramX4) +
                              2*self.paramE4*np.sin(2*psis[2] - psis[3] + self.paramX4) -
                              self.paramE4*np.sin(psis[1] + psis[2] - 2*psis[3] - self.paramX4) +
                              self.paramE4*np.sin(2*psis[1] - psis[2] - psis[3] + self.paramX4))/16 + \
               self.paramEps*(-6*self.paramE5*np.sin(psis[2] - self.paramX5) +
                              6*self.paramE5*np.sin(psis[2] + self.paramX5) -
                              2*self.paramE5*np.sin(2*psis[2] - self.paramX5) +
                              2*self.paramE5*np.sin(2*psis[2] + self.paramX5) +
                              self.paramE5*np.sin(-psis[1] + psis[2] + self.paramX5) +
                              2*self.paramE5*np.sin(-psis[1] + 2*psis[2] + self.paramX5) +
                              4*self.paramE5*np.sin(psis[1] - 2*psis[2] + self.paramX5) +
                              5*self.paramE5*np.sin(psis[1] - psis[2] + self.paramX5) -
                              self.paramE5*np.sin(psis[1] + psis[2] - self.paramX5) +
                              2*self.paramE5*np.sin(psis[1] + psis[2] + self.paramX5) +
                              2*self.paramE5*np.sin(2*psis[1] - 2*psis[2] + self.paramX5) +
                              4*self.paramE5*np.sin(-2*psis[2] + psis[3] + self.paramX5) +
                              2*self.paramE5*np.sin(-2*psis[2] + 2*psis[3] + self.paramX5) +
                              5*self.paramE5*np.sin(-psis[2] + psis[3] + self.paramX5) +
                              self.paramE5*np.sin(psis[2] - psis[3] + self.paramX5) -
                              self.paramE5*np.sin(psis[2] + psis[3] - self.paramX5) +
                              2*self.paramE5*np.sin(psis[2] + psis[3] + self.paramX5) +
                              2*self.paramE5*np.sin(2*psis[2] - psis[3] + self.paramX5) +
                              2*self.paramE5*np.sin(-psis[1] + psis[2] + psis[3] + self.paramX5) +
                              4*self.paramE5*np.sin(psis[1] - 2*psis[2] + psis[3] + self.paramX5) +
                              2*self.paramE5*np.sin(psis[1] - psis[2] - psis[3] + self.paramX5) -
                              self.paramE5*np.sin(psis[1] + psis[2] - 2*psis[3] - self.paramX5) -
                              2*self.paramE5*np.sin(psis[1] + psis[2] - psis[3] - self.paramX5) +
                              2*self.paramE5*np.sin(psis[1] + psis[2] - psis[3] + self.paramX5) +
                              self.paramE5*np.sin(2*psis[1] - psis[2] - psis[3] + self.paramX5))/64

    def f2x3(self, psis):
        return self.paramEps*(self.paramE1*np.sin(psis[3] + self.paramX1) -
                              self.paramE1*np.sin(-psis[2] + psis[3] + self.paramX1) +
                              2*self.paramE2*np.sin(2*psis[3] + self.paramX2) -
                              2*self.paramE2*np.sin(-2*psis[2] + 2*psis[3] + self.paramX2))/4 + \
               self.paramEps*(2*self.paramE3*np.sin(psis[3] + self.paramX3) +
                              2*self.paramE3*np.sin(2*psis[3] + self.paramX3) +
                              2*self.paramE3*np.sin(psis[1] + psis[3] + self.paramX3) -
                              2*self.paramE3*np.sin(-2*psis[2] + psis[3] + self.paramX3) -
                              2*self.paramE3*np.sin(-2*psis[2] + 2*psis[3] + self.paramX3) -
                              2*self.paramE3*np.sin(-psis[2] + psis[3] + self.paramX3) +
                              2*self.paramE3*np.sin(psis[2] + psis[3] + self.paramX3) -
                              2*self.paramE3*np.sin(psis[1] - 2*psis[2] + psis[3] + self.paramX3))/16 + \
               self.paramEps*(self.paramE4*np.sin(psis[3] - self.paramX4) +
                              self.paramE4*np.sin(psis[3] + self.paramX4) +
                              2*self.paramE4*np.sin(2*psis[3] + self.paramX4) +
                              2*self.paramE4*np.sin(-psis[1] + 2*psis[3] + self.paramX4) -
                              self.paramE4*np.sin(2*psis[1] - psis[3] + self.paramX4) -
                              2*self.paramE4*np.sin(-2*psis[2] + 2*psis[3] + self.paramX4) -
                              self.paramE4*np.sin(-psis[2] + psis[3] + self.paramX4) +
                              self.paramE4*np.sin(psis[2] - psis[3] + self.paramX4) -
                              self.paramE4*np.sin(psis[2] + psis[3] - self.paramX4) -
                              self.paramE4*np.sin(2*psis[2] - psis[3] + self.paramX4) +
                              2*self.paramE4*np.sin(psis[1] + psis[2] - 2*psis[3] - self.paramX4) +
                              self.paramE4*np.sin(2*psis[1] - psis[2] - psis[3] + self.paramX4))/16 + \
               self.paramEps*(-self.paramE5*np.sin(psis[3] - self.paramX5) +
                              5*self.paramE5*np.sin(psis[3] + self.paramX5) +
                              2*self.paramE5*np.sin(2*psis[3] + self.paramX5) +
                              2*self.paramE5*np.sin(-psis[1] + 2*psis[3] + self.paramX5) +
                              2*self.paramE5*np.sin(psis[1] + psis[3] + self.paramX5) -
                              self.paramE5*np.sin(2*psis[1] - psis[3] + self.paramX5) -
                              2*self.paramE5*np.sin(-2*psis[2] + psis[3] + self.paramX5) -
                              2*self.paramE5*np.sin(-2*psis[2] + 2*psis[3] + self.paramX5) -
                              5*self.paramE5*np.sin(-psis[2] + psis[3] + self.paramX5) -
                              self.paramE5*np.sin(psis[2] - psis[3] + self.paramX5) -
                              self.paramE5*np.sin(psis[2] + psis[3] - self.paramX5) +
                              2*self.paramE5*np.sin(psis[2] + psis[3] + self.paramX5) -
                              self.paramE5*np.sin(2*psis[2] - psis[3] + self.paramX5) +
                              2*self.paramE5*np.sin(-psis[1] + psis[2] + psis[3] + self.paramX5) -
                              2*self.paramE5*np.sin(psis[1] - 2*psis[2] + psis[3] + self.paramX5) +
                              2*self.paramE5*np.sin(psis[1] - psis[2] - psis[3] + self.paramX5) +
                              2*self.paramE5*np.sin(psis[1] + psis[2] - 2*psis[3] - self.paramX5) +
                              2*self.paramE5*np.sin(psis[1] + psis[2] - psis[3] - self.paramX5) -
                              2*self.paramE5*np.sin(psis[1] + psis[2] - psis[3] + self.paramX5) +
                              self.paramE5*np.sin(2*psis[1] - psis[2] - psis[3] + self.paramX5))/64

    def f3x1(self, psis):
        return self.paramEps*(self.paramE1*np.sin(psis[1] + self.paramX1) -
                              self.paramE1*np.sin(psis[1] - psis[3] + self.paramX1) +
                              2*self.paramE2*np.sin(2*psis[1] + self.paramX2) -
                              2*self.paramE2*np.sin(2*psis[1] - 2*psis[3] + self.paramX2))/4 + \
               self.paramEps*(2*self.paramE3*np.sin(psis[1] + self.paramX3) +
                              2*self.paramE3*np.sin(2*psis[1] + self.paramX3) +
                              2*self.paramE3*np.sin(psis[1] + psis[2] + self.paramX3) -
                              2*self.paramE3*np.sin(psis[1] - 2*psis[3] + self.paramX3) -
                              2*self.paramE3*np.sin(psis[1] - psis[3] + self.paramX3) +
                              2*self.paramE3*np.sin(psis[1] + psis[3] + self.paramX3) -
                              2*self.paramE3*np.sin(2*psis[1] - 2*psis[3] + self.paramX3) -
                              2*self.paramE3*np.sin(psis[1] + psis[2] - 2*psis[3] + self.paramX3))/16 + \
               self.paramEps*(self.paramE4*np.sin(psis[1] - self.paramX4) +
                              self.paramE4*np.sin(psis[1] + self.paramX4) +
                              2*self.paramE4*np.sin(2*psis[1] + self.paramX4) -
                              self.paramE4*np.sin(-psis[1] + 2*psis[2] + self.paramX4) +
                              self.paramE4*np.sin(-psis[1] + psis[3] + self.paramX4) -
                              self.paramE4*np.sin(-psis[1] + 2*psis[3] + self.paramX4) -
                              self.paramE4*np.sin(psis[1] - psis[3] + self.paramX4) -
                              self.paramE4*np.sin(psis[1] + psis[3] - self.paramX4) +
                              2*self.paramE4*np.sin(2*psis[1] - psis[2] + self.paramX4) -
                              2*self.paramE4*np.sin(2*psis[1] - 2*psis[3] + self.paramX4) -
                              self.paramE4*np.sin(psis[1] - 2*psis[2] + psis[3] - self.paramX4) -
                              2*self.paramE4*np.sin(2*psis[1] - psis[2] - psis[3] + self.paramX4))/16 + \
               self.paramEps*(-self.paramE5*np.sin(psis[1] - self.paramX5) +
                              5*self.paramE5*np.sin(psis[1] + self.paramX5) +
                              2*self.paramE5*np.sin(2*psis[1] + self.paramX5) -
                              self.paramE5*np.sin(-psis[1] + 2*psis[2] + self.paramX5) -
                              self.paramE5*np.sin(-psis[1] + psis[3] + self.paramX5) -
                              self.paramE5*np.sin(-psis[1] + 2*psis[3] + self.paramX5) +
                              2*self.paramE5*np.sin(psis[1] + psis[2] + self.paramX5) -
                              2*self.paramE5*np.sin(psis[1] - 2*psis[3] + self.paramX5) -
                              5*self.paramE5*np.sin(psis[1] - psis[3] + self.paramX5) -
                              self.paramE5*np.sin(psis[1] + psis[3] - self.paramX5) +
                              2*self.paramE5*np.sin(psis[1] + psis[3] + self.paramX5) +
                              2*self.paramE5*np.sin(2*psis[1] - psis[2] + self.paramX5) -
                              2*self.paramE5*np.sin(2*psis[1] - 2*psis[3] + self.paramX5) -
                              2*self.paramE5*np.sin(-psis[1] + psis[2] + psis[3] + self.paramX5) -
                              self.paramE5*np.sin(psis[1] - 2*psis[2] + psis[3] - self.paramX5) -
                              2*self.paramE5*np.sin(psis[1] - psis[2] - psis[3] + self.paramX5) -
                              2*self.paramE5*np.sin(psis[1] - psis[2] + psis[3] - self.paramX5) +
                              2*self.paramE5*np.sin(psis[1] - psis[2] + psis[3] + self.paramX5) -
                              2*self.paramE5*np.sin(psis[1] + psis[2] - 2*psis[3] + self.paramX5) -
                              2*self.paramE5*np.sin(2*psis[1] - psis[2] - psis[3] + self.paramX5))/64

    def f3x2(self, psis):
        return self.paramEps*(self.paramE1*np.sin(psis[2] + self.paramX1) -
                              self.paramE1*np.sin(psis[2] - psis[3] + self.paramX1) +
                              2*self.paramE2*np.sin(2*psis[2] + self.paramX2) -
                              2*self.paramE2*np.sin(2*psis[2] - 2*psis[3] + self.paramX2))/4 + \
               self.paramEps*(2*self.paramE3*np.sin(psis[2] + self.paramX3) +
                              2*self.paramE3*np.sin(2*psis[2] + self.paramX3) +
                              2*self.paramE3*np.sin(psis[1] + psis[2] + self.paramX3) -
                              2*self.paramE3*np.sin(psis[2] - 2*psis[3] + self.paramX3) -
                              2*self.paramE3*np.sin(psis[2] - psis[3] + self.paramX3) +
                              2*self.paramE3*np.sin(psis[2] + psis[3] + self.paramX3) -
                              2*self.paramE3*np.sin(2*psis[2] - 2*psis[3] + self.paramX3) -
                              2*self.paramE3*np.sin(psis[1] + psis[2] - 2*psis[3] + self.paramX3))/16 + \
               self.paramEps*(self.paramE4*np.sin(psis[2] - self.paramX4) +
                              self.paramE4*np.sin(psis[2] + self.paramX4) +
                              2*self.paramE4*np.sin(2*psis[2] + self.paramX4) +
                              2*self.paramE4*np.sin(-psis[1] + 2*psis[2] + self.paramX4) -
                              self.paramE4*np.sin(2*psis[1] - psis[2] + self.paramX4) +
                              self.paramE4*np.sin(-psis[2] + psis[3] + self.paramX4) -
                              self.paramE4*np.sin(-psis[2] + 2*psis[3] + self.paramX4) -
                              self.paramE4*np.sin(psis[2] - psis[3] + self.paramX4) -
                              self.paramE4*np.sin(psis[2] + psis[3] - self.paramX4) -
                              2*self.paramE4*np.sin(2*psis[2] - 2*psis[3] + self.paramX4) +
                              2*self.paramE4*np.sin(psis[1] - 2*psis[2] + psis[3] - self.paramX4) +
                              self.paramE4*np.sin(2*psis[1] - psis[2] - psis[3] + self.paramX4))/16 + \
               self.paramEps*(-self.paramE5*np.sin(psis[2] - self.paramX5) +
                              5*self.paramE5*np.sin(psis[2] + self.paramX5) +
                              2*self.paramE5*np.sin(2*psis[2] + self.paramX5) +
                              2*self.paramE5*np.sin(-psis[1] + 2*psis[2] + self.paramX5) +
                              2*self.paramE5*np.sin(psis[1] + psis[2] + self.paramX5) -
                              self.paramE5*np.sin(2*psis[1] - psis[2] + self.paramX5) -
                              self.paramE5*np.sin(-psis[2] + psis[3] + self.paramX5) -
                              self.paramE5*np.sin(-psis[2] + 2*psis[3] + self.paramX5) -
                              2*self.paramE5*np.sin(psis[2] - 2*psis[3] + self.paramX5) -
                              5*self.paramE5*np.sin(psis[2] - psis[3] + self.paramX5) -
                              self.paramE5*np.sin(psis[2] + psis[3] - self.paramX5) +
                              2*self.paramE5*np.sin(psis[2] + psis[3] + self.paramX5) -
                              2*self.paramE5*np.sin(2*psis[2] - 2*psis[3] + self.paramX5) +
                              2*self.paramE5*np.sin(-psis[1] + psis[2] + psis[3] + self.paramX5) +
                              2*self.paramE5*np.sin(psis[1] - 2*psis[2] + psis[3] - self.paramX5) +
                              2*self.paramE5*np.sin(psis[1] - psis[2] - psis[3] + self.paramX5) +
                              2*self.paramE5*np.sin(psis[1] - psis[2] + psis[3] - self.paramX5) -
                              2*self.paramE5*np.sin(psis[1] - psis[2] + psis[3] + self.paramX5) -
                              2*self.paramE5*np.sin(psis[1] + psis[2] - 2*psis[3] + self.paramX5) +
                              self.paramE5*np.sin(2*psis[1] - psis[2] - psis[3] + self.paramX5))/64

    def f3x3(self, psis):
        return self.paramEps*(-self.paramE1*np.sin(psis[3] - self.paramX1) +
                              self.paramE1*np.sin(psis[3] + self.paramX1) +
                              self.paramE1*np.sin(psis[1] - psis[3] + self.paramX1) +
                              self.paramE1*np.sin(psis[2] - psis[3] + self.paramX1) -
                              2*self.paramE2*np.sin(2*psis[3] - self.paramX2) +
                              2*self.paramE2*np.sin(2*psis[3] + self.paramX2) +
                              2*self.paramE2*np.sin(2*psis[1] - 2*psis[3] + self.paramX2) +
                              2*self.paramE2*np.sin(2*psis[2] - 2*psis[3] + self.paramX2))/4 + \
               self.paramEps*(-2*self.paramE3*np.sin(psis[3] - self.paramX3) +
                              2*self.paramE3*np.sin(psis[3] + self.paramX3) -
                              2*self.paramE3*np.sin(2*psis[3] - self.paramX3) +
                              2*self.paramE3*np.sin(2*psis[3] + self.paramX3) +
                              4*self.paramE3*np.sin(psis[1] - 2*psis[3] + self.paramX3) +
                              2*self.paramE3*np.sin(psis[1] - psis[3] + self.paramX3) +
                              2*self.paramE3*np.sin(psis[1] + psis[3] + self.paramX3) +
                              2*self.paramE3*np.sin(2*psis[1] - 2*psis[3] + self.paramX3) +
                              4*self.paramE3*np.sin(psis[2] - 2*psis[3] + self.paramX3) +
                              2*self.paramE3*np.sin(psis[2] - psis[3] + self.paramX3) +
                              2*self.paramE3*np.sin(psis[2] + psis[3] + self.paramX3) +
                              2*self.paramE3*np.sin(2*psis[2] - 2*psis[3] + self.paramX3) +
                              4*self.paramE3*np.sin(psis[1] + psis[2] - 2*psis[3] + self.paramX3))/16 + \
               self.paramEps*(-2*self.paramE4*np.sin(2*psis[3] - self.paramX4) +
                              2*self.paramE4*np.sin(2*psis[3] + self.paramX4) -
                              self.paramE4*np.sin(-psis[1] + psis[3] + self.paramX4) +
                              2*self.paramE4*np.sin(-psis[1] + 2*psis[3] + self.paramX4) +
                              self.paramE4*np.sin(psis[1] - psis[3] + self.paramX4) -
                              self.paramE4*np.sin(psis[1] + psis[3] - self.paramX4) +
                              2*self.paramE4*np.sin(2*psis[1] - 2*psis[3] + self.paramX4) -
                              self.paramE4*np.sin(-psis[2] + psis[3] + self.paramX4) +
                              2*self.paramE4*np.sin(-psis[2] + 2*psis[3] + self.paramX4) +
                              self.paramE4*np.sin(psis[2] - psis[3] + self.paramX4) -
                              self.paramE4*np.sin(psis[2] + psis[3] - self.paramX4) +
                              2*self.paramE4*np.sin(2*psis[2] - 2*psis[3] + self.paramX4) -
                              self.paramE4*np.sin(psis[1] - 2*psis[2] + psis[3] - self.paramX4) +
                              self.paramE4*np.sin(2*psis[1] - psis[2] - psis[3] + self.paramX4))/16 + \
               self.paramEps*(-6*self.paramE5*np.sin(psis[3] - self.paramX5) +
                              6*self.paramE5*np.sin(psis[3] + self.paramX5) -
                              2*self.paramE5*np.sin(2*psis[3] - self.paramX5) +
                              2*self.paramE5*np.sin(2*psis[3] + self.paramX5) +
                              self.paramE5*np.sin(-psis[1] + psis[3] + self.paramX5) +
                              2*self.paramE5*np.sin(-psis[1] + 2*psis[3] + self.paramX5) +
                              4*self.paramE5*np.sin(psis[1] - 2*psis[3] + self.paramX5) +
                              5*self.paramE5*np.sin(psis[1] - psis[3]+ self.paramX5) -
                              self.paramE5*np.sin(psis[1] + psis[3] - self.paramX5) +
                              2*self.paramE5*np.sin(psis[1] + psis[3] + self.paramX5) +
                              2*self.paramE5*np.sin(2*psis[1] - 2*psis[3] + self.paramX5) +
                              self.paramE5*np.sin(-psis[2] + psis[3] + self.paramX5) +
                              2*self.paramE5*np.sin(-psis[2] + 2*psis[3] + self.paramX5) +
                              4*self.paramE5*np.sin(psis[2] - 2*psis[3] + self.paramX5) +
                              5*self.paramE5*np.sin(psis[2] - psis[3] + self.paramX5) -
                              self.paramE5*np.sin(psis[2] + psis[3] - self.paramX5) +
                              2*self.paramE5*np.sin(psis[2] + psis[3] + self.paramX5) +
                              2*self.paramE5*np.sin(2*psis[2] - 2*psis[3] + self.paramX5) +
                              2*self.paramE5*np.sin(-psis[1] + psis[2] + psis[3] + self.paramX5) -
                              self.paramE5*np.sin(psis[1] - 2*psis[2] + psis[3] - self.paramX5) +
                              2*self.paramE5*np.sin(psis[1] - psis[2] - psis[3] + self.paramX5) -
                              2*self.paramE5*np.sin(psis[1] - psis[2] + psis[3] - self.paramX5) +
                              2*self.paramE5*np.sin(psis[1] - psis[2] + psis[3] + self.paramX5) +
                              4*self.paramE5*np.sin(psis[1] + psis[2] - 2*psis[3] + self.paramX5) +
                              self.paramE5*np.sin(2*psis[1] - psis[2] - psis[3] + self.paramX5))/64


    def getRestrictionJac(self,X):
        x, y = X
        psis = [0., x, y, 0.]
        return np.array([[self.f1x1(psis), self.f1x2(psis)],
                         [ self.f2x1(psis),self.f2x2(psis)]])
    
    def getReducedSystemJac(self,X):  
        x, y, z = X
        psis = [0., x, y, z]
        return np.array([[self.f1x1(psis), self.f1x2(psis),self.f1x3(psis)],
                         [ self.f2x1(psis),self.f2x2(psis),self.f2x3(psis)],
                         [ self.f3x1(psis),self.f3x2(psis),self.f3x3(psis)]
                        ])
    
    def getParams(self):          
        return [self.paramE1,self.paramE2,self.paramE3,self.paramE4, self.paramE5, self.paramX1, self.paramX2,
                self.paramX3, self.paramX4, self.paramX5. self.paramEps]

#sym = sp.symbols('self.paramE1, self.paramE2, self.paramE3, self.paramE4, self.paramE5, self.paramX1, self.paramX2, self.paramX3, self.paramX4, self.paramX5, self.paramEps, psis[1], psis[2], psis[3]')
#sys = FivePhaseOscillators(sym[0], sym[1], sym[2], sym[3], sym[4], sym[5], sym[6], sym[7], sym[8], sym[9], sym[10])
#syst = sys.getSystemS()
#print(sp.diff(syst[3], sym[13]))