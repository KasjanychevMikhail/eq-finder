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

    def funcG2s(self, phi):
        return self.paramE1 * sp.cos(phi + self.paramX1) + self.paramE2 * sp.cos(2 * phi + self.paramX2)

    def funcG3s(self, phi):
        return self.paramE3 * sp.cos(phi + self.paramX3)

    def funcG4s(self, phi):
        return self.paramE4 * sp.cos(phi + self.paramX4)

    def funcG5s(self, phi):
        return self.paramE5 * sp.cos(phi + self.paramX5)

    def sum1s(self, j):
        psis = sp.symbols('x, y, z')
        psis = [0.] + psis
        tmp = 0
        for k in range(3):
            tmp += self.funcG2s(psis[k] - psis[j]) - self.funcG2s(psis[k])
        return tmp[1:]

    def sum2s(self, j):
        psis = sp.symbols('x, y, z')
        psis = [0.] + psis
        tmp = 0
        for k in range(3):
            for l in range(3):
                tmp += self.funcG3s(psis[k] + psis[l] - 2 * psis[j]) - self.funcG3s(psis[k] + psis[l])
        return tmp[1:]

    def sum3s(self, j):
        psis = sp.symbols('x, y, z')
        psis = [0.] + psis
        tmp = 0
        for k in range(3):
            for l in range(3):
                tmp += self.funcG4s(2 * psis[k] - psis[l] - psis[j]) - self.funcG4s(2 * psis[k] - psis[l])
        return tmp[1:]

    def sum4s(self, j):
        psis = sp.symbols('x, y, z')
        psis = [0.] + psis
        tmp = 0
        for k in range(3):
            for l in range(3):
                for m in range(3):
                    tmp += self.funcG5s(psis[k] + psis[l] - psis[m] - psis[j]) - self.funcG5s(psis[k] + psis[l] - psis[m])
        return tmp[1:]

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

    def getSystemS(self):
        res = []
        for i in range(3):
            res[i] = self.paramEps / 4 * self.sum1s(i) + self.paramEps / 16 * self.sum2s(i) + \
                     self.paramEps / 16 * self.sum3s(i) + self.paramEps / 64 * self.sum4s(i)
        return res

    def getRestriction(self,psi):       
        psi = list(psi)      
        gammas = [0] + psi
        rhsPsi = self.getReducedSystem(gammas)
        return rhsPsi[1:]
    
    def DiagComponentJac2d(self, x, y):
        psis = sp.symbols('x, y, z')
        sys = self.getSystemS()

        return sp.diff(sys[0], psis[0]).subs([psis[0], psis[1], psis[2]], [x, y, 0.])

    def NotDiagComponentJac(self, x,y):
        psis = sp.symbols('x, y, z')
        sys = self.getSystemS()

        return sp.diff(sys[1], psis[0]).subs([psis[0], psis[1], psis[2]], [x, y, 0.])
    
    def DiagComponentJac3d(self, x,y,z):
        psis = sp.symbols('x, y, z')
        sys = self.getSystemS()
        
        return sp.diff(sys[0], psis[0]).subs([psis[0], psis[1], psis[2]], [x, y, z])
        
    def getRestrictionJac(self,X):  
        x,y=X       
        return np.array([[self.DiagComponentJac2d(x,y), self.NotDiagComponentJac(x,y)],
                         [ self.NotDiagComponentJac(y,x),self.DiagComponentJac2d(y,x)]])
    
    def getReducedSystemJac(self,X):  
        x,y,z=X       
        return np.array([[self.DiagComponentJac3d(x,y,z), self.NotDiagComponentJac(x,y),self.NotDiagComponentJac(x,z)],
                         [ self.NotDiagComponentJac(y,x),self.DiagComponentJac3d(y,x,z),self.NotDiagComponentJac(y,z)],
                         [ self.NotDiagComponentJac(z,x),self.NotDiagComponentJac(z,y),self.DiagComponentJac3d(z,x,y)]
                        ])
    
    def getParams(self):          
        return [self.paramE1,self.paramE2,self.paramE3,self.paramE4, self.paramE5, self.paramX1, self.paramX2,
                self.paramX3, self.paramX4, self.paramX5. self.paramEps]