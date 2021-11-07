import numpy as np

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

    def getRestriction(self,psi):       
        psi = list(psi)      
        gammas = [0] + psi
        rhsPsi = self.getReducedSystem(gammas)
        return rhsPsi[1:]
    
    def DiagComponentJac2d(self, x, y):
        M = [x, y]
        psisx1 = [0., 0., M[0], M[1] + 0.0001]
        psisy1 = [0., 0., M[0] + 0.0001, M[1]]
        psisx2 = [0., 0., M[0], M[1] - 0.0001]
        psisy2 = [0., 0., M[0] - 0.0001, M[1]]
        sys1 = self.getReducedSystem(psisx1)
        sys2 = self.getReducedSystem(psisy1)
        sys3 = self.getReducedSystem(psisx2)
        sys4 = self.getReducedSystem(psisy2)

        a = (sys1[2] - sys3[2]) / (2 * 0.0001)
        b = (sys2[2] - sys4[2]) / (2 * 0.0001)
        c = (sys1[3] - sys3[3]) / (2 * 0.0001)
        d = (sys2[3] - sys4[3]) / (2 * 0.0001)
        
        return a
    def NotDiagComponentJac(self, x,y):
        M = [x, y]
        psisx1 = [0., 0., M[0], M[1] + 0.0001]
        psisy1 = [0., 0., M[0] + 0.0001, M[1]]
        psisx2 = [0., 0., M[0], M[1] - 0.0001]
        psisy2 = [0., 0., M[0] - 0.0001, M[1]]
        sys1 = self.getReducedSystem(psisx1)
        sys2 = self.getReducedSystem(psisy1)
        sys3 = self.getReducedSystem(psisx2)
        sys4 = self.getReducedSystem(psisy2)

        a = (sys1[2] - sys3[2]) / (2 * 0.0001)
        b = (sys2[2] - sys4[2]) / (2 * 0.0001)
        c = (sys1[3] - sys3[3]) / (2 * 0.0001)
        d = (sys2[3] - sys4[3]) / (2 * 0.0001)
        return c
    
    def DiagComponentJac3d(self, x,y,z):
        
        
        return (
            (-np.cos(x + self.paramA) + 2*self.paramR * np.cos(2 * x + self.paramB)) +
                
            (-np.cos(x-y + self.paramA) + 2*self.paramR * np.cos(2 * (x-y) + self.paramB))+
                
            (-np.cos(x-z + self.paramA) + 2*self.paramR * np.cos(2 * (x-z) + self.paramB))-
                
            (np.cos((-x) + self.paramA) - 2*self.paramR * np.cos(2 * (-x) + self.paramB))
               )/4
        
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