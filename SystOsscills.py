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
    
    def JacComp1(self, x,y):
        
        
        return (2*(-np.cos(x + self.paramA) + 2*self.paramR * np.cos(2 * x + self.paramB)) +
                (-np.cos(x-y + self.paramA) + 2*self.paramR * np.cos(2 * (x-y) + self.paramB)) -
                (np.cos((-x) + self.paramA) - 2*self.paramR * np.cos(2 * (-x) + self.paramB))
               )
    def JacComp2(self, x,y):
        
        return ((np.cos(x-y + self.paramA) - 2*self.paramR * np.cos(2 * (x-y) + self.paramB)) -
                (np.cos((-y) + self.paramA) - 2*self.paramR * np.cos(2 * (-y) + self.paramB))
               )
    
    def JacComp3(self, x,y,z):
        
        
        return (
            2*(-np.cos(x + self.paramA) + 2*self.paramR * np.cos(2 * x + self.paramB)) +
                
            (-np.cos(x-y + self.paramA) + 2*self.paramR * np.cos(2 * (x-y) + self.paramB))+
                
            (-np.cos(x-z + self.paramA) + 2*self.paramR * np.cos(2 * (x-z) + self.paramB))-
                
            (np.cos((-x) + self.paramA) - 2*self.paramR * np.cos(2 * (-x) + self.paramB))
               )    
        
    def getRestrictionJac(self,X):  
        x,y=X       
        return np.array([[self.JacComp1(x,y), self.JacComp2(x,y)],[ self.JacComp2(y,x),self.JacComp1(y,x)]])
    
    def getReducedSystemJac(self,X):  
        x,y,z=X       
        return np.array([[self.JacComp3(x,y,z), self.JacComp2(x,y),self.JacComp2(x,z)],
                         [ self.JacComp2(y,x),self.JacComp3(y,x,z),self.JacComp2(y,z)],
                         [ self.JacComp2(z,x),self.JacComp2(z,y),self.JacComp3(z,x,y)]
                        ])
    
    def getParams(self):          
        return [self.paramW,self.paramA,self.paramB,self.paramR]