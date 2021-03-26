import numpy as np
import matplotlib.pyplot as plt


def saveHeteroclinicsDataAsTxt(HeteroclinicsData, pathToDir, fileName ):
    """
    (i, j, a, b, dist)
    """
    if HeteroclinicsData:
        headerStr = (
                'i  j  alpha  beta  dist  startPtX  startPtY  startPtZ\n' +
                '0  1  2      3     4     5         6         7')
        fmtList = ['%2u',
                   '%2u',
                   '%+18.15f',
                   '%+18.15f',
                   '%+18.15f',
                   '%+18.15f',
                   '%+18.15f',
                   '%+18.15f',]
        np.savetxt("{}{}.txt".format(pathToDir,fileName), HeteroclinicsData, header=headerStr,
                   fmt=fmtList)

def prepareHeteroclinicsData(data):
    """
        Accepts result of running heteroclinics analysis on grid.
        Expects elements to be tuples in form (i, j, a, b, result)
        """
    HeteroclinicsData=[]
    sortedData = sorted(data, key=lambda X: (X[0], X[1]))
    for d in sortedData:
        if d[4]:
            for dt in d[4]:
            #dict = min(d[4], key=lambda i: i['dist'])
                Xpt,Ypt,Zpt = dt['stPt']
                HeteroclinicsData.append((d[0], d[1], d[2], d[3], dt['dist'], Xpt,Ypt,Zpt))

    return HeteroclinicsData

def plotHeteroclinicsData(heteroclinicsData, firstParamInterval ,secondParamInterval, imageName):
    """
    (i, j, a, b, dist)
    """
    N = len(firstParamInterval)
    M = len(secondParamInterval)

    colorGridDist = np.zeros((M, N))

    for data in heteroclinicsData:
        i = data[0]
        j = data[1]
        colorGridDist[j][i] = 1

    plt.pcolormesh(firstParamInterval, secondParamInterval, colorGridDist, cmap=plt.cm.get_cmap('RdBu'))
    plt.colorbar()
    plt.savefig(imageName)

