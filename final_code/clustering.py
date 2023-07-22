from random import randrange
from math import dist
from tool import valuecopy

def newMean(K, labels, data):
    """計算新的平均"""
    mean = list([0.0] * len(data[0]) for i in range(K))
    count = [0] * K
    for i in range(len(labels)):
        for j in range(len(data[i])):
            mean[labels[i]][j] += data[i][j]
        count[labels[i]] += 1
    for i in range(len(mean[0])):
        for j in range(len(count)):
            mean[j][i]/=count[j]
    return mean

def kMeans(K, data):
    """進行kmeans分群"""
    mean = list(data[randrange(len(data))] for i in range(K))
    meanTemp = [[0.0] * len(data[0])] * K
    meanList = list(dist(meanTemp[i], mean[i]) for i in range(K))
    while sum(meanList):
        label = []
        for i in range(len(data)):
            distanceList = list(dist(mean[j], data[i]) for j in range(K))
            label.append(distanceList.index(min(distanceList)))
        meanTemp = mean
        mean = newMean(K, label, data)
        meanList = list(dist(meanTemp[i], mean[i]) for i in range(K))
    return label

def countNeighbors(Data, targetPoint, eps):
    neighbors = []
    for point in Data:
        if dist(targetPoint, point) < eps and targetPoint != point:
            neighbors.append(Data.index(point))
    return neighbors

def DBSCAN(Data, eps, minPts):
    count = 0
    dataLabel = [-1] * len(Data)
    for point in Data:
        if dataLabel[Data.index(point)] != -1:
            continue
        neighborsIndex = countNeighbors(Data, point, eps)
        if len(neighborsIndex) < minPts:
            dataLabel[Data.index(point)] = -2
            continue
        dataLabel[Data.index(point)] = count
        for index in neighborsIndex:
            if dataLabel[index] == -2:
                dataLabel[index] = count
            if dataLabel[index] != -1:
                continue
            dataLabel[index] = count
            subneighborsIndex = countNeighbors(Data,Data[index],eps)
            if len(subneighborsIndex) >= minPts:
                for i in subneighborsIndex:
                    if i not in neighborsIndex:
                        neighborsIndex.append(i)
        count += 1
    for i in range(len(dataLabel)):
        if dataLabel[i] == -2:
            dataLabel[i] = 'noise'
    return dataLabel

def inputUnknowLabel(K):
    unknownlabel = []
    unknownlabecount = []
    for i in range(K):
        outfromTrain = input("Input " + str(i+1) + " label: ")
        unknownlabel.append(outfromTrain)
        unknownlabecount.append([0] * K)
    return [unknownlabel, unknownlabecount]

def renewResult(K, result, unknownlabel, unknownpair, testLabel):
    unknownlabellist, unknownlabelcount = unknownpair[0], unknownpair[1]
    index = 0
    unknownlabellist = valuecopy(unknownlabellist)
    for i in range(len(unknownlabelcount)):
        for j in range(len(unknownlabelcount[i])):
            unknownlabelcount[i][j] = 0
    # 將原先的unknown改成分群之後的結果
    for i in range(len(result)):
        if(result[i] == 'unknown'):
            result[i] = unknownlabel[index]
            #統計新標籤與群心的相似度
            if testLabel[i] in unknownlabellist and str(type(result[i])) == "<class 'int'>":
                unknownlabelcount[unknownlabellist.index(testLabel[i])][result[i]] += 1
            index += 1
    #一統計表格後的建立對照扁
    unknownlabellist = valuecopy(unknownlabellist)
    # print(unknownlabelcount)
    for i in range(K):
        unknownlabellist.append(unknownlabelcount[i].index(max(unknownlabelcount[i])))
    # print(unknownlabellist)
    #更新結果
    for i in range(len(result)):
        if(str(type(result[i])) == "<class 'int'>"):
            for j in range(K):
                if result[i] == unknownlabellist[K+j]:
                    result[i] = unknownlabellist[j]
    return result