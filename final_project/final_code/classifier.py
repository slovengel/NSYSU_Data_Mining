from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from math import dist
import numpy as np
import heapq
from collections import Counter

def RandomForest(trainData,trainLabel,testData):
    RF = RandomForestClassifier().fit(trainData, trainLabel)
    result1 = list(RF.predict(testData))
    pro = list(RF.predict_proba(testData))
    return list(result1[i] if max(pro[i]) > 0.7 else 'unknown' for i in range(len(result1)))

def classifyByDistance(trainData, trainLabel,testData, K):
    """分類：傳入的test_data依序和train_data計算距離，
    取出最小的K筆資料，回傳分類。如果和無法分類(數量相同)的話，回傳-1"""
    #計算兩筆資料的距離
    distance = list(map(lambda train: dist(testData[:], train[:]), trainData))

    nearest = heapq.nsmallest(K, enumerate(distance), key=lambda x: x[1])
    _, nearestDistance = zip(*nearest)
    nearestClass = [trainLabel[i] for i in range(0, len(distance)) if distance[i] <= nearestDistance[-1]]
    countDic = Counter(nearestClass)
    count_max = countDic.most_common(1)[0][1]
    majority = [key for key, count in countDic.most_common() if count == count_max]
    if(sum(nearestDistance) > 200 * K + 100):
        majority[0] = 'unknown'
    return majority[0] if len(majority) == 1 else -1

def KNN(k, trainData, trainLabel, testData):
    """依序將test_data傳入classify中判斷分類，將分類結果回傳"""
    return list(map(lambda t: classifyByDistance(trainData, trainLabel, t, k), testData))

def getUnknownData(result, testData):
    unknownData = []
    for i in range(len(result)):
        if result[i] == 'unknown':
            unknownData.append(testData[i])
    return unknownData