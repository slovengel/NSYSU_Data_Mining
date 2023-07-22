import tool
import pandas as pd
import heapq
from math import dist
from collections import Counter
from random import randrange

def classify(test, K):
    """分類：傳入的test_data依序和train_data計算距離，
    取出最小的K筆資料，回傳分類。如果和無法分類(數量相同)的話，回傳-1"""
    distance = list(map(lambda train: dist(test[:], train[:]), train_data))
    """計算兩筆資料的距離"""

    nearest = heapq.nsmallest(K, enumerate(distance), key=lambda x: x[1])
    _, nearestDistance = zip(*nearest)
    nearestClass = [train_label[i] for i in range(0, len(distance)) if distance[i] <= nearestDistance[-1]]
    countDic = Counter(nearestClass)
    count_max = countDic.most_common(1)[0][1]
    majority = [key for key, count in countDic.most_common() if count == count_max]

    return majority[0] if len(majority) == 1 else -1

def knn(k):
    "依序將test_data傳入classify中判斷分類，將分類結果回傳"
    return list(map(lambda t: classify(t, k), test_data))

# 選擇實驗A、B
exp = input("which exp:")

# 選擇是否將結果以檔案形式輸出
opt = input("是否將結果以檔案形式輸出(Y/N): ")

# 回傳結果
data = tool.preprocessor(exp,'classifier')
train_data, train_label, test_data, test_label = data[0], data[1], data[2], data[3]

bar = pd.DataFrame({'Classifier':[], 'Performance Metric':[], 'Percentage':[]})
#計算k = 1~9 的情況
for k in range(1, 10):
    result = knn(k)
    rand = 0
    # 如果值為-1表示無法分類，則隨機分配
    for i in range(len(result)):
        if result[i] == -1:
            rand += 1
            result[i] = randrange(2)
    # 統計結果
    print("---------------------")
    print("k = ", k, sep='')
    bar = pd.concat([bar, tool.performance(str(k), result, test_label)], ignore_index = True)
    print("random percentage: ", round(rand/len(result) * 100), "%",  sep='')
    
    # 產生分類結果CSV檔
    if opt == 'Y' or opt == 'y':
        tool.writeResult(exp, str(k) + "NN", result)

tool.statisticalResults(bar, exp, 'KNN')