import os
import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def readData(exp):
    with open('./實驗' + exp + '/train_data.csv', newline = '') as train:
        train_data = list(csv.reader(train))[1:]
    with open('./實驗' + exp +'/test_data.csv', newline= '') as test:
        test_data = list(csv.reader(test))[1:]
    return train_data, test_data

def writeResult(exp, clf, result):
    try:
        os.mkdir('實驗' + exp + '結果')
    except FileExistsError:
        pass
    df = pd.read_csv('./實驗' + exp + '/test_data.csv')
    df['Prediction'] = result
    df.to_csv('./實驗' + exp + '結果/' + clf + '.csv')
    print("\n", clf + ".csv created!", sep='')

def toNumber(data):
    return list(list((float(x) if i == 5 or i == 6 else int(x)) for i, x in enumerate(d)) for d in data)

def splitLable(data):
    return map(list, zip(*[(d[:-1], d[-1]) for d in data]))

def normalize(data, maxmin):
    return list(map(list, zip(*map(lambda x, mm: [(i - mm[1]) / (mm[0] - mm[1]) for i in x], map(list, zip(*data)), maxmin))))

def performance(clf, result, test_label):
    type1, type2, realPos, predictPos = 0, 0, 0, 0
    for r, t in zip(result, test_label):
        type1 += (t and r != t)
        type2 += (not t and r != t)
        realPos += t
        predictPos += r
        
    print("accuracy: " , a := (100 - round((type1 + type2) / len(test_label) * 100)), "%",  sep='')
    print("precision: ", p := (100 - round(type2 / (predictPos if predictPos else 1) * 100)), "%",  sep='')
    print("recall: ", r := (100 - round(type1 / (realPos if realPos else 1) * 100)), "%",  sep='')
    return pd.DataFrame({'Classifier':[clf, clf, clf], 'Performance Metric':['Accuracy', 'Precision', 'Recall'], 'Percentage':[a, p, r]})

def preprocessor(exp, Classification):
    # 讀檔案
    train_data, test_data = readData(exp)

    # 轉數字
    train_data = toNumber(train_data)
    test_data =  toNumber(test_data)

    # 分開標籤
    train_data, train_label = splitLable(train_data)
    test_data, test_label = splitLable(test_data)

    # 正規化
    if Classification == 'KNN':
        MaxMin = list(map(lambda x: [max(x), min(x)], map(list, zip(*train_data))))
        train_data = normalize(train_data, MaxMin)
        test_data = normalize(test_data, MaxMin)
    
    return train_data, train_label, test_data, test_label

def statisticalResults(bar, exp, Classification):
    ax = sns.barplot(x = 'Classifier', y = 'Percentage', hue = 'Performance Metric', data = bar)
    if Classification == 'KNN':
        ax.set(ylim = (30, 88), xlabel = 'K')
    else:
        ax.set(ylim = (0, 100))
    ax.set_title('Comparison of ' + Classification + ' Performance (EXP ' + exp + ')')
    sns.move_legend(ax, "upper center", bbox_to_anchor = (.5, 1), ncol = 3, title = None, frameon = False)
    plt.show()