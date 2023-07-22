import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def valuecopy(data):
    return list(term for term in data)

def readCSV(filename):
    """CSV讀檔"""
    with open(filename + '.csv', newline = '') as train:
        train_data = list(csv.reader(train))[1:]
    return train_data

def Transform(data, Type = 'float'):
    """若為data為資料則轉成float，若data為標籤則轉成str"""
    result = list(list((float(x) if Type == 'float' else str(x)) for i, x in enumerate(d[1:])) for d in data)
    temp = result
    if Type == 'str':
        result = list(term[0] for term in temp)
    return result

def writeResult(clf, result):
    df =  pd.read_csv('test_label.csv')
    df['Classifier'] = result[0]
    df['K-Means'] = result[1]
    df['DBSCAN'] = result[2]
    df.to_csv(clf+'.csv')
    print('\n'+clf+'.csv create!')

def performance(clf, resultOfClassifier, resultOfkmeans,resultOfDBSCAN, testLabel):
    diffOfClas, diffOfkmeans, diffOfDBSCAN = 0, 0, 0
    for r, t in zip(resultOfClassifier, testLabel):
        diffOfClas += r!=t
    for r, t in zip(resultOfkmeans, testLabel):
        diffOfkmeans += r!= t
    for r, t in zip(resultOfDBSCAN, testLabel):
        diffOfDBSCAN += r!= t
    print(clf)
    print("            classifier: ", a:=(100 - round((diffOfClas)/len(testLabel)*100)), "%", sep = '')
    print("accuracy of Kmeans: ", b:=(100 - round((diffOfkmeans)/len(testLabel)*100)), "%", sep = '')
    print("            DBSCAN: ", c:=(100 - round((diffOfDBSCAN)/len(testLabel)*100)), "%", sep = '')

    return pd.DataFrame({'Classifier':[clf,clf,clf], 'Cluster':['None','K-means', 'DBSCAN'], 'Accuracy(%)':[a,b,c]})

def statisticalResults(bar, Classification):
    sns.set(style="whitegrid")
    ax = sns.barplot(x = 'Cluster', y = 'Accuracy(%)', hue = 'Classifier', data = bar)
    ax.set(ylim = (0, 110))
    ax.set_title('Comparison of ' + Classification + ' Performance')
    sns.move_legend(ax, "upper left", ncol = 1, title = 'Classifier', frameon = True)
    plt.show()