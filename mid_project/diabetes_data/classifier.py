import tool
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier

dic = {'DT':'decision_tree', 'RF':'random_forest', 'SGD':'stochastic_gradient_descent', 'MLP':'multilayer_perceptron'}

# 選擇實驗A、B
exp = input("which exp: ")

# 選擇是否將結果以檔案形式輸出
opt = input("是否將結果以檔案形式輸出(Y/N): ")

data = tool.preprocessor(exp,'classifier')
train_data, train_label, test_data, test_label = data[0], data[1], data[2], data[3]

bar = pd.DataFrame({'Classifier':[], 'Performance Metric':[], 'Percentage':[]})
for clf in dic:
    # 訓練
    if clf == 'DT':
        result = list(DecisionTreeClassifier(max_depth=5).fit(train_data,train_label).predict(test_data))
    elif clf == 'RF':
        result = list(RandomForestClassifier().fit(train_data, train_label).predict(test_data))
    elif clf == 'SGD':
        result = list(SGDClassifier(loss = 'log').fit(train_data, train_label).predict(test_data))
    elif clf == 'MLP':
        result = list(MLPClassifier(activation = 'logistic', max_iter = 1000).fit(train_data, train_label).predict(test_data))

    # 統計結果
    print("---------------------")
    print(clf, " : ", dic[clf])
    bar = pd.concat([bar, tool.performance(clf, result, test_label)], ignore_index = True)

    # 產生分類結果CSV檔
    if opt == 'Y' or opt == 'y':
        tool.writeResult(exp, dic[clf], result)

tool.statisticalResults(bar, exp, 'Classifier')