from tool import valuecopy,readCSV,Transform,writeResult,performance,statisticalResults
from classifier import RandomForest, getUnknownData, KNN
from clustering import kMeans,DBSCAN ,inputUnknowLabel, renewResult
import pandas as pd

trainData = Transform(readCSV('train_data'))
testData = Transform(readCSV('test_data'))
trainLabel = Transform(readCSV('train_label'), 'str')
testLabel = Transform(readCSV('test_label'), 'str')

bar = pd.DataFrame({'Classifier':[], 'Cluster':[], 'Accuracy(%)':[]})

K = 2
unknownpair = inputUnknowLabel(K)

#KNN
resultOfClassifierKnn = KNN(9, trainData, trainLabel, testData)
resultKNNKmeans = renewResult(K, valuecopy(resultOfClassifierKnn), kMeans(K,getUnknownData(valuecopy(resultOfClassifierKnn),testData)), unknownpair, testLabel)
resultKNNDBS = renewResult(K, valuecopy(resultOfClassifierKnn), DBSCAN(getUnknownData(valuecopy(resultOfClassifierKnn),testData), 200, 5), unknownpair, testLabel)
bar = pd.concat([bar, performance('KNN', resultOfClassifierKnn, resultKNNKmeans, resultKNNDBS, testLabel)], ignore_index = True)

#random forest
resultOfClassifierRF = RandomForest(trainData, trainLabel, testData)
resultRFKmeans = renewResult(K, valuecopy(resultOfClassifierRF), kMeans(K,getUnknownData(valuecopy(resultOfClassifierRF),testData)), unknownpair, testLabel)
resultRFDBS = renewResult(K, valuecopy(resultOfClassifierRF), DBSCAN(getUnknownData(valuecopy(resultOfClassifierRF),testData), 200, 5), unknownpair, testLabel)
bar = pd.concat([bar, performance('Random Forest', resultOfClassifierRF, resultRFKmeans, resultRFDBS, testLabel)], ignore_index = True)

opt = input("Output result as a csv file?(Y/N) ")
if(opt == 'Y' or opt == 'y'):
    writeResult('KNN result',[valuecopy(resultOfClassifierKnn),valuecopy(resultKNNKmeans),valuecopy(resultKNNDBS)])
    writeResult('RF result',[valuecopy(resultOfClassifierRF),valuecopy(resultRFKmeans),valuecopy(resultRFDBS)])

#畫圖
statisticalResults(bar, 'Cluster')

