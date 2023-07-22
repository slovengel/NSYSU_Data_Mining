共有三個python檔案(.py)，分別是classifier.py、KNN.py、tool.py，放在diabetes_data的資料之下。

tool.py並非進行分類，是進行前處理與統計輸出的部分，因此無法執行。
執行classifier.py是進行LC(SGD)、DT、RF、MLP的分類。執行時需要輸入對哪個實驗進行分類(A/B)，以及是否需要將各筆資料的結果輸出(y/n or Y/N)。若輸入分別為A Y，那結果會輸至資料夾"實驗A結果"，若沒有這個資料夾則會創立該資料夾(共4種分類器，4種檔案)。
KNN.py和classifier.py輸入與執行的部分一樣，若選擇輸出各筆資料的話，那結果會輸出9的檔案(1NN.csv ~ 9NN.csv)。
除了python本身之外，以上程式還有使用numpy、pandas、sklearn、seaborn、matplotlib這五個套件，因此若沒有這五個套件將無法執行以上程式。