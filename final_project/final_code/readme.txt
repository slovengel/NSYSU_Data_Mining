共有四個python檔案(.py)，分別是main.py、tool.py、classifier.py、clustering.py，放在gene expression cancer RNA-Seq Data Set之下，和資料標籤在同一層級。

只有main.py是可以執行的，執行後會呼叫classifier.py、clustering.py來進行分類和分群的運算，並透過tool.py進行檔案讀寫及結果輸出。

執行main.py後，會先讀取檔案資料(以這次的資料及大約需要十秒鐘)，之後需要輸入未知的分類標籤(COAD、PRAD)，接著會進行KNN+KMeans、KNN+DBSCAN、RF+KMeans、RF+DBSCAN四種運算(約需要兩分鐘)，接著要求輸入是否將結果輸出成檔案，輸入Y會在同一個資料層級產生KNN result.csv、RF result.csv兩個檔案。

輸出的結果中(KNN result.csv、RF result.csv)，row A為編號、row B為測試資料id、row C為分類器分類結果、row D為分類器+KMeans結果、row E為分類器+DBSCAN結果。

除了python本身之外，以上程式還有使用numpy、pandas、sklearn、seaborn、matplotlib這五個套件，因此若沒有這五個套件將無法執行以上程式。
