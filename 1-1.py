import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
import time
from sklearn.model_selection import train_test_split
import numpy as np

from collections import OrderedDict     #順序付きDict
import matplotlib.pyplot as plt     #グラフのライブラリ

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.reshape(60000, 28*28)
X_test  = X_test.reshape(10000, 28*28)

# 10,000件のデータを 80%:20% に分割
trData, testData, trLabels, testLabels = train_test_split(np.array(X_test), y_test, test_size=0.2, random_state=10)
 

    
''' NN法のモデル生成  
ここで､Kの値を敢えてrange(1,21)で1から20にする､
図を生成するために､実行時間とスコアを格納するdicを作る｡
'''

scores = OrderedDict()
durations = OrderedDict()

for n_neighbors in range(1,21): 
    start = time.time()  # 時間計測開始
    model = KNeighborsClassifier(int(n_neighbors))
    model.fit(trData, trLabels)  # 学習データ8,000件の配置

    # テスト用データ2,000件に対してNN法を実行
    score = model.score(testData, testLabels)
    duration = time.time() - start  # 計測終了
    scores[str(n_neighbors)] = score * 100
    durations[str(n_neighbors)] =duration
    
    print("Time: %.2f [s]" % duration)
    print("accuracy=%.2f%%" % (score * 100))
    
def graph_show_accuracy(scores):       # 図を生成するファンクション  
    plt.plot(scores.keys(),scores.values(),linewidth = 2)
    plt.xlabel("K")
    plt.ylabel("Accuracy[%]")
    plt.show()

def graph_show_time(durations):
    plt.plot(durations.keys(),durations.values(),linewidth = 2)
    plt.xlabel("K")
    plt.ylabel("Time[s]")
    plt.show()

graph_show_accuracy(scores)
graph_show_time(durations)

