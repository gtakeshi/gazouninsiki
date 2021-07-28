import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
import time
from sklearn.model_selection import train_test_split
import numpy as np

from collections import OrderedDict  # 順序付きDict
import matplotlib.pyplot as plt  # グラフのライブラリ

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

scores = OrderedDict()
"""次の処理はデータの正規化"""
X_60000, X_10000 = ((X_train.reshape(60000, 28 * 28)).astype(np.float64),
                    (X_test.reshape(10000, 28 * 28)).astype(np.float64))

X_60000 /= X_60000.max()
X_10000 /= X_10000.max()
"""事前に､テストデータの量を決める"""
tsData, tsData2, tsLabel, tsLable2 = train_test_split(np.array(X_10000), y_test, test_size=0.5, random_state=10)

trData, trData2, trLabels, trLabels2 = train_test_split(np.array(X_60000), y_train, test_size=0.1,
                                                        random_state=10)
''' NN法のモデル生成  
ここで､Kの値を敢えてrange(1,21)で1から20にする､
図を生成するために､実行時間とスコアを格納するdicを作る｡
'''

scores = OrderedDict()
durations = OrderedDict()

for n_neighbors in range(1, 21):
    start = time.time()  # 時間計測開始
    model = KNeighborsClassifier(int(n_neighbors))
    model.fit(trData, trLabels)  # 学習データ5,000件の配置

    # テスト用データ2,000件に対してNN法を実行
    score = model.score(tsData, tsLabel)
    duration = time.time() - start  # 計測終了
    scores[str(n_neighbors)] = score * 100
    durations[str(n_neighbors)] = duration

    print("Time: %.2f [s]" % duration)
    print("accuracy=%.2f%%" % (score * 100))


def graph_show_accuracy(scores):  # 図を生成するファンクション
    plt.plot(scores.keys(), scores.values(), linewidth=2)
    plt.xlabel("K")
    plt.ylabel("Accuracy[%]")
    plt.show()


def graph_show_time(durations):
    plt.plot(durations.keys(), durations.values(), linewidth=2)
    plt.xlabel("K")
    plt.ylabel("Time[s]")
    plt.show()


graph_show_accuracy(scores)
graph_show_time(durations)


