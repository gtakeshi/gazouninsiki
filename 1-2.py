import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
import time
from sklearn.model_selection import train_test_split
import numpy as np

from collections import OrderedDict     #順序付きDict
import matplotlib.pyplot as plt     #グラフのライブラリ

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
# X_train = X_train.reshape(60000, 28*28)
# X_test  = X_test.reshape(10000, 28*28)  

'''
ここで､データの総数が多ければ多いほどいいと思って､
X_trainとX_testをまとめる｡
データ数が70000になる
'''
X_all = np.append(X_test,X_train)
X_all = X_all.reshape(70000,28*28)
y_all = np.append(y_test,y_train)

'''
まとめた70000件のデータの分割
ここで､test_sizesというLISTを作って､
10%から90%まで10%ずつの数値を入れておく
for関数を用いて､test_sizeを自動的に変化させる
'''
#test_sizes = [0.7,0.8]
test_sizes = [x/10 for x in list(range(1,10))]    # rangeを用いて､testデータの割合のリスト
scores = OrderedDict()                             #[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]を生成する
durations = OrderedDict()                           #ここで､size自動的に変更させたあとの精度を格納する場所

#print(test_sizes)
for test_size in test_sizes:
    
    trData, testData, trLabels, testLabels = train_test_split(np.array(X_all), y_all, test_size = test_size, random_state=10)    # test_size = test_sizeしなきゃいけない､しないとエラーが出るが､理由は不明
     
    ''' NN法のモデル生成  
    ここで､Kの値を第1問の結果 k=3 にする､
    図を生成するために､実行時間とスコアを格納するdicを作る｡
    '''

    start = time.time()  # 時間計測開始
    model = KNeighborsClassifier(n_neighbors = 3)      #ここで第1問の結果 k = 3にする
    model.fit(trData, trLabels)  # 学習データの配置

    # テスト用データに対してNN法を実行
    score = model.score(testData, testLabels)
    duration = time.time() - start  # 計測終了
    scores[str(round((1 - test_size) * 100 , 1))] = score * 100
    durations[str(round((1 - test_size) * 100 , 1))] = duration            #整型要注意!  ここround()しないと､30.000000000000004が出てくる!!!!!!!!!!
    
    print("train_size/test_size: " + str(round((1 - test_size) * 100 , 1)) + "/" + str(test_size * 100))
    print("Time: %.2f [s]" % duration)
    print("accuracy=%.2f%%" % (score * 100))

def graph_show_time(durations):            # グラフを生成するファンクション 
    plt.plot(durations.keys() ,durations.values(),linewidth = 2)
    plt.xlabel("Percentage of Train data[%]")
    plt.ylabel("Time[s]")
    plt.show()

def graph_show_after_change_test_size(scores):    
    plt.plot(scores.keys() ,scores.values(),linewidth = 2)
    plt.xlabel("Percentage of Train data[%]")
    plt.ylabel("Accuracy[%]")
    plt.show()

graph_show_after_change_test_size(scores)
graph_show_time(durations)

