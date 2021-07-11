import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
import time
from collections import OrderedDict
"""活性化関数の変更"""

"""次の処理はMNISTデータをインポートする
    X_train 訓練データ
    y_train そのラベル
    X_test  テストデータ
    y_test  そのラベル"""
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

scores = OrderedDict()
"""次の処理はデータの正規化"""
X_60000 , X_10000 = ((X_train.reshape(60000,28*28)).astype(np.float64),
                     (X_test.reshape(10000,28*28)).astype(np.float64))

X_60000 /= X_60000.max()
X_10000 /= X_10000.max()
"""事前に､テストデータの量を決める"""
tsData, tsData2, tsLabel, tsLable2 = train_test_split(np.array(X_10000), y_test, test_size=0.5, random_state=10)


"""学習データの量が0.1から0.9に設定する"""
train_sizes = [x/10 for x in list(range(9,0,-1))]    # rangeを用いて､testデータの割合のリスト
# scores = OrderedDict()                             #0.9から0.1を生成する
# durations = OrderedDict()                           #ここで､size自動的に変更させたあとの精度を格納する場所

activations = ("tanh","relu","logistic")

"""学習開始"""
try :
    """ここで､自動的に活性化関数を変更"""
    for activation in activations:

        """モデルの選択
        多層パーセプトロン"""

        model = MLPClassifier(hidden_layer_sizes=(100,), activation=activation)
        trData, trData2, trLabels,trLabels2 = train_test_split(np.array(X_60000), y_train, test_size=0.4, random_state=10)
        start = time.time()

        model.fit(trData, trLabels)  

        duration = time.time() - start
        print("Volume of Train data: " + str(len(trData)))
        print('Time: %.3f [s]' % (duration))
        #print( model.classes_ )
        print( 'Number of Layers: ', model.n_layers_ )
        print( 'Loss: ', model.loss_ )
        print( 'Iterations: ', model.n_iter_ )

        # テストデータでクラス（数字）予測
        predictions = model.predict( tsData )
        score = model.score(tsData , tsLabel) * 100
        scores[str(activation)] = score 
        print("accuracy=%.2f%%" % score)
        print("Method: {}".format(activation))
        # 識別レポート表示
        print( confusion_matrix(tsLabel, predictions) )  # 混同行列
        print( classification_report(tsLabel, predictions) )
except : 
    pass

def graphprint(scores):    
    plt.plot(scores.keys() ,scores.values(),linewidth = 2)
    plt.xlabel("Activation Method")
    plt.ylabel("Accuracy[%]")
    plt.show()
    
graphprint(scores)
