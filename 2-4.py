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

plt.figure(figsize=(14,10))
"""誤認識画像出力"""

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
tsData, tsData2, tsLabel, tsLable2 = train_test_split(np.array(X_10000), y_test, test_size=0.2, random_state=10)

"""モデルの選択
    多層パーセプトロン"""
model = MLPClassifier(hidden_layer_sizes=(100,), activation='tanh')

"""学習開始"""
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

wrong_data = []   # 識別が誤った画像のINDEXを記録するリスト
y_predict = model.predict(tsData[:201])      #testData 200件に対する予測

for index in range(0,200):
    if y_predict[index] != tsLabel[index]:
        wrong_data.append(index)

for i in range(5):
    for j in range(10):
        if (1 + 10 * i + j) < len(wrong_data):         #画像を5X10の表で生成する
            ax = plt.subplot(5, 10, 1 + 10 * i + j)
            ax.set_axis_off()
            ax.set_title("Wrong: " + str(y_predict[wrong_data[10 * i + j]]))        # 誤った数値
            plt.imshow(tsData[wrong_data[10 * i + j]].reshape(28, 28).astype(np.float64))
plt.show()


