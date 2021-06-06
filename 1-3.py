import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
import time
from sklearn.model_selection import train_test_split
import numpy as np

import matplotlib.pyplot as plt     #グラフのライブラリ
plt.figure(figsize=(14,10))

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.reshape(60000, 28*28)
X_test  = X_test.reshape(10000, 28*28)

# 10,000件のデータを 80%:20% に分割
trData, testData, trLabels, testLabels = train_test_split(np.array(X_test), y_test, test_size=0.2, random_state=10)

model = KNeighborsClassifier(n_neighbors = 3)
model.fit(trData, trLabels)  # 学習データの配置

'''
ここで､識別が誤った画像のインデックスを記録するリストを作り､
そして､テストデータの最初の200件のデータを予測する｡
予測したラベルをtrLableと一致するか確認させる｡
もし誤ったら､インデックスをwrong_dataに入れる｡
最後に､誤った画像を出力する｡
頭に「Wrong:」という､その画像はどの数字に誤って認識されたかを表すタイトルがある
'''
wrong_data = []   # 識別が誤った画像のINDEXを記録するリスト

for index in range(0,200):
    y_predict = model.predict(testData[:200])      #testData 200件に対する予測
#     print(str(index), end = ",")  
#     if index % 20 == 0 and index != 0:     
#         print("\n")
    if y_predict[index] != testLabels[index]:
        wrong_data.append(index)

for i in range(5):
    for j in range(10):
        if (1 + 10 * i + j) < len(wrong_data):         #画像を5X10の表で生成する
            ax = plt.subplot(5, 10, 1 + 10 * i + j)
            ax.set_axis_off()
            ax.set_title("Wrong: " + str(y_predict[wrong_data[10 * i + j]]))        # 誤った数値
            plt.imshow(testData[wrong_data[10 * i + j]].reshape(28, 28).astype("uint8"))
plt.show()
        

