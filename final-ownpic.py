import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import time
from collections import OrderedDict
from PIL import Image

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

scores = OrderedDict()
"""次の処理はデータの正規化"""
BTnore = Image.open("BlackTshirt.jpg").resize((28,28),Image.ANTIALIAS)
BT = np.array(BTnore.convert("L"))
BT = BT.reshape(1,28*28).astype(np.float64)
BSnore = Image.open("blacksniker.jpg").resize((28,28),Image.ANTIALIAS)
BS = np.array(BSnore.convert("L"))
BS = BS.reshape(1,28*28).astype(np.float64)

X_60000, X_10000 = ((X_train.reshape(60000, 28 * 28)).astype(np.float64),
                    (X_test.reshape(10000, 28 * 28)).astype(np.float64))

X_60000 /= X_60000.max()
X_10000 /= X_10000.max()

"""ユニット数と中間層"""
# units = ((50,), (50, 50, 50,), (100,), (100, 100, 100,), (1000, 1000, 1000,))
units = ((300,300,300,300,300,300,300,))

"""学習開始"""
try:
    model = MLPClassifier(hidden_layer_sizes=units, activation="relu")
    trData, trData2, trLabels, trLabels2 = train_test_split(np.array(X_60000), y_train, test_size=0.1,
                                                            random_state=10)

    start = time.time()

    model.fit(trData, trLabels)

    duration = time.time() - start

    # テストデータでクラス（数字）予測
    predictionBT = model.predict(BT)
    predictionBS = model.predict(BS)

    print("predictionT: " + str(predictionBT))
    print("predictionS: " + str(predictionBS))
except:
    raise



