import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
# from google.colab import files



classes = ["cat","penguin","hedgehog","snake"]
num_classes = len(classes)
image_size = 50


# ハイパーパラメータ
batch_size = 32
epoch = 10
lr = 0.0001


def main():
    X_train, X_test, Y_train, Y_test = np.load("./animal_aug.npy")

    # 正規化
    X_train = X_train.astype("float") / 255
    X_test = X_test.astype("float") / 255


    # one-hot encoding
    Y_train = np_utils.to_categorical(Y_train, num_classes)
    Y_test = np_utils.to_categorical(Y_test, num_classes)

    start = time.time()
    model = model_train(X_train, X_test, Y_train, Y_test)
    model_eval(model, X_test, Y_test)
    print("time: ", time.time() - start)


###
# 学習
###

def model_train(X_train, X_test, y_train, y_test):

    model = Sequential()

    model.add(Conv2D(32, (3 ,3), padding="same", input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.summary()

    # 最適化
    opt = keras.optimizers.rmsprop(lr=lr, decay=1e-6)

    model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

    early_stopping = EarlyStopping(patience=0, verbose=1)

    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        epochs=epoch,
                        verbose=1,
                        validation_data=(X_test, y_test),
                        callbacks=[early_stopping])

    model.save("./animal_cnn_aug.h5")
    # files.download('animal_cnn_aug.h5')

    visualization(history)

    return model



###
# 評価
###

def model_eval(model, X, y):
    scores = model.evaluate(X, y, verbose=1)
    print("test loss: ", scores[0])
    print("test acc: ", scores[1])


###
# 可視化
###

def visualization(history):

    # Accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()



if __name__ == "__main__":
    main()
