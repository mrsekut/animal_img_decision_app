from PIL import Image
import os, glob
import numpy as np

classes = ["cat","penguin","hedgehog","snake"]
num_classes = len(classes)
image_size = 50
num_testdata = 100

X_train = []
X_test = []
Y_train = []
Y_test = []

for index, clss in enumerate(classes):
    photos_dir = "./" + clss
    files = glob.glob(photos_dir + "/*.jpg") # ファイル一覧を取得

    for i, file in enumerate(files):
        if i >= 200: break
        image = Image.open(file)
        image = image.convert("RGB") # rgbに変換
        image = image.resize((image_size, image_size))
        data = np.asarray(image) # 数値データに変換

        if i < num_testdata:
            X_test.append(data)
            Y_test.append(index)
        else:
            # 画像を回転させる
            for angle in range(-20,20,5):
                img_r = image.rotate(angle)
                data = np.asarray(img_r)
                X_train.append(data)
                Y_train.append(index)

                # 画像を反転させる
                img_trans = img_r.transpose(Image.FLIP_LEFT_RIGHT)
                data = np.asarray(img_trans)
                X_train.append(data)
                Y_train.append(index)

X_train = np.array(X_train)
X_test = np.array(X_test)
Y_train = np.array(Y_train)
Y_test = np.array(Y_test)

xy = (X_train, X_test, Y_train, Y_test)
np.save("./animal_aug.npy", xy)
