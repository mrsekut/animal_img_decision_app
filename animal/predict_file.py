import os
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from flask import send_from_directory
from keras.models import Sequential, load_model
import keras, sys
from PIL import Image
import numpy as np
from flask_dropzone import Dropzone

classes = ["cat","penguin","hedgehog","snake"]
num_classes = len(classes)
image_size = 50

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['jpg','png','gif'])

app = Flask(__name__)

app.config.update(
    UPLOAD_FOLDER=UPLOAD_FOLDER,
    DROPZONE_MAX_FILES=1,
    DROPZONE_REDIRECT_VIEW='predict_page',
)

dropzone = Dropzone(app)


# ファイルのアップロードの可否を判定する
def allowed_file(filename):
    # 正しい拡張子が指定されているか
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    return render_template('index.html')

@app.route('/predict_page', methods=['GET','POST'])
def predict_page():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash("ファイルがありません")
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash("ファイルがありません")
            return redirect(request.url)
        if file and allowed_file(file.filename):

            # サニタイズ処理
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            model = load_model('./animal_cnn_aug.h5')

            image = Image.open(filepath)
            image = image.convert("RGB")
            image = image.resize((image_size, image_size))
            data = np.array(image)
            X = []
            X.append(data)
            X = np.array(X)

            result = model.predict([X])[0]
            predicted = result.argmax()
            percentage = int(result[predicted] * 100)

            return "ラベル: " + classes[predicted] + ", 確率:" + str(percentage) + "%"


if __name__ == '__main__':
    app.run(debug=True)
