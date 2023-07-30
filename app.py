from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__, template_folder='templates')

@app.route("/")
def code():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    model = tf.keras.models.load_model("fer_model8.h5")
    image = Image.open(request.files["image"])
    image = image.convert('L')  # convert image to grayscale
    image = image.resize((48, 48))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    emotion = np.argmax(prediction)
    x = ""
    emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    x = emotions[int(str(emotion))]
    return jsonify({"emotion": x})

if __name__ == "__main__":
    app.run(port='4000', debug=True)
