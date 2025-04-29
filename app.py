from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf


from utils import preprocess_image

app = Flask(__name__)

model = tf.keras.models.load_model("best_model_resnet_like.keras")

classes = ["chicken", "cow", "dog", "elephant"]

@app.route('/predict/', methods=['POST'])
def predict():
    file = request.files['file']
    try:
        image_bytes = file.read()
        image = preprocess_image(image_bytes)

        probs = model.predict(image)[0]
        predicted_index = np.argmax(probs)
        predicted_label = classes[predicted_index]

        return jsonify({
            'predicted_class': predicted_label,
            'probabilities': {
                label: float(prob)
                for label, prob in zip(classes, probs)
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
