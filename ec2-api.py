from flask import Flask, request, jsonify
from flask_cors import CORS  # Import Flask-CORS
import numpy as np
import cv2
import typing
from mltu.configs import BaseModelConfigs
from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder
from mltu.transformers import ImageResizer
import requests
import base64

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})

class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, image: np.ndarray):
        image = ImageResizer.resize_maintaining_aspect_ratio(
            image, *self.input_shape[:2][::-1]
        )

        image_pred = np.expand_dims(image, axis=0).astype(np.float32)

        preds = self.model.run(None, {self.input_name: image_pred})[0]

        text = ctc_decoder(preds, self.char_list)[0]

        return text


# Define an endpoint for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
         # Get the base64-encoded image data from the request
        image_data_base64 = request.json.get('image_data')

        if not image_data_base64:
            return jsonify({'error': 'Image data is missing'})

        # Decode the base64 image data to bytes
        image_data = base64.b64decode(image_data_base64)

        if not image_data:
            return jsonify({'error': 'Failed to decode image data'})

        configs = BaseModelConfigs.load("./configs.yaml")
        configs.model_path = "./model.onnx"
        
        # Convert the image data to a NumPy array
        image_np = np.frombuffer(image_data, dtype=np.uint8)
        image_np = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        # Perform inference
        model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)
        prediction_text = model.predict(image_np)

        # Return the prediction result as json object
        return jsonify({'prediction_text': prediction_text})
    except Exception as e:
        return jsonify({'error': str(e)})
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0')
