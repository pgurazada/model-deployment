import io
import numpy as np
import tensorflow as tf

from PIL import Image

from flask import Flask, request, jsonify

label_map = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x']

app = Flask("romannumeralpredictor")


def load_model():
	# load the pre-trained Keras model

	global model
	model = tf.keras.models.load_model("roman-numeral-predictor-mobilenet")


def prepare_image(image, target):
	# if the image mode is not RGB, convert it
	if image.mode != "RGB":
		image = image.convert("RGB")

	# resize the input image and preprocess it
	image = image.resize(target)
	image = tf.keras.preprocessing.image.img_to_array(image)
	image = np.expand_dims(image, axis=0)
	image = tf.keras.applications.imagenet_utils.preprocess_input(image)

	# return the processed image
	return image


@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}

	# ensure an image was properly uploaded to our endpoint
    # 
    if request.method == "POST":
        if request.files.get("image"):
            # read the image in PIL format
            image = request.files["image"].read()
            image = Image.open(io.BytesIO(image))

			# preprocess the image and prepare it for classification
            image = prepare_image(image, target=(224, 224))

			# classify the input image and then initialize the list
			# of predictions to return to the client
            preds = model.predict(image)

            final_prediction = np.argmax(preds)
            final_prediction_prob = preds.max()

            data["predictions"] = []

			# loop over the results and add them to the list of
			# returned predictions
            # 
            r = {"label": label_map[final_prediction], 
                 "probability": float(final_prediction_prob)}
			
            data["predictions"].append(r)

			# indicate that the request was a success
            
            data["success"] = True

	# return the data dictionary as a JSON response
    
    return jsonify(data)


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
		      "please wait until server has fully started"))
	load_model()
	app.run()
