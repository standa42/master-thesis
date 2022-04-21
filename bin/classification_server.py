from flask import Flask, request, Response
import jsonpickle
import numpy as np
import cv2
from PIL import Image
import keras


import tensorflow as tf
from tensorflow.keras.applications import ResNet50, VGG16



class ClassificationModel:
    def __init__(self):
        self.labels_mapping = ['10', '100', '101', '110', '120', '130', '140', '141', '150', '151', '160', '170', '171', '180', '181', '190', '20', '200', '210', '220', '230', '240', '30', '40', '50', '60', '70', '71', '80', '90', '91', 'unrecognized']
        self.num_classes = len(self.labels_mapping)

        # img_shape = (256, 256, 3)

        # model = tf.keras.Sequential()
        # base_model = ResNet50(include_top=False, input_shape=img_shape, weights = 'imagenet')

        # how_many_layers_to_train = 40
        # for layer in base_model.layers[:-how_many_layers_to_train]: #175
        #     layer.trainable = False

        # for layer in base_model.layers[-how_many_layers_to_train:]: #175
        #     layer.trainable = True

        # model.add(base_model)
        # model.add(tf.keras.layers.Flatten())
        # model.add(tf.keras.layers.Dense(self.num_classes, activation='softmax'))

        # model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
        #                 optimizer='adam', 
        #                 metrics=['accuracy'])

        # model.load_weights("./model/rims_classification_checkpoints/checkpoint2")
        model = keras.models.load_model("./model/rims_classification_checkpoints\EfficientNetB0_weights-imagenet_unfreezed-25-layers")

        self.model = model
    
    def inference(self, image):
        # image = list(image)
        # image = np.array(image)
        # image = image / 255.0
        image = np.reshape(image, (1,256,256,3))

        print(f"Number of classes is: {self.num_classes}")
        print(f"Labels mapping is: {self.labels_mapping}")
        
        prediction_probs = self.model.predict(image)
        
        print(f"Raw prediction is: {prediction_probs}")
        
        prediction = prediction_probs.argmax(axis=-1)
        
        print(f"Type of prediction: {type(prediction)}")
        print(f"ArgMaxed prediction is: {prediction}")
        print(f"Prediction[0] is: {prediction[0]}")
        
        label = self.labels_mapping[prediction[0]]
        
        print(f"Label is: {label}")

        print(f"Prediction prob for label (argmaxed prediction) {(prediction_probs[0])[prediction]}")

        if (prediction_probs[0])[prediction] < 0.9:
            print(f"Probability was low {(prediction_probs[0])[prediction]}, returning unrecognized")
            return 'unrecognized'
        
        return label

classification_model = ClassificationModel()

# Initialize the Flask application
app = Flask(__name__)

# route http posts to this method
@app.route('/api/test', methods=['POST'])
def test():
    r = request
    # convert string of image data to uint8
    nparr = np.frombuffer(r.data, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    saving_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    saving_image = Image.fromarray(saving_image)
    saving_image.save("C:/Users/rnsk/Desktop/abc.png")

    label = classification_model.inference(img)

    # do some fancy processing here....

    # build a response dict to send back to client
    response = {'message': str(label)}
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")

print("first inference: " + str(classification_model.inference( cv2.imread("C:/Users/rnsk/Desktop/abc.png")) ))
# start flask app
app.run(host="0.0.0.0", port=5000, threaded=True)