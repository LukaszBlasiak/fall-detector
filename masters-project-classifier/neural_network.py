from tensorflow.keras import models
import numpy as np
import os

dirname = os.path.dirname(__file__)
model = models.load_model(os.path.join(dirname, 'models\\model.h5'))


def classify_frames(flatted_buffer):
    prediction = model.predict_classes(np.asarray([flatted_buffer]))
    return prediction[0][0]
