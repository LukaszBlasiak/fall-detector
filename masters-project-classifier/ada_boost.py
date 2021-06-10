from sklearn.externals import joblib
import os

dirname = os.path.dirname(__file__)
model = joblib.load(os.path.join(dirname, 'models\\model_ada_boost.pkl'))


def classify_frames(flatted_buffer):
    prediction = model.predict([flatted_buffer])
    return prediction[0]
