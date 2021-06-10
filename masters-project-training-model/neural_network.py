from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import TensorBoard
import datetime
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tensorflow.keras.optimizers import Adam
from dataset_util import load_dataset


# load the dataset
x_train, x_test, y_train, y_test = load_dataset(test_split_ratio=0, shuffle=True)

# define the keras model
model = Sequential()
model.add(Dense(400, input_dim=512, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(Adam(lr=.0001), loss='binary_crossentropy', metrics=['accuracy'])

log_dir = "logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir)

# fit the keras model on the dataset
model.fit(x_train, y_train, epochs=250, batch_size=10, callbacks=[tensorboard_callback])
# evaluate the keras model
loss, accuracy = model.evaluate(x_train, y_train)
print('Accuracy: %.2f' % (accuracy*100))
print('Loss: %.2f' % loss)

model.save('model.h5')

