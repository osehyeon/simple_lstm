from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Activation, Reshape

# Define the model
model = Sequential()
model.add(LSTM(16, input_shape=(128, 1)))
model.add(Reshape((16, 1))) 
model.add(LSTM(16, input_shape=(16, 1)))
model.add(Reshape((16, 1))) 
model.add(LSTM(16, input_shape=(16, 1)))
model.add(Reshape((16, 1))) 
model.add(LSTM(1, input_shape=(16, 1)))
# Adding the final Dense layer with 1 output unit
#model.add(Activation('sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy')

# Save the model as an h5 file
model.save('lstm_test.h5')
