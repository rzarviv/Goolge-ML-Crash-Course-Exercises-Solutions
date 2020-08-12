import tensorflow as tf
import numpy as np
from tensorflow import keras

# To identify the desired pattern, we create a one-layered network, which 
# consists of 1 neuron (i.e. unit) and returns 1 value (i.e. its input shape is 
# just one value).
model = tf.keras.Sequential(keras.layers.Dense(units=1, input_shape=[1]))

# Next we define the optimizer (i.e. the "learning function") and the loss 
# function (i.e. a function that takes the expected output [from the 'house_prices_array'] 
# and comapres it with the actual output [i.e. the NN 'guess'] and measures how 
# close is the expected output to the actual output).

# - 'sgd' stands for 'stochastic gradient descent'
model.compile(optimizer='sgd', loss='mean_squared_error')

# Next we define the data set and our 'expected output' set
num_of_bedrooms_array = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 9.0, 10.0], dtype=float)
house_prices_array = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.5, 5.0, 5.5], dtype=float)

# Now we let the model 'learn' the pattern from the data we supplied to it for
# 500 epochs (loops)
model.fit(num_of_bedrooms_array, house_prices_array, epochs=500)

# Finally we check what's the model's price guess for a 7 bedroom house
print(model.evaluate([7.0, 11.0, 12.0],[4.0, 6.0, 6.5]))
#print(model.predict([7.0]))
