import tensorflow as tf
import numpy as np 
import logging

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

celsius_q = np.array([-40,-10,0,8,15,22,38], dtype=float)
farenheit_a = np.array([-40,14,32,46,59,72,100], dtype=float)

# for i,c in enumerate(celsius_q):
#     print(f"celsius = {c} and farenheit = {farenheit_a[i]}")

l0 = tf.keras.layers.Dense(units = 1, input_shape=[1])
model = tf.keras.Sequential([l0])
model.compile(loss = "mean_squared_error", optimizer = tf.keras.optimizers.Adam(0.1))

history = model.fit(celsius_q,farenheit_a, epochs = 500, verbose = False)
print('finished training the model')
print(model.predict([100.0]))