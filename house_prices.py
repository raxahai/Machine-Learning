import tensorflow as tf
import numpy as np

model = tf.keras.Sequential([
    (tf.keras.layers.Dense(units = 1, input_shape= [1]))
])
model.compile(optimizer = "sgd",
                loss = "mean_squared_error")
xs = np.array([1.0,2.0,3.0,5.0,8.0,9.0], dtype= float)
ys = np.array([1.0,1.5,2.0,3.0,4.5,5.0], dtype=float) #scaled down to one decimal place
# in order to make model work well
model.fit(xs,ys,epochs = 700)
print(model.predict([7.0]))
print(model.predict([4.0]))
