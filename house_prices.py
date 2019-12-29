import tensorflow as tf
import numpy as np

model = tf.keras.Sequential([
    (tf.keras.layers.Dense(units = 1, input_shape= [1]))
])
model.compile(optimizer = "sgd",
                loss = "mean_squared_error")
xs = np.array([1.0,2.0,3.0,5.0,8.0,9.0], dtype= float)
ys = np.array([100,150,200,300,450,500], dtype=float)
model.fit(xs,ys,epochs = 700)
print(model.predict([7.0]))
print(model.predict([4.0]))