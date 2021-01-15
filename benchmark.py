from datetime import datetime
import tensorflow as tf

iterations = 20000
start = datetime.now()

for x in range(iterations):
    a = tf.random.normal([256, 256], 0, 1, tf.float32)
    b = tf.random.normal([256, 256], 0, 1, tf.float32)
    c = tf.tensordot(a, b, axes = 1)

print(datetime.now() - start)
