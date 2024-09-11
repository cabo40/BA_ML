import tensorflow as tf

x = tf.Variable(2.0)
y = tf.Variable(3.0)

with tf.GradientTape() as t:
    z = 1 / (x ** 2) + y

grad = t.gradient(z, {'x': x, 'y': y})

print('dz/dx:', grad['x'])
print('dz/dy:', grad['y'])

# SIMD
