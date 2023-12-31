﻿# Automatic Differentiation
Calculating derivatives is the crucial step in all the optimization algorithms that we will use to train deep networks. While the calculations are straightforward, working them out by hand can be tedious and error-prone, and these issues only grow as our models become more complex.

Fortunately all modern deep learning frameworks take this work off our plates by offering automatic differentiation (often shortened to autograd). As we pass data through each successive function, the framework builds a computational graph that tracks how each value depends on others. To calculate derivatives, automatic differentiation works backwards through this graph applying the chain rule. The computational algorithm for applying the chain rule in this fashion is called backpropagation.
```Python
import tensorflow as tf
```
## A Simple Function
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/372461d9-3f6e-4235-a05c-8624fc9c1b43)
```Python
x = tf.range(4, dtype=tf.float32)
x
```
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/9a662e09-c0d6-4ae3-ac6a-22946c2dd60b)
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/1563937f-3314-4048-886c-6ea60b7947c4)
```Python
x = tf.Variable(x)
```
We now calculate our function of x and assign the result to y.
```Python
# Record all computations onto a tape
with tf.GradientTape() as t:
    y = 2 * tf.tensordot(x, x, axes=1)
y
```
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/ed885935-0f74-4880-a7a5-8af566547083)
We can now calculate the gradient of y with respect to x by calling the gradient method.
```Python
x_grad = t.gradient(y, x)
x_grad
```
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/0a27ea16-7d27-4588-80ef-6a77fd4f8115)
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/4c59be5d-8445-4f00-ad2b-22fb5ec0bcca)
```Python
x_grad == 4 * x
```
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/32caa9b6-460a-4d6e-b3ee-e2f3811e884f)
Now let’s calculate another function of x and take its gradient. Note that TensorFlow resets the gradient buffer whenever we record a new gradient.
```Python
with tf.GradientTape() as t:
    y = tf.reduce_sum(x)
t.gradient(y, x)  # Overwritten by the newly calculated gradient
```
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/3226735f-abb1-4f34-b1bb-a9ee9ba215df)
