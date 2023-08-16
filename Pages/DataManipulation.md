# 1.1. Data Manipulation
## Getting started
To start, we import tensorflow. For brevity, practitioners often assign the alias tf.
```Python
import tensorflow as tf
```
A tensor represents a (possibly multidimensional) array of numerical values. In the one-dimensional case, i.e., when only one axis is needed for the data, a tensor is called a vector. With two axes, a tensor is called a matrix. With k >2 axes, we drop the specialized names and just refer to the object as a k^th -order tensor.

TensorFlow provides a variety of functions for creating new tensors prepopulated with values. For example, by invoking range(n), we can create a vector of evenly spaced values, starting at 0 (included) and ending at n (not included). By default, the interval size is 1. Unless otherwise specified, new tensors are stored in main memory and designated for CPU-based computation.

```Python
x = tf.range(12, dtype=tf.float32)
x
```
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/2f41e575-e389-466b-9029-f6b0f6343bc3)
Each of these values is called an element of the tensor. The tensor x contains 12 elements. We can inspect the total number of elements in a tensor via the size function.
```Python
tf.size(x)
```
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/b2bd8331-7137-45b2-93f4-5d82e048abfd)
We can access a tensor’s shape (the length along each axis) by inspecting its shape attribute. Because we are dealing with a vector here, the shape contains just a single element and is identical to the size.
```Python
x.shape
```
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/a7be8b16-e6d0-494a-80e8-52f3b688f7e8)
```Python
X = tf.reshape(x, (3, 4))
X
```
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/611c89d2-d221-45e2-aa1c-9926100ab72c)

Note that specifying every shape component to reshape is redundant. Because we already know our tensor’s size, we can work out one component of the shape given the rest. For example, given a tensor of size nand target shape (h,w), we know that w=n/h. To automatically infer one component of the shape, we can place a -1 for the shape component that should be inferred automatically. In our case, instead of calling x.reshape(3, 4), we could have equivalently called x.reshape(-1, 4) or x.reshape(3, -1).
Practitioners often need to work with tensors initialized to contain all 0s or 1s. We can construct a tensor with all elements set to 0 and a shape of (2, 3, 4) via the zeros function
```Python
tf.zeros((2, 3, 4))
```
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/099b1a32-5be6-46ff-8937-4e9fde07d0f8)

```Python
tf.ones((2, 3, 4))
```
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/365c7b95-673f-4d46-ab82-ced95cee2337)
We often wish to sample each element randomly (and independently) from a given probability distribution. For example, the parameters of neural networks are often initialized randomly. The following snippet creates a tensor with elements drawn from a standard Gaussian (normal) distribution with mean 0 and standard deviation 1.
```Python
tf.random.normal(shape=[3, 4])
```
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/39bbb25d-49f9-46b5-82f3-f4a2504c0986)
we can construct tensors by supplying the exact values for each element by supplying (possibly nested) Python list(s) containing numerical literals. Here, we construct a matrix with a list of lists, where the outermost list corresponds to axis 0, and the inner list corresponds to axis 1.
```Python
tf.constant([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/c8b6a578-76d9-45f2-a483-201312bf9086)
## Indexing and Slicing
