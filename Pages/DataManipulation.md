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
As with Python lists, we can access tensor elements by indexing (starting with 0). To access an element based on its position relative to the end of the list, we can use negative indexing. Finally, we can access whole ranges of indices via slicing (e.g., X[start:stop]), where the returned value includes the first index (start) but not the last (stop). Finally, when only one index (or slice) is specified for a k^th-order tensor, it is applied along axis 0. Thus, in the following code, [-1] selects the last row and [1:3] selects the second and third rows.
```Python
X[-1], X[1:3]
```
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/fbd41484-dc34-4009-83e9-bd3464f77ec1)
Tensors in TensorFlow are immutable, and cannot be assigned to. Variables in TensorFlow are mutable containers of state that support assignments. Keep in mind that gradients in TensorFlow do not flow backwards through Variable assignments.
Beyond assigning a value to the entire Variable, we can write elements of a Variable by specifying indices.
```Python
X_var = tf.Variable(X)
X_var[1, 2].assign(9)
X_var
```
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/d517c890-00ca-4072-9ee9-fd6eda885b00)
If we want to assign multiple elements the same value, we apply the indexing on the left-hand side of the assignment operation. For instance, [:2, :] accesses the first and second rows, where : takes all the elements along axis 1 (column). While we discussed indexing for matrices, this also works for vectors and for tensors of more than two dimensions.
```Python
X_var = tf.Variable(X)
X_var[:2, :].assign(tf.ones(X_var[:2,:].shape, dtype=tf.float32) * 12)
X_var
```
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/7ef73d8f-40de-4e95-abb1-986866e9ef02)
## Operations
```Python
tf.exp(x)
```
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/4c77b49f-66f5-4e3b-a4fb-1742ed3a264e)
```Python
x = tf.constant([1.0, 2, 4, 8])
y = tf.constant([2.0, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y
```
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/6d891c57-9656-43c8-818e-8a4a37037d00)
We can also concatenate multiple tensors, stacking them end-to-end to form a larger one. We just need to provide a list of tensors and tell the system along which axis to concatenate. The example below shows what happens when we concatenate two matrices along rows (axis 0) instead of columns (axis 1). We can see that the first output’s axis-0 length (6) is the sum of the two input tensors’ axis-0 lengths (3+3); while the second output’s axis-1 length (8) is the sum of the two input tensors’ axis-1 lengths (4+4).
```Python
X = tf.reshape(tf.range(12, dtype=tf.float32), (3, 4))
Y = tf.constant([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
tf.concat([X, Y], axis=0), tf.concat([X, Y], axis=1)
```
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/8a82d25e-9151-4f72-9820-6c6bc3401371)
Sometimes, we want to construct a binary tensor via logical statements. Take X == Y as an example. For each position i, j, if X[i, j] and Y[i, j] are equal, then the corresponding entry in the result takes value 1, otherwise it takes value 0.
```Python
X == Y
```
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/78220eed-5ec9-4d6d-ae6d-81454f76a128)
Summing all the elements in the tensor yields a tensor with only one element.
```Python
tf.reduce_sum(X)
```
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/924e6576-7506-4e1e-8883-640d26a0f93a)



