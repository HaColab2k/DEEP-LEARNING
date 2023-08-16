# Linear Algebra
## Scalars
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/35645911-0416-45fb-947c-e7852bca6057)
```Python
x = tf.constant(3.0)
y = tf.constant(2.0)

x + y, x * y, x / y, x**y
```
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/9a7880a8-e466-44de-8106-ba349e4fc792)
## Vectors

![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/65bd31ac-ed08-4840-b39a-c4db12b3d873)
```Python
x = tf.range(3)
x
```
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/e5723f18-56dd-4a9b-9799-7f8daf182c11)

![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/b6700d06-3282-46cd-911a-e45e20bd268e)
```Python
x[2]
```

![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/b5802f82-f152-4cdd-b1a2-b6d81541b0a1)

![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/9d0593bd-321a-4ff0-ad13-2b24717cbc59)
```Python
len(x)
```

![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/e54ef107-ea43-474f-a3b2-7098df76f76d)
```Python
x.shape
```
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/1522d1f2-dedb-4f77-a084-6d853d33ec1a)

## Matrices
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/d927ea32-9d3e-4bb4-8ad0-c9f9731c05f4)
```Python
A = tf.reshape(tf.range(6), (3, 2))
A
```
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/6259e00a-3f7b-4d7a-bda2-cf05b755936b)
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/514aa115-0bf6-4dbc-a476-0cecd5eed70d)
```Python
tf.transpose(A)
```
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/aed01989-dbe5-4ae3-80a8-7d40fff0f639)
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/2a5be8eb-cd29-4f41-9b02-615785ac502e)
```Python
A = tf.constant([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
A == tf.transpose(A)
```
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/fb56d8a3-fae0-4356-888d-da790dee49ce)
Matrices are useful for representing datasets. Typically, rows correspond to individual records and columns correspond to distinct attributes.
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/160826fb-117f-495f-9122-8868c82e67fc)
```Python
tf.reshape(tf.range(24), (2, 3, 4))
```
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/80dcb7ee-9ee6-4bfc-81e9-6f3f5074d679)
## Basic Properties of Tensor Arithmetic
Scalars, vectors, matrices, and higher-order tensors all have some handy properties. For example, elementwise operations produce outputs that have the same shape as their operands.
```Python
A = tf.reshape(tf.range(6, dtype=tf.float32), (2, 3))
B = A  # No cloning of A to B by allocating new memory
A, A + B
```
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/d27b8309-7102-456e-81a3-67071bef2131)
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/1db5f397-e011-4244-b96a-d73fd42021b4)
```Python
A * B
```
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/8cf48f69-71bc-406f-8723-54964c202868)
Adding or multiplying a scalar and a tensor produces a result with the same shape as the original tensor. Here, each element of the tensor is added to (or multiplied by) the scalar.
```Python
a = 2
X = tf.reshape(tf.range(24), (2, 3, 4))
a + X, (a * X).shape
```
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/13dfb09c-25df-42d9-9aea-20917014c6c4)
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/b24ad8b0-3ac8-4d4a-af13-b70d06ccb4f9)
```Python
A * B
```
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/5ee9fbc1-cff1-40b1-8092-ebcf74519440)
Adding or multiplying a scalar and a tensor produces a result with the same shape as the original tensor. Here, each element of the tensor is added to (or multiplied by) the scalar.
```Python
a = 2
X = tf.reshape(tf.range(24), (2, 3, 4))
a + X, (a * X).shape
```
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/b91f6104-e8af-42e9-a291-5ef38b250422)
## Reduction
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/e5b16db7-fcff-41db-b344-d8142fe9b9fd)
```Python
x = tf.range(3, dtype=tf.float32)
x, tf.reduce_sum(x)
```
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/950aa3ba-6a05-477d-bd18-238e495d5593)
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/fb65aeaa-66cf-41ea-aeb3-85de75081aef)
```Python
A.shape, tf.reduce_sum(A)
```
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/043badf8-da97-4833-a06e-1142b1c793f3)
By default, invoking the sum function reduces a tensor along all of its axes, eventually producing a scalar. Our libraries also allow us to specify the axes along which the tensor should be reduced. To sum over all elements along the rows (axis 0), we specify axis=0 in sum. Since the input matrix reduces along axis 0 to generate the output vector, this axis is missing from the shape of the output.
```Python
A.shape, tf.reduce_sum(A, axis=0).shape
```
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/79ec31e9-3aa5-4de7-9d44-c8bc95a8a5c0)
Specifying axis=1 will reduce the column dimension (axis 1) by summing up elements of all the columns.
```Python
A.shape, tf.reduce_sum(A, axis=1).shape
```
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/f364e47b-8a33-4a26-a168-f2b74ddd4bb5)
Reducing a matrix along both rows and columns via summation is equivalent to summing up all the elements of the matrix.
```Python
tf.reduce_sum(A, axis=[0, 1]), tf.reduce_sum(A)  # Same as tf.reduce_sum(A)
```
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/b0829f9f-ec80-4c3b-a519-e13bb0e24a53)
A related quantity is the mean, also called the average. We calculate the mean by dividing the sum by the total number of elements. Because computing the mean is so common, it gets a dedicated library function that works analogously to sum.
```Python
tf.reduce_mean(A), tf.reduce_sum(A) / tf.size(A).numpy()
```
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/06a82b4f-9d70-4b9d-9ec2-d65766871aa8)
Likewise, the function for calculating the mean can also reduce a tensor along specific axes.
```Python
tf.reduce_mean(A, axis=0), tf.reduce_sum(A, axis=0) / A.shape[0]
```
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/a7c30fd3-3c51-4bd0-9a71-82a56785c0e6)
## Non-Reduction Sum
Sometimes it can be useful to keep the number of axes unchanged when invoking the function for calculating the sum or mean. This matters when we want to use the broadcast mechanism.
```Python
sum_A = tf.reduce_sum(A, axis=1, keepdims=True)
sum_A, sum_A.shape
```
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/1744dd42-5713-49f1-a708-1d0e4d9a29f9)
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/3ed6a9ef-cb5a-4010-b1d2-02ae7b8dba77)
```Python
A / sum_A
```
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/c615abd2-e68b-437a-9da2-41504e62e8b2)
If we want to calculate the cumulative sum of elements of A along some axis, say axis=0 (row by row), we can call the cumsum function. By design, this function does not reduce the input tensor along any axis.
```Python
tf.cumsum(A, axis=0)
```
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/5e6339c6-c277-4bd0-8608-1943405c1198)
## Dot Products
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/58b36361-a0fd-400d-9b1a-3ee5aebae973)
```Python
y = tf.ones(3, dtype=tf.float32)
x, y, tf.tensordot(x, y, axes=1)
```
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/20a1344d-471d-4f6f-91f3-5f465b6d3dff)
Equivalently, we can calculate the dot product of two vectors by performing an elementwise multiplication followed by a sum:
```Python
tf.reduce_sum(x * y)
```
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/8151059e-7eda-4514-a714-a4916cb09b6e)
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/06712dc1-148c-4a51-b2b3-2375043d1645)
## Matrix–Vector Products
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/6d074288-7f83-4b0c-845b-207e3b2acb3c)
To express a matrix–vector product in code, we use the matvec function. Note that the column dimension of A (its length along axis 1) must be the same as the dimension of x (its length).
```Python
A.shape, x.shape, tf.linalg.matvec(A, x)
```
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/9eb8329d-f500-4ada-b9c3-c58c4bd80e0f)
## Matrix–Matrix Multiplication
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/6ff37033-daad-4731-a907-c5387ffbd11d)
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/0da4e651-7a2d-41e8-8dfc-a74e7e9f8438)
```Python
B = tf.ones((3, 4), tf.float32)
tf.matmul(A, B)
```
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/c50c1c83-08fb-4158-a950-ca8fd844d123)
The term matrix–matrix multiplication is often simplified to matrix multiplication, and should not be confused with the Hadamard product.
## Norms
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/b0992063-763e-4d0d-a84d-1a53b30c7d77)
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/fd769497-81d2-4108-bf03-35b0385f2d86)

```Python
u = tf.constant([3.0, -4.0])
tf.norm(u)
```
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/4523648d-ab6f-41cd-84db-e9611d9a9d42)
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/a2b408c4-0cb9-4cbe-a6ec-7d4e64b0df8f)
```Python
tf.norm(tf.ones((4, 9)))
```
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/13f47bdd-936d-4aa7-948c-9e366b37837f)
While we do not want to get too far ahead of ourselves, we already can plant some intuition about why these concepts are useful. In deep learning, we are often trying to solve optimization problems: maximize the probability assigned to observed data; maximize the revenue associated with a recommender model; minimize the distance between predictions and the ground-truth observations; minimize the distance between representations of photos of the same person while maximizing the distance between representations of photos of different people. These distances, which constitute the objectives of deep learning algorithms, are often expressed as norms.

To recap:

Scalars, vectors, matrices, and tensors are the basic mathematical objects used in linear algebra and have zero, one, two, and an arbitrary number of axes, respectively.

Tensors can be sliced or reduced along specified axes via indexing, or operations such as sum and mean, respectively.

Elementwise products are called Hadamard products. By contrast, dot products, matrix–vector products, and matrix–matrix products are not elementwise operations and in general return objects having shapes that are different from the the operands.

Compared to Hadamard products, matrix–matrix products take considerably longer to compute (cubic rather than quadratic time).

Norms capture various notions of the magnitude of a vector (or matrix), and are commonly applied to the difference of two vectors to measure their distance apart.

Common vector norms include the 
 and 
 norms, and common matrix norms include the spectral and Frobenius norms.
