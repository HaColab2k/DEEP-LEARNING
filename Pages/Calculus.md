# 1.4 Calculus
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/7baa2871-0f64-44ff-bee7-0b9565772426)
```Python
%matplotlib inline
import numpy as np
from matplotlib_inline import backend_inline
from d2l import tensorflow as d2l
```
## Derivatives and Differentiation
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/4c950e7e-f7eb-4091-9feb-40f7b9d24417)
```Python
def f(x):
    return 3 * x ** 2 - 4 * x
```
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/88917ca2-9e1d-47a4-abbd-a5a321224895)
```Python
for h in 10.0**np.arange(-1, -6, -1):
    print(f'h={h:.5f}, numerical limit={(f(1+h)-f(1))/h:.5f}')
```
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/85ccfe7d-8847-448d-8429-014004d907ae)
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/36d56c0a-c37e-42b8-94ff-5dceef829322)
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/0db972d1-f5b6-40d7-8c5f-a01740d1b7bf)

## Visualization Utilities
We can visualize the slopes of functions using the matplotlib library. We need to define a few functions. As its name indicates, use_svg_display tells matplotlib to output graphics in SVG format for crisper images. The comment #@save is a special modifier that allows us to save any function, class, or other code block to the d2l package so that we can invoke it later without repeating the code, e.g., via d2l.use_svg_display().
```Python
def use_svg_display():  #@save
    """Use the svg format to display a plot in Jupyter."""
    backend_inline.set_matplotlib_formats('svg')
```
Conveniently, we can set figure sizes with set_figsize. Since the import statement from matplotlib import pyplot as plt was marked via #@save in the d2l package, we can call d2l.plt.
```Python
def set_figsize(figsize=(3.5, 2.5)):  #@save
    """Set the figure size for matplotlib."""
    use_svg_display()
    d2l.plt.rcParams['figure.figsize'] = figsize
```
The set_axes function can associate axes with properties, including labels, ranges, and scales.
```Python
#@save
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib."""
    axes.set_xlabel(xlabel), axes.set_ylabel(ylabel)
    axes.set_xscale(xscale), axes.set_yscale(yscale)
    axes.set_xlim(xlim),     axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()
```
With these three functions, we can define a plot function to overlay multiple curves. Much of the code here is just ensuring that the sizes and shapes of inputs match.
```Python
#@save
def plot(X, Y=None, xlabel=None, ylabel=None, legend=[], xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """Plot data points."""

    def has_one_axis(X):  # True if X (tensor or list) has 1 axis
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X): X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)

    set_figsize(figsize)
    if axes is None:
        axes = d2l.plt.gca()
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        axes.plot(x,y,fmt) if len(x) else axes.plot(y,fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
```
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/4bcb2f0a-b297-497b-8541-81cbdf6a6a7d)
```Python
x = np.arange(0, 3, 0.1)
plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
```
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/975995e1-b8bc-4e1b-87fd-686e371c300d)

## Partial Derivatives and Gradients
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/7c2b3507-b2e8-4092-b6ed-9c3e7488984d)
## Chain Rule
![image](https://github.com/HaColab2k/DEEP-LEARNING/assets/127838132/f609708f-9519-44d6-9e34-7b7c9d14c1ac)

