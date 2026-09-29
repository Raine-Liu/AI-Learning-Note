[https://docs.pytorch.org/tutorials/beginner/introyt/tensors_deeper_tutorial.html]
Prepare
```python
import torch
import math
```
#### Creating Tensors: 
using `torch.empty`
```python
x = torch.empty(3, 4) #为3*4的Tensor申请内存,但不做初始化
print(type(x))
print(x)

print(type(x)) # <class 'torch.Tensor'> Python对象类型
print(x.dtype) # torch.float32 元素的数据类型
print(x.device) # cpu 张量所在设备
print(x.type()) # torch.FloatTensor 张量使用的是哪种具体的张量类型

a = torch.empty(3, 4, dtype=torch.float64) 
b = torch.empty(3, 4, dtype=torch.int64) #显式指定数据类型
```

out:
- by default, PyTorch tensors are populated with 32-bit floating point numbers
```bash
<class 'torch.Tensor'>
tensor([[-2.8994e+10,  4.5908e-41, -2.0906e+10,  4.5908e-41],
        [-2.3894e+15,  4.5907e-41, -4.0558e+15,  4.5907e-41],
        [-2.8994e+10,  4.5908e-41, -4.0910e+15,  4.5907e-41]])
```
initialize：
```python
zeros = torch.zeros(2,3) #全0
ones = torch.ones(2,3) #全1

torch.manual_seed(1729) #把 PyTorch 随机数生成器的“种子”设置为 1729，同样的种子会生成一样的随机数
random = torch.rand(2, 3)
#torch.rand表示生成均匀分布在[0, 1)内的随机浮点数
x = 10 + torch.rand(2, 3) * 10 #范围就是[10,20)
```

```bash
tensor([[0., 0., 0.], #小数点表示是浮点数
        [0., 0., 0.]])
tensor([[1., 1., 1.],
        [1., 1., 1.]])
tensor([[0.3126, 0.3791, 0.3087],
        [0.0736, 0.4216, 0.0691]])
```

##### Tensor Shapes
Often, when you’re performing operations on two or more tensors, they will need to be of the same _shape_.
`torch.*_like()` methods: 生成和x形状一样的张量
```python
x = torch.empty(2, 2, 3)
print(x.shape) #torch.Size([2, 2, 3])

zeros_like_x = torch.zeros_like(x)
ones_like_x = torch.ones_like(x)
rand_like_x = torch.rand_like(x)
```
参数形状直接写多个整数或者写整数元组都行
##### specify its data directly
```python
some_constants = torch.tensor([[3.1415926, 2.71828], [1.61803, 0.0072897]])
#tensor([[3.1416, 2.7183], [1.6180, 0.0073]])

some_integers = torch.tensor((2, 3, 5, 7, 11, 13, 17, 19))
# tensor([ 2,  3,  5,  7, 11, 13, 17, 19])

more_integers = torch.tensor(((2, 4, 6), [3, 6, 9]))
#tensor([[2, 4, 6],
        #[3, 6, 9]])
```
`torch.tensor()` creates a copy of the data.

#### Tensor Data Types
setting datatype
```python
a = torch.ones((2, 3), dtype=torch.int16)
b = torch.rand((2, 3), dtype=torch.float64) * 20.0
c = b.to(torch.int32)
```
```bash
tensor([[1, 1, 1],
        [1, 1, 1]], dtype=torch.int16)
tensor([[ 0.9956,  1.4148,  5.8364],
        [11.2406, 11.2083, 11.6692]], dtype=torch.float64)
tensor([[ 0,  1,  5],
        [11, 11, 11]], dtype=torch.int32)
```

####  Math & Logic with PyTorch Tensors
basic arithmetic
```python
ones = torch.zeros(2, 2) + 1
twos = torch.ones(2, 2) * 2
threes = (torch.ones(2, 2) * 7 - 1) / 2
fours = twos**2
sqrt2s = twos**0.5
```
```bash
tensor([[1., 1.],
        [1., 1.]])
tensor([[2., 2.],
        [2., 2.]])
tensor([[3., 3.],
        [3., 3.]])
tensor([[4., 4.],
        [4., 4.]])
tensor([[1.4142, 1.4142],
        [1.4142, 1.4142]])
```
Arithmetic operations between tensors and scalars, such as addition, subtraction, multiplication, division, and exponentiation are distributed over every element of the tensor.
Similar operations between two tensors:
```python
powers2 = twos ** torch.tensor([ [1, 2], [3, 4]])
fives = ones + fours
dozens = threes * fours
```
```bash
tensor([[ 2.,  4.],
        [ 8., 16.]])
tensor([[5., 5.],
        [5., 5.]])
tensor([[12., 12.],
        [12., 12.]])
```
做形状不同的Tensor之间的运算会`run-time error`

#### Tensor Broadcasting
broadcasting 可以使形状不同的张量相乘, 将形状较小的张量“扩展”成合适的形状
```python
rand = torch.rand(2, 4)
doubled = rand * (torch.ones(1, 4) * 2)
```
```bash
tensor([[0.6146, 0.5999, 0.5013, 0.9397],
        [0.8656, 0.5207, 0.6865, 0.3614]])
tensor([[1.2291, 1.1998, 1.0026, 1.8793],
        [1.7312, 1.0413, 1.3730, 0.7228]])
```
The rules for broadcasting:
-  Each tensor must have at least one dimension - no empty tensors.
- Comparing the dimension sizes of the two tensors, _going from last to first（右对齐）:_
    - Each dimension must be equal, _or_
    - One of the dimensions must be of size 1, _or_
    - The dimension does not exist in one of the tensors

满足上述规则的例子
```python
a = torch.ones(4, 3, 2)
b = a * torch.rand(3, 2)  # 3rd & 2nd dims identical to a, dim 1 absent
c = a * torch.rand(3, 1)  # 3rd dim = 1, 2nd dim identical to a
d = a * torch.rand(1, 2)  # 3rd dim identical to a, 2nd dim = 1
```
Tensor还可以进行很多种数学操作[https://docs.pytorch.org/docs/2.13/torch.html#math-operations]

####  Altering Tensors in Place
运算中张量是怎么变化的
```python
a = torch.tensor([0, math.pi / 4, math.pi / 2, 3 * math.pi / 4])
print(a)
print(torch.sin(a))  # this operation creates a new tensor in memory
print(a)  # a has not changed

b = torch.tensor([0, math.pi / 4, math.pi / 2, 3 * math.pi / 4])
print(b)
print(b.sin_())  # note the underscore
print(b)  # b has changed

print(a.add_(b)) # a has changed
print(b.mul_(b)) # b has changed
```
```python
c = torch.zeros(2, 2)
d = torch.matmul(a, b, out=c)
torch.rand(2, 2, out=c)
```
把结果放入c

#### Copy Tensors
assigning a tensor to a variable makes the variable a _label_ of the tensor, and does not copy it
```python
a = torch.ones(2, 2)
b = a

a[0][1] = 561  # we change a...
print(b)  # ...and b is also altered
```

want a separate copy of the data to work on --- use `clone()`
```python
a = torch.ones(2, 2)
b = a.clone()

assert b is not a  # different objects in memory...
print(torch.eq(a, b))  # ...but still with the same contents!

a[0][1] = 561  # a changes...
print(b)  # ...but b is still all ones
```
`clone() ` 只复制数据，不会切断自动求导(autograd)关系
如果你想复制一个 Tensor，同时让它和原来的计算图完全无关，需要使用 `detach()`
```python
a = torch.rand(2, 2, requires_grad=True)  # turn on autograd
print(a) #requires_grad=True

b = a.clone()
print(b) #grad_fn=<CloneBackward0>

c = a.detach().clone()
print(c) #不计算梯度

print(a)
```

|操作|复制数据|新内存|保留梯度|
|---|---|---|---|
|`b=a`|❌|❌|共享|
|`b=a.clone()`|✅|✅|✅|
|`b=a.detach()`|❌(共享数据)|❌|❌|
|`b=a.detach().clone()`|✅|✅|❌|

#### Moving to Accelerator
First, we should check whether an accelerator is available, with the`is_available()` method.
```python
if torch.accelerator.is_available():
    print("We have an accelerator!")
else:
    print("Sorry, CPU only.")
```
创建gpu上的张量
```python
if torch.accelerator.is_available():
    gpu_rand = torch.rand(2, 2, device=torch.accelerator.current_accelerator())
    print(gpu_rand)
else:
    print("Sorry, CPU only.")
```
先判断设备，再计算
```python
my_device = (
    torch.accelerator.current_accelerator()
    if torch.accelerator.is_available()
    else torch.device("cpu")
)
print(f"Device: {my_device}")

x = torch.rand(2, 2, device=my_device)
print(x)
```
移动数据去另一个设备`to`
```python
y = torch.rand(2, 2)
y = y.to(my_device)
```

#### Manipulating Tensor Shapes
##### Changing the Number of Dimensions
turn a tensor to a batch: The `unsqueeze()` method adds a dimension of extent 1. `unsqueeze(0)` adds it as a new zeroth dimension - now you have a batch of one!
```python
a = torch.rand(3, 226, 226)
b = a.unsqueeze(0)

print(a.shape) #torch.Size([3, 226, 226])
print(b.shape) #torch.Size([1, 3, 226, 226])
```
`squeeze()`：删除大小为 1 的维度
`unsqueeze`和`squeeze`都只能进行一维操作
```python
a = torch.rand(1, 20)
print(a.shape) #torch.Size([1, 20])

b = a.squeeze(0)
print(b.shape) #torch.Size([20])

c = torch.rand(2, 2)
print(c.shape) #torch.Size([2, 2])

d = c.squeeze(0)
print(d.shape) #torch.Size([2, 2])
```
可以用`unsqueeze`来帮助广播机制的进行
```python
a = torch.ones(4, 3, 2)
b = torch.rand(3)  # trying to multiply a * b will give a runtime error
c = b.unsqueeze(1)  # change to a 2-dimensional tensor, adding new dim at the end
print(c.shape)
print(a * c)  # broadcasting works again!
```

