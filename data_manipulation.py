# numpy only supports CPU computation, unlike pytorch or tensorflow
# tensors are an n-dimensional array
# tensor class supports automatic differentiation
# tutorial can be done in MXNET, PYTORCH, or TENSORFLOW
# pytorch tutorial is done here

# import torch instead of pytorch
# tensor with one axis = vector, two = matrix; kth order tensor
import torch
'''
# create new tensor of evenly spaced values
x = torch.arange(12, dtype=torch.float32)
print(x)
print(x.shape)
print(x.numel())
X = x.reshape(3,4)
print(X)
# -1 means the tensor automatically infers this dimension
X = x.reshape(-1,4)
print(X)
# 2 here is another dimension
z = torch.zeros((2, 3, 4))
print(z)

o = torch.ones((2, 3, 4))
print(o)

# sample elements from a standard Gaussian (normal) distribution
r = torch.randn(3, 4)
print(r)

# specify exact values in a tensor
s = torch.tensor([[2,1,4,3],[1,2,3,4],[4,3,2,1]])
print(s)
'''
'''
# operations
# elementwise operations
# operators can be lifted to elementwise operators
# so long as tensors are identically shaped?

x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
# can use +, -, *, /, **
print(x + y, x ** y)

# apply unary operator like exponentiation
# can specify the exponential with another tensor
print(torch.exp(x))

X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0,1,4,3],[1,2,3,4],[4,3,2,1]])
# concatenate
# dim 0 = row, dim 1 = column
XY = torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1)
print(XY)

# binary tensor
# true if the entry at the same position is the same
print(X == Y)
# sum all elements in a tensor
print(X.sum())


# boradcasting mechanism
# perform elementwise operations even when shapes differ
# need to copy elements so the tensors have the same shape
a = torch.arange(3).reshape((3, 1))
print(a)
b = torch.arange(2).reshape((1, 2))
print(b)

# broadcast entries in a and b to a 3x2 matriX (prev 3x1 and 1x2)
print(a + b)

# indexing and slicing
# -1 is the last element, 1:3 is the second and third
print(X[-1], X[1:3])

# write elements of a matrix by specifying indices
X[1, 2] = 9
print(X)
X[0:2, :] = 12
print(X)

# saving memory
# i don't really understand this section
# id gives exact location of referenced object in memory
before = id(Y)
print(before)
Y = Y + X
print(id(Y) == before)

# updates to objects should be done in place to prevent
# references from pointing to old memory location

# in-place operations
# zeros_like allocates a block of 0 entries
Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
# assign result of operation to previously allocated array
# with slice notation
Z[:] = X + Y
print('id(Z):', id(Z))

# use X[:] = X + Y or X += Y to 'reduce memory overhead'
before = id(X)
X += Y
print(id(X) == before)

# conversion to other python objects
# can change a NumPy tensor to a torch tensor
# keeping the same memory location
# using in-place operations
A = X.numpy()
B = torch.from_numpy(A)
print(type(A), type(B))

# convert tensor to scalar
a = torch.tensor([3.5])
print(a, a.item(), float(a), int(a))

print(X < Y)

a = torch.arange(1, 6, dtype=torch.float32).reshape((5,1))
b = torch.arange(1, 3).reshape((1, 2))
print(a, b)
print(a + b)
'''





























































