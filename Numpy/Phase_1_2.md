# 🚀 NumPy Complete Guide - Phase 1 & 2

> A comprehensive guide to NumPy fundamentals and intermediate operations

---

## 📋 Table of Contents

- [Data Structures Overview](#data-structures-overview)
- [Phase 1: Foundation](#phase-1-foundation)
  - [Creating Arrays](#creating-arrays)
  - [List vs NumPy Array](#list-vs-numpy-array)
  - [Creating Arrays from Scratch](#creating-arrays-from-scratch)
  - [Understanding Dimensions](#understanding-dimensions)
  - [Array Properties](#array-properties)
  - [Array Reshaping](#array-reshaping)
- [Phase 2: Intermediate Operations](#phase-2-intermediate-operations)
  - [Indexing and Slicing](#indexing-and-slicing)
  - [Sorting Arrays](#sorting-arrays)
  - [Filtering and Conditionals](#filtering-and-conditionals)
  - [Adding and Removing Data](#adding-and-removing-data)

---

## 📊 Data Structures Overview

NumPy works with three main data structures:

### 📏 Vector (1D Array)
```python
[1, 2, 3, 4, 5, 6]
```
A **one-dimensional** array of numbers

### 📐 Matrix (2D Array)
```python
[[1, 2],
 [3, 4]]
```
A **two-dimensional** array with rows and columns

### 🧊 Tensors (3D+ Arrays)
```python
[[[1, 2], [3, 4]],
 [[5, 6], [7, 8]]]
```
**Multi-dimensional** arrays (3D and beyond)

### 💡 Quick Reference

| Type | Dimensions | Example |
|------|------------|---------|
| **Vector** | 1D | `[1, 2, 3, 4, 5, 6]` |
| **Matrix** | 2D | `[[1, 2], [3, 4]]` |
| **Tensors** | 3D+ | Multi-dimensional arrays |

---

## 📦 PHASE 1: Foundation

> **NumPy Arrays and Basics**

### Getting Started

```python
import numpy as np
```

---

## 🎯 Creating Arrays

### From Python Lists

Convert a Python list to a NumPy array:

```python
# 1D Array
arr_1d = np.array([1, 2, 3, 4, 5])
# Output: array([1, 2, 3, 4, 5])

# 2D Array
arr_2d = np.array([[1, 2], [3, 4]])
# Output: array([[1, 2],
#                [3, 4]])
```

**Key Point:** Transform Python lists into powerful NumPy arrays using `np.array()`

---

## ⚖️ List vs NumPy Array

### Comparison Table

| Feature | Python List | NumPy Array |
|---------|-------------|-------------|
| **Speed** | Slower ⏱️ | ⚡ Faster (50-100x) |
| **Memory** | More memory usage | 💾 Compact & efficient |
| **Operations** | Limited | Mathematical operations |
| **Type** | Mixed types allowed | Homogeneous (same type) |
| **Syntax** | Basic operations | Vectorized operations |

### 🔄 Behavior Comparison

#### Python List - Concatenation
```python
py_list = [1, 2, 3]
py_list *= 2
# Output: [1, 2, 3, 1, 2, 3]
```
Lists **repeat** elements when multiplied

#### NumPy Array - Element-wise Operation
```python
np_array = np.array([1, 2, 3])
np_array *= 2
# Output: [2, 4, 6]
```
Arrays perform **mathematical operations** on each element

### ⏱️ Performance Example

```python
import time

# List operation
start = time.time()
py_list = list(range(1000000))
result = [x * 2 for x in py_list]
print(f"List time: {time.time() - start:.4f}s")

# NumPy operation
start = time.time()
np_array = np.arange(1000000)
result = np_array * 2
print(f"NumPy time: {time.time() - start:.4f}s")
```

**💡 NumPy is significantly faster for large-scale numerical operations!**

---

## 🔨 Creating Arrays from Scratch

### `np.zeros()` - Array of Zeros

```python
zeros = np.zeros((3, 4))
# Output: [[0. 0. 0. 0.]
#          [0. 0. 0. 0.]
#          [0. 0. 0. 0.]]
```
Creates a **3×4 array** filled with zeros

### `np.ones()` - Array of Ones

```python
ones = np.ones((2, 3))
# Output: [[1. 1. 1.]
#          [1. 1. 1.]]
```
Creates a **2×3 array** filled with ones

### `np.full()` - Array with Custom Value

```python
full = np.full((2, 2), 7)
# Output: [[7 7]
#          [7 7]]
```
Creates a **2×2 array** filled with **7**

### `np.random.random()` - Random Values

```python
randoms = np.random.random((2, 3))
# Output: [[0.234 0.876 0.123]
#          [0.987 0.456 0.789]]
```
Creates a **2×3 array** with random values between **0 and 1**

### `np.arange()` - Sequence with Steps

```python
sequence = np.arange(0, 11, 3)  # (start, stop, step)
# Output: [0 3 6 9]
```
Creates a sequence from **0 to 10** with a **step of 3**

---

## 🔢 Understanding Dimensions

### 📏 Vector (1D Array)

```python
vector = np.array([1, 2, 3])
# Output: [1 2 3]
```
A **one-dimensional** array

### 📐 Matrix (2D Array)

```python
matrix = np.array([[1, 2], [3, 4]])
# Output: [[1 2]
#          [3 4]]
```
A **two-dimensional** array with rows and columns

### 🧊 Tensor (3D Array)

```python
tensor = np.array([[[1, 2], [3, 4]], [[4, 5], [6, 7]]])
# Output: [[[1 2]
#           [3 4]]
#
#          [[4 5]
#           [6 7]]]
```
A **multi-dimensional** array (3D and beyond)

### 💡 Dimension Reference

| Type | Dimensions | Shape | Example |
|------|------------|-------|---------|
| **Vector** | 1D | `(3,)` | `[1, 2, 3]` |
| **Matrix** | 2D | `(2, 2)` | `[[1, 2], [3, 4]]` |
| **Tensor** | 3D+ | `(2, 2, 2)` | `[[[1, 2], [3, 4]], [[4, 5], [6, 7]]]` |

---

## 📊 Array Properties

### Core Properties

```python
arr = np.array([1, 2, 3])
```

#### 📷 `shape` - Dimensions of the Array

```python
arr.shape
# Output: (3,)
```
**Shape** tells you the size along each dimension  
`(3,)` means 3 elements in a 1D array

#### 📷 `ndim` - Number of Dimensions

```python
arr.ndim
# Output: 1
```
**ndim** tells you how many dimensions (axes) the array has  
`1` means it's a 1D array (vector)

#### 📷 `size` - Total Number of Elements

```python
arr.size
# Output: 3
```
**size** tells you the total count of elements  
`3` means there are 3 elements in total

#### 📷 `dtype` - Data Type of Elements

```python
arr.dtype
# Output: dtype('int64')
```
**dtype** tells you the data type of array elements  
`int64` means 64-bit integers

---

### 🧪 Type Conversion Examples

#### Mixed Integer and Boolean

```python
arr_2 = np.array([1, 2, True])
# Output: [1 2 1]

arr_2.dtype
# Output: dtype('int64')
```
✅ **Works!** Boolean `True` converts to `1`, all become integers

#### Mixed Float, Integer, and Boolean

```python
arr_3 = np.array([1, 4.5, True])
# Output: [1.  4.5 1. ]

arr_3.dtype
# Output: dtype('float64')
```
✅ **Works!** All values convert to float (highest precision type)

---

### 📋 Complete Example

```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])

matrix.shape   # (2, 3)    → 2 rows, 3 columns
matrix.ndim    # 2         → 2 dimensions (matrix)
matrix.size    # 6         → 6 total elements
matrix.dtype   # int64     → 64-bit integers
```

### 💡 Properties Reference

| Property | What it tells you | Example Output |
|----------|-------------------|----------------|
| **shape** | Size along each dimension | `(3,)` or `(2, 3)` |
| **ndim** | Number of dimensions | `1`, `2`, `3` |
| **size** | Total element count | `6`, `12` |
| **dtype** | Data type of elements | `int64`, `float64` |

### 🎯 Key Takeaway

> NumPy automatically converts mixed types to the **most general type** to maintain array homogeneity!

---

## 🔄 Array Reshaping

### Creating the Array

```python
arr = np.arange(12)
# Output: [0 1 2 3 4 5 6 7 8 9 10 11]
```
Creates a 1D array with values from **0 to 11**

---

### 📐 `reshape()` - Change Array Dimensions

```python
reshaped = arr.reshape((3, 4))
# Output: [[ 0  1  2  3]
#          [ 4  5  6  7]
#          [ 8  9 10 11]]
```
**Reshape** transforms the array into a new shape without changing data  
Converts `(12,)` → `(3, 4)` (3 rows × 4 columns)

---

### 📏 `flatten()` - Convert to 1D (Copy)

```python
flattened = reshaped.flatten()
# Output: [0 1 2 3 4 5 6 7 8 9 10 11]
```
**Flatten** converts multi-dimensional array to 1D  
Creates a **new copy** of the data

---

### 📏 `ravel()` - Convert to 1D (View)

```python
raveled = reshaped.ravel()
# Output: [0 1 2 3 4 5 6 7 8 9 10 11]
```
**Ravel** also converts to 1D array  
Returns a **view** (reference) when possible, not a copy

---

### 🔃 `T` - Transpose (Flip Rows ↔ Columns)

```python
transpose = reshaped.T
# Output: [[ 0  4  8]
#          [ 1  5  9]
#          [ 2  6 10]
#          [ 3  7 11]]
```
**Transpose** swaps rows and columns  
Shape changes from `(3, 4)` → `(4, 3)`

---

### 💡 Reshaping Methods Comparison

| Method | What it does | Creates Copy? | Shape Change |
|--------|--------------|---------------|--------------|
| **reshape()** | Changes dimensions | No (view) | `(12,)` → `(3, 4)` |
| **flatten()** | Converts to 1D | ✅ Yes | `(3, 4)` → `(12,)` |
| **ravel()** | Converts to 1D | ❌ No (view) | `(3, 4)` → `(12,)` |
| **T** | Swaps rows/columns | No (view) | `(3, 4)` → `(4, 3)` |

### 🎯 Key Difference: `flatten()` vs `ravel()`

> **flatten()** creates a new copy (safer, uses more memory)  
> **ravel()** creates a view (faster, shares memory with original)

---

## 📦 PHASE 2: Intermediate Operations

> **Advanced Array Manipulation & Techniques**

---

## 🎯 Indexing and Slicing

### 1D Array Slicing

```python
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
```

#### 📌 Basic Slicing

```python
arr[2:7]
# Output: [3 4 5 6 7]
```
Slices from index **2 to 6** (7-1)  
Format: `[start:stop]` (stop is exclusive)

#### 🪜 Slicing with Stepping

```python
arr[0:10:2]
# Output: [1 3 5 7 9]
```
Takes every **2nd element**  
Format: `[start:stop:step]`

#### 🔄 Negative Indexing (Reverse)

```python
arr[::-1]
# Output: [9 8 7 6 5 4 3 2 1]
```
**Reverses** the entire array  
Step of `-1` goes backwards

---

### 📐 2D Array Slicing

```python
arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
```

#### 🎯 Accessing Specific Element

```python
arr_2d[0, 2]  # NumPy style (preferred)
# Output: 3

arr_2d[0][2]  # Also works, but less efficient
# Output: 3
```
Gets element at **row 0, column 2**  
✨ NumPy allows `[row, col]` syntax!

#### 📊 Accessing Entire Row

```python
arr_2d[0]
# Output: [1 2 3]
```
Gets the **first row** (row index 0)

#### 📊 Accessing Entire Column

```python
arr_2d[:, 0]
# Output: [1 4 7]
```
Gets the **first column** (column index 0)  
`:` means "all rows"

```python
arr_2d[:, 1]
# Output: [2 5 8]
```
Gets the **second column**

---

## 🔃 Sorting Arrays

### 1D Array Sorting

```python
unsorted_arr = np.array([3, 2, 6, 4, 3, 2, 6])
sorted_arr = np.sort(unsorted_arr)
# Output: [2 2 3 3 4 6 6]
```
Sorts array in **ascending order**

---

### 2D Array Sorting

```python
arr_2d_unsorted = np.array([[1, 3],
                            [4, 1],
                            [19, 11]])
```

#### 📏 Column-wise Sorting (`axis=0`)

```python
arr_2d_sorted_col_wise = np.sort(arr_2d_unsorted, axis=0)
# Output: [[ 1  1]
#          [ 4  3]
#          [19 11]]
```

**axis=0** sorts **down the columns** (vertically)

**How it works:**
- Column 0: `[1, 4, 19]` → sorted → `[1, 4, 19]` ✓
- Column 1: `[3, 1, 11]` → sorted → `[1, 3, 11]` ✓

#### 📏 Row-wise Sorting (`axis=1`)

```python
arr_2d_sorted_row_wise = np.sort(arr_2d_unsorted, axis=1)
# Output: [[ 1  3]
#          [ 1  4]
#          [11 19]]
```

**axis=1** sorts **across the rows** (horizontally)

**How it works:**
- Row 0: `[1, 3]` → sorted → `[1, 3]` ✓
- Row 1: `[4, 1]` → sorted → `[1, 4]` ✓
- Row 2: `[19, 11]` → sorted → `[11, 19]` ✓

### 💡 Understanding Axis

| Axis | Direction | What it does |
|------|-----------|--------------|
| **axis=0** | ↓ Vertical | Sorts **down columns** |
| **axis=1** | → Horizontal | Sorts **across rows** |

> 🎯 **Remember**: `axis=0` = rows, `axis=1` = columns

---

## 🔍 Filtering and Conditionals

### Creating the Array

```python
num = np.arange(10)
# Output: [0 1 2 3 4 5 6 7 8 9]
```

---

### 🎯 Method 1: Direct Boolean Filtering

```python
even_nums = num[num % 2 == 0]
# Output: [0 2 4 6 8]
```

**How it works:**
1. `num % 2 == 0` checks each element
2. Creates boolean array: `[True, False, True, False, ...]`
3. Returns only elements where condition is `True`

---

### 🎭 Method 2: Using Masks

#### What is a Mask?

```python
mask = num > 5
# Output: [False False False False False False True True True True]
```

A **mask** is a boolean array that filters data  
`True` = include, `False` = exclude

#### Applying the Mask

```python
num[mask]
# Output: [6 7 8 9]
```

**Visual Representation:**
```
Array:  [0  1  2  3  4  5  6  7  8  9]
Mask:   [F  F  F  F  F  F  T  T  T  T]
Result: [               6  7  8  9]
```

---

### 🎪 Fancy Indexing

```python
indices = [0, 1, 4, 3, 4]
num[indices]
# Output: [0 1 4 3 4]
```

**Fancy indexing** selects elements by their index positions  
Can access elements in **any order** and **repeat** them!

**Visual Representation:**
```
Array:   [0  1  2  3  4  5  6  7  8  9]
Indices:  0  1     3  4     (pick these)
Result:  [0  1     3  4  4]
```

---

### 🔎 `np.where()` - Finding Indices

```python
where = np.where(num > 5)
# Output: (array([6, 7, 8, 9]),)
```

`np.where()` returns **indices** where condition is `True`  
Unlike filtering, it gives you the **positions**, not values

**Comparison:**

| Method | Returns | Example Output |
|--------|---------|----------------|
| `num[num > 5]` | **Values** | `[6, 7, 8, 9]` |
| `np.where(num > 5)` | **Indices** | `(array([6, 7, 8, 9]),)` |

---

### 🔀 Conditional Array with `np.where()`

#### Example 1: Conditional Transformation

```python
cond_arr = np.where(num > 5, num * 2, num)
# Output: [0 1 2 3 4 5 12 14 16 18]
```

**Step-by-Step Explanation:**
```
If condition is True  → use first value (num * 2)
If condition is False → use second value (num)
```

**Visual Breakdown:**
```
Array:     [0  1  2  3  4  5  6   7   8   9]
Condition: [F  F  F  F  F  F  T   T   T   T]  (num > 5)
If True:                      12  14  16  18  (num * 2)
If False:  [0  1  2  3  4  5]                 (num)
Result:    [0  1  2  3  4  5  12  14  16  18]
```

#### Example 2: No Transformation

```python
cond_arr = np.where(num > 5, num, num)
# Output: [0 1 2 3 4 5 6 7 8 9]
```

Both True and False return `num`, so **no change**

#### Example 3: String Labels

```python
cond_arr = np.where(num > 5, "true", "false")
# Output: ['false' 'false' 'false' 'false' 'false' 'false'
#          'true' 'true' 'true' 'true']
```

**Visual Breakdown:**
```
Array:     [0  1  2  3  4  5  6      7      8      9]
Condition: [F  F  F  F  F  F  T      T      T      T]
Result:    [false false false false false false true true true true]
```

---

### 💡 `np.where()` Syntax

```python
np.where(condition, value_if_true, value_if_false)
```

**Think of it as:**
```python
if (condition):
    return value_if_true
else:
    return value_if_false
```

But applied to **every element** in the array!

---

### 📊 Filtering Quick Reference

| Operation | Syntax | Returns | Use Case |
|-----------|--------|---------|----------|
| **Boolean Filtering** | `arr[arr > 5]` | Values | Get filtered values |
| **Mask** | `mask = arr > 5` | Boolean array | Reusable condition |
| **Fancy Indexing** | `arr[[0,2,4]]` | Values | Select specific indices |
| **np.where (indices)** | `np.where(arr > 5)` | Indices | Find positions |
| **np.where (conditional)** | `np.where(arr > 5, x, y)` | Values | Transform conditionally |

---

## ➕➖ Adding and Removing Data

### Creating Arrays

```python
arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([4, 5, 6, 7])
arr3 = np.array([10, 11, 12, 13])
```

---

### 🔢 Combining Arrays

#### Element-wise Addition

```python
combined = arr1 + arr2 + arr2
# Output: [9 12 15 18]
```

**Step-by-step:**
```
arr1:     [1  2  3  4]
arr2:     [4  5  6  7]
arr2:     [4  5  6  7]
         ---------------
Result:   [9 12 15 18]
```

Performs **element-wise addition** (not concatenation!)

#### Concatenation (Joining Arrays)

```python
combined = np.concatenate((arr1, arr2, arr3))
# Output: [1 2 3 4 4 5 6 7 10 11 12 13]
```

`np.concatenate()` **joins arrays end-to-end**

**Visual:**
```
arr1: [1 2 3 4]
arr2:          [4 5 6 7]
arr3:                   [10 11 12 13]
Result: [1 2 3 4 4 5 6 7 10 11 12 13]
```

---

### 🔍 Checking Shape Compatibility

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = np.array([7, 9])

a.shape == b.shape == c.shape
# Output: False

a.shape == b.shape
# Output: True
```

**Shapes:**
```
a.shape = (3,)  ✓
b.shape = (3,)  ✓  Same shape!
c.shape = (2,)  ✗  Different!
```

Arrays must have **compatible shapes** for element-wise operations

---

### 📊 Adding Rows and Columns to 2D Arrays

#### Original Array

```python
original = np.array([[1, 2],
                     [3, 4]])
# Output: [[1 2]
#          [3 4]]
```

#### 🔥 `np.vstack()` - Adding Rows (Vertical Stack)

```python
newRow = np.array([[5, 6]])
with_new_row = np.vstack((original, newRow))
# Output: [[1 2]
#          [3 4]
#          [5 6]]
```

**Visual Representation:**
```
Original:  [[1 2]     New Row:  [[5 6]]
            [3 4]]

Result:    [[1 2]
            [3 4]
            [5 6]]  ← New row added at bottom
```

**vstack** = **V**ertical stack (adds **rows**)

#### 🔥 `np.hstack()` - Adding Columns (Horizontal Stack)

```python
newCol = np.array([[7],
                   [8]])
with_new_col = np.hstack((original, newCol))
# Output: [[1 2 7]
#          [3 4 8]]
```

**Visual Representation:**
```
Original:  [[1 2]     New Col:  [[7]
            [3 4]]                [8]]

Result:    [[1 2 7]  ← New column added on right
            [3 4 8]]
```

**hstack** = **H**orizontal stack (adds **columns**)

### 💡 Quick Memory Aid

```
vstack → Vertical   → Rows    ↓
hstack → Horizontal → Columns →
```

---

### 🗑️ Deleting Elements

```python
arr = np.array([1, 2, 3, 4, 5])
deleted = np.delete(arr, 2)  # Remove element at index 2
# Output: [1 2 4 5]
```

**Visual:**
```
Original: [1  2  3  4  5]
           0  1  2  3  4  (indices)
                ↑
          Delete index 2

Result:   [1  2  4  5]
```

**Important:** Returns a **new array** without the element  
The element at **index 2** (which is `3`) is removed

---

### 📋 Operations Quick Reference

| Operation | Function | What it does |
|-----------|----------|--------------|
| **Element-wise add** | `arr1 + arr2` | Adds corresponding elements |
| **Concatenate** | `np.concatenate()` | Joins arrays end-to-end |
| **Add rows** | `np.vstack()` | Stacks arrays vertically (↓) |
| **Add columns** | `np.hstack()` | Stacks arrays horizontally (→) |
| **Delete** | `np.delete(arr, idx)` | Removes element at index |

### 🎯 Key Points

- ✅ `arr1 + arr2` does **math**, not joining
- ✅ Use `concatenate()` to **join** arrays
- ✅ **vstack** for rows, **hstack** for columns
- ✅ `delete()` returns **new array**, doesn't modify original

---

## 🎓 Summary

This guide covered:

### Phase 1: Foundation
- Creating arrays from lists and scratch
- Understanding vectors, matrices, and tensors
- Array properties (shape, ndim, size, dtype)
- Reshaping operations (reshape, flatten, ravel, transpose)

### Phase 2: Intermediate Operations
- Indexing and slicing (1D and 2D)
- Sorting arrays (axis=0 and axis=1)
- Filtering with boolean indexing and masks
- Conditional operations with `np.where()`
- Adding and removing data (concatenate, vstack, hstack, delete)

---

## 🚀 Next Steps

Continue your NumPy journey by exploring:
- Broadcasting
- Universal functions (ufuncs)
- Linear algebra operations
- Statistical operations
- Advanced indexing techniques

---

**Happy coding! 🎉**
