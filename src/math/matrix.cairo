use array::ArrayTrait;
use option::OptionTrait;

use neural_network::math::int33;
use neural_network::math::int33::i33;

impl Arrayi33Drop of Drop::<Array::<i33>>;

#[derive(Drop)]
struct Matrix {
    rows: usize,
    cols: usize,
    data: Array::<i33>,
}

trait MatrixTrait {
    fn new(rows: usize, cols: usize, data: Array::<i33>) -> Matrix;
    fn get(self: @Matrix, i: usize, j: usize) -> i33;
    fn dot(self: @Matrix, other: @Matrix) -> Matrix;
    fn add(self: @Matrix, other: @Matrix) -> Matrix;
    fn len(self: @Matrix) -> usize;
}

impl MatrixImpl of MatrixTrait {
    fn new(rows: usize, cols: usize, data: Array::<i33>) -> Matrix {
        assert(data.len() == rows * cols, 'Matrix not match dimensions');
        matrix_new(rows, cols, data)
    }

    /// Gets the value of a particular element in the matrix.
    ///
    /// # Arguments
    ///
    /// * `self` - A reference to the matrix.
    /// * `i` - The row index of the element.
    /// * `j` - The column index of the element.
    ///
    /// # Returns
    ///
    /// The value of the element at the specified row and column.
    ///
    /// # Panics
    ///
    /// Panics if the row index or column index is out of bounds.
    fn get(self: @Matrix, i: usize, j: usize) -> i33 {
        assert(i < *self.rows, 'row out of bounds');
        assert(j < *self.cols, 'column out of bounds');

        *self.data.at(i * *self.cols + j)
    }

    /// Computes the matrix dot product of this matrix and another matrix.
    ///
    /// # Arguments
    ///
    /// * `self` - A reference to the left-hand matrix.
    /// * `other` - A reference to the right-hand matrix.
    ///
    /// # Returns
    ///
    /// A new matrix that is the matrix dot product of this matrix and the other matrix.
    ///
    /// # Panics
    ///
    /// Panics if the number of columns in the left-hand matrix does not match the number of rows in the right-hand matrix.
    fn dot(self: @Matrix, other: @Matrix) -> Matrix {
        let mut arr = ArrayTrait::<i33>::new();

        _dot_inner(self, ref arr, other, 0_usize);

        MatrixTrait::new(*self.rows, *other.cols, arr)
    }

    /// Adds another matrix to this matrix element-wise.
    ///
    /// # Arguments
    ///
    /// * `self` - A reference to the matrix.
    /// * `other` - A reference to the matrix to be added.
    ///
    /// # Returns
    ///
    /// A new matrix that is the element-wise sum of this matrix and the other matrix.
    ///
    /// # Panics
    ///
    /// Panics if the input matrices have different dimensions.
    fn add(self: @Matrix, other: @Matrix) -> Matrix {
        assert(*self.rows == *other.rows, 'Matrix not match dimensions');
        assert(*self.cols == *other.cols, 'Matrix not match dimensions');

        let mut arr = ArrayTrait::<i33>::new();

        _add_inner(self, ref arr, other, 0_usize);

        MatrixTrait::new(*self.rows, *self.cols, arr)
    }

    /// Returns the number of elements in the matrix.
    ///
    /// # Arguments
    ///
    /// * `self` - A reference to the matrix.
    ///
    /// # Returns
    ///
    /// The number of elements in the matrix.
    fn len(self: @Matrix) -> usize {
        self.data.len()
    }
}

fn matrix_new(rows: usize, cols: usize, data: Array::<i33>) -> Matrix {
    Matrix { rows: rows, cols: cols, data: data,  }
}

// **********************
// * Matrix DOT product *
// **********************

fn _row_dot_vec(
    self: @Matrix, ref arr: Array::<i33>, other: @Matrix, row_index: usize, col_index: usize
) -> i33 {
    // TODO: Remove when automatically handled by compiler.
    match gas::get_gas() {
        Option::Some(_) => {},
        Option::None(_) => {
            let mut data = array_new::<felt>();
            array_append::<felt>(ref data, 'OOG');
            panic(data);
        },
    }

    // End of the recursion
    if (col_index == *self.cols) {
        return (i33 { inner: 0_u32, sign: true });
    }

    let mut ele = i33 { inner: 0_u32, sign: true };
    // Calculates the product
    match self.data.get(*self.cols * row_index + col_index) {
        Option::Some(x) => {
            ele = *x;
        },
        Option::None(_) => {
            let mut data = ArrayTrait::new();
            data.append('out of bounds');
            panic(data)
        }
    }

    let mut other_ele = i33 { inner: 0_u32, sign: true };
    match other.data.get(col_index) {
        Option::Some(x) => {
            other_ele = *x;
        },
        Option::None(_) => {
            let mut data = ArrayTrait::new();
            data.append('out of bounds');
            panic(data)
        }
    }
    let result = ele * other_ele;
    let acc = _row_dot_vec(self, ref arr, other, row_index, col_index + 1_usize);

    // Returns the sum of the current product with the previous ones
    return acc + result;
}


fn _dot_inner(self: @Matrix, ref arr: Array::<i33>, other: @Matrix, row_index: usize) {
    // TODO: Remove when automatically handled by compiler.
    match gas::get_gas() {
        Option::Some(_) => {},
        Option::None(_) => {
            let mut data = array_new::<felt>();
            array_append::<felt>(ref data, 'OOG');
            panic(data);
        },
    }

    // End of the recursion
    if row_index == *self.rows {
        return ();
    }

    // Compute dot product of the row
    let dot = _row_dot_vec(self, ref arr, other, row_index, 0_usize);

    arr.append(dot);
    _dot_inner(self, ref arr, other, row_index + 1_usize);
}

// **************
// * Matrix ADD *
// **************

fn _row_add_vec(
    self: @Matrix, ref arr: Array::<i33>, other: @Matrix, row_index: usize, col_index: usize
) {
    // TODO: Remove when automatically handled by compiler.
    match gas::get_gas() {
        Option::Some(_) => {},
        Option::None(_) => {
            let mut data = array_new::<felt>();
            array_append::<felt>(ref data, 'OOG');
            panic(data);
        },
    }

    // End of the recursion
    if (col_index == *self.cols) {
        return ();
    }

    let mut ele = i33 { inner: 0_u32, sign: true };
    // Calculates the product
    match self.data.get(*self.cols * row_index + col_index) {
        Option::Some(x) => {
            ele = *x;
        },
        Option::None(_) => {
            let mut data = ArrayTrait::new();
            data.append('out of bounds');
            panic(data)
        }
    }

    let mut other_ele = i33 { inner: 0_u32, sign: true };
    match other.data.get(*other.cols * row_index + col_index) {
        Option::Some(x) => {
            other_ele = *x;
        },
        Option::None(_) => {
            let mut data = ArrayTrait::new();
            data.append('out of bounds');
            panic(data)
        }
    }
    arr.append(ele + other_ele);
    _row_add_vec(self, ref arr, other, row_index, col_index + 1_usize);
}


fn _add_inner(self: @Matrix, ref arr: Array::<i33>, other: @Matrix, row_index: usize) {
    // TODO: Remove when automatically handled by compiler.
    match gas::get_gas() {
        Option::Some(_) => {},
        Option::None(_) => {
            let mut data = array_new::<felt>();
            array_append::<felt>(ref data, 'OOG');
            panic(data);
        },
    }

    // End of the recursion
    if row_index == *self.rows {
        return ();
    }

    // Compute dot product of the row
    _row_add_vec(self, ref arr, other, row_index, 0_usize);

    _add_inner(self, ref arr, other, row_index + 1_usize);
}

