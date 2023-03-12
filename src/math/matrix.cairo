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
    fn get(self: @Matrix, i: usize, j: usize) -> i33 ;
    fn dot(self: @Matrix, other: @Matrix) -> Matrix;
    fn add(self: @Matrix, other: @Matrix) -> Matrix;
    fn len(self: @Matrix) -> usize;
    // Need to add:
    // - exp
}

impl MatrixImpl of MatrixTrait {
    
    // #[inline(always)]
    fn new(rows: usize, cols: usize, data: Array::<i33>) -> Matrix {
        assert(data.len() == rows * cols, 'Matrix not match dimensions');
        matrix_new(rows, cols, data)
    }

    fn get(self: @Matrix, i: usize, j: usize) -> i33 {
        assert(i < *self.rows, 'row out of bounds');
        assert(j < *self.cols, 'column out of bounds');

        *self.data.at(i * *self.cols + j)
    }

    fn dot(self: @Matrix, other: @Matrix) -> Matrix {

        let mut arr = ArrayTrait::<i33>::new();
      
        _dot_inner(self, ref arr, other, 0_usize);

        MatrixTrait::new(*self.rows, *other.cols, arr)
    }

    fn add(self: @Matrix, other: @Matrix) -> Matrix {

        // assert(*self.rows == *other.rows, 'Matrix not match dimensions');
        // assert(*self.cols == *other.cols, 'Matrix not match dimensions');

        let mut arr = ArrayTrait::<i33>::new();
      
        _add_inner(self, ref arr, other, 0_usize);

        MatrixTrait::new(*self.rows, *self.cols, arr)
    }

    fn len(self: @Matrix) -> usize {
        self.data.len()
    }

}

fn matrix_new(rows: usize, cols: usize, data: Array::<i33>) -> Matrix {
    Matrix {
            rows: rows,
            cols: cols,
            data: data,
        }
}

// **********************
// * Matrix DOT product *
// **********************

fn _row_dot_vec(self: @Matrix, ref arr: Array::<i33>, other: @Matrix, row_index: usize, col_index: usize) -> i33 {

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

fn _row_add_vec(self: @Matrix, ref arr: Array::<i33>, other: @Matrix, row_index: usize, col_index: usize) {

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

