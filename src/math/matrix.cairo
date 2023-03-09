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
    fn dot(self: @Matrix, other: Matrix) -> Matrix;
    fn len(self: @Matrix) -> usize;
    // Need to add:
    // - add
    // - exp
}

impl MatrixImpl of MatrixTrait {
    
    #[inline(always)]
    fn new(rows: usize, cols: usize, data: Array::<i33>) -> Matrix {
        assert(data.len() == rows * cols, 'Matrix not match dimensions');
        matrix_new(rows, cols, data)
    }

    fn get(self: @Matrix, i: usize, j: usize) -> i33 {
        assert(i < *self.rows, 'row out of bounds');
        assert(j < *self.cols, 'column out of bounds');

        *self.data.at(i * *self.cols + j)
    }

    fn dot(self: @Matrix, other: Matrix) -> Matrix {

        let mut arr = ArrayTrait::<i33>::new();
        let mut row_index = 0_usize;
        let mut col_index = 0_usize;
        
        _dot_inner(self, ref arr, other, ref row_index, ref col_index);

        MatrixTrait::new(other.rows, other.cols, arr)
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

fn _row_dot_vec(self: @Matrix, ref arr: Array::<i33>, other: Matrix, ref row_index: usize, ref col_index: usize) -> i33 {

    // --- End of the recursion ---
    if (col_index == other.cols) {
        return (i33 { inner: 0_u32, sign: true });
    }

    // --- Calculates the product ---
    let ele = self.data.get(other.cols * other.rows + col_index);
    let result = ele * other.data.get(col_index);

    col_index = col_index + 1_usize;
    let acc = _row_dot_vec(self, ref arr, other, ref row_index, ref col_index);

    // --- Returns the sum of the current product with the previous ones ---
    return acc + result;
}


fn _dot_inner(self: @Matrix, ref arr: Array::<i33>, other: Matrix, ref row_index: usize, ref col_index: usize) {

    // --- End of the recursion ---
    if row_index == other.rows {
        return ();
    }

    // --- Compute dot product of the row ---
    let dot = _row_dot_vec(self, ref arr, other, ref row_index, ref col_index);

    arr.append(dot);
    row_index = row_index + 1_usize;
    _dot_inner(self, ref arr, other, ref row_index, ref col_index)

}
