use array::ArrayTrait;
use option::OptionTrait;

#[derive(Drop)]
struct Matrix {
    rows: usize,
    cols: usize,
    data: Array::<u256>,
}

trait MatrixTrait {
    fn new(rows: usize, cols: usize, data: Array::<u256>) -> Matrix;
    fn get(self: @Matrix, i: usize, j: usize) -> Option::<u256> ;
    fn len(self: @Matrix) -> usize;
}

impl MatrixImpl of MatrixTrait {
    
    #[inline(always)]
    fn new(rows: usize, cols: usize, data: Array::<u256>) -> Matrix {
        assert(data.len() == rows * cols, 'Matrix not match dimensions');
        matrix_new(rows, cols, data)
    }

    fn get(self: @Matrix, i: usize, j: usize) -> Option::<u256> {
        assert(i < *self.rows, 'row out of bounds');
        assert(j < *self.cols, 'column out of bounds');

        Option::Some(*self.data.at(i * *self.cols + j))
    }

    fn len(self: @Matrix) -> usize {
        self.data.len()
    }
}

fn matrix_new(rows: usize, cols: usize, data: Array::<u256>) -> Matrix {
    Matrix {
            rows: rows,
            cols: cols,
            data: data,
        }
}