use array::ArrayTrait;
use option::OptionTrait;

struct Matrix<T> {
    rows: usize,
    cols: usize,
    data: Array::<T>,
}

trait MatrixTrait<T> {
    fn new(rows: usize, cols: usize, data: Array::<T>) -> Matrix::<T>;
    fn get(self: Matrix::<T>, i: usize, j: usize) -> T ;
    fn len(self: Matrix::<T>) -> usize;
}

impl MatrixImpl<T> of MatrixTrait::<T> {
    
    #[inline(always)]
    fn new(rows: usize, cols: usize, data: Array::<T>) -> Matrix::<T> {
        assert(data.len() == rows * cols, 'Matrix not match dimensions');
        matrix_new(rows_usize, cols_usize, data)
    }

    fn get(self: @Matrix::<T>, i: usize, j: usize) -> T {
        assert(i < self.rows, 'ROW OUT OF BOUNDS');
        assert(j < self.cols, 'COLUMN OUT OF BOUNDS');

        self.data.get(i_usize * self.cols + j_usize)
    }

    fn len(self: Matrix::<T>) -> usize {
        self.data.len()
    }
}

fn matrix_new<T>(rows: usize, cols: usize, data: Array::<T>) -> Matrix::<T> {
    Matrix::<T> {
            rows: rows_usize,
            cols: cols_usize,
            data: data,
        }
}

impl MatrixCopy of Copy::<Matrix>;
impl MatrixDrop of Drop::<Matrix>;