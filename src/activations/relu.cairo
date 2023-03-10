use array::ArrayTrait;
use traits::Into;
use option::OptionTrait;

use neural_network::matrix::Matrix;
use neural_network::matrix::MatrixTrait;


fn relu(z: Matrix) -> Matrix {
        let mut arr = ArrayTrait::<u256>::new();
        
        relu_inner(ref arr, z.data, 0_usize, z.data.len());
        MatrixTrait::new(z.rows, z.cols, arr)
        // if x > 0 -> x
        // if x < 0 -> 0
        
    }

fn relu_inner(ref arr: Array::<u256>, input: Array::<u256>, index: usize, len: usize) -> Option::<Array::<u256>> {

    if index == len {
        return Option::None(());
    }

    let val_0: u256 = 0.into();

    if *arr.at(index) > val_0 {
        arr.append(*input.at(index));
    }
    else {
        arr.append(val_0);
    }
    relu_inner(ref arr, input, index + 1_usize, len)
}
