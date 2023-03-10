use array::ArrayTrait;
use traits::Into;
use option::OptionTrait;

use neural_network::matrix::Matrix;
use neural_network::matrix::MatrixTrait;

// 1. Take array
// 2. Calculate exponential for every item
// 3. Sum all the exponentials in the input vector
// 4. Divide each exponential value by the sum of exponentials

fn softmax(z: Matrix) -> Matrix {
        let mut exponential_arr = ArrayTrait::<u256>::new();
        
        // 2. Calculate exponential for every item
        exponential_inner(ref exponential_arr, z.data, 0_usize, z.data.len());

        // 3. Sum all the exponentials in the input vector
        let mut exp_sum: u256 = 0.into();
        exponential_sum_inner(ref exp_sum, exponential_arr, 0_usize, exponential_arr.len());

        // 4. Divide each exponential value by the sum of exponentials
        let mut divided_exponentials = ArrayTrait::<u256>::new();
        divide_exponentials_inner(ref divided_exponentials, exp_sum, 0_usize, exponential_arr.len());

        MatrixTrait::new(z.rows, z.cols, divided_exponentials)
    }

fn exponential_inner(ref exponential_arr: Array::<u256>, input: Array::<u256>, index: usize, len: usize) -> Array::<u256> {

    if index == len {
        return Option::None(());
    }

    // Replace exp by the pow function in quaireaux
    // let exponential = exp(*input.at(i));

    // Placeholder until function is replaced
    let val_1: u256 = 1.into();
    let exponential = *input.at(index) + val_1;

    exponential_arr.append(exponential);
    exponential_inner(ref exponential_arr, input, index + 1_usize, len)
}

fn exponential_sum_inner(ref exp_sum: u256, exponential_arr: Array::<u256>, index: usize, len: usize) -> Option::<u256> {

    if index == len {
        return Option::None(());
    }

    // Replace exp by the pow function in quaireaux
    exp_sum = exp_sum + *exponential_arr.at(index);
    exponential_sum_inner(ref exp_sum, exponential_arr, index + 1_usize, len)
}

fn divide_exponentials_inner(ref divided_exponentials: Array::<u256>, exp_sum: u256, index: usize, len: usize) -> Option::<Array::<u256>> {
    
    if index == len {
        return Option::None(());
    }

    let divided_exp = *divided_exponentials.at(index) / exp_sum;
    divided_exponentials.append(divided_exp)
}