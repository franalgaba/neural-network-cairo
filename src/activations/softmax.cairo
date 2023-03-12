use array::ArrayTrait;
use traits::Into;
use option::OptionTrait;

use neural_network::math::matrix::Matrix;
use neural_network::math::matrix::MatrixTrait;

use neural_network::math::int33;
use neural_network::math::int33::i33;

// 1. Take array
// 2. Calculate exponential for every item
// 3. Sum all the exponentials in the input vector
// 4. Divide each exponential value by the sum of exponentials

// The implementation is using pseudo-softmax for now
fn softmax(z: Matrix) -> Matrix {
    let mut exponential_arr = ArrayTrait::<i33>::new();

    // 2. Calculate exponential for every item
    exponential_inner(ref exponential_arr, z.data, 0_usize, z.data.len());

    // 3. Sum all the exponentials in the input vector
    let mut exp_sum = i33 { inner: 0_u32, sign: true };
    exponential_sum_inner(ref exp_sum, exponential_arr, 0_usize, exponential_arr.len());

    // 4. Divide each exponential value by the sum of exponentials
    let mut divided_exponentials = ArrayTrait::<i33>::new();
    divide_exponentials_inner(ref divided_exponentials, exp_sum, 0_usize, exponential_arr.len());

    MatrixTrait::new(z.rows, z.cols, divided_exponentials)
}

fn exponential_inner(
    ref exponential_arr: Array::<i33>, input: Array::<i33>, index: usize, len: usize
) -> Array::<i33> {
    if index == len {
        return (i33 { inner: 0_u32, sign: true });
    }

    // Replace exp by the pow function in quaireaux
    // let exponential = exp(*input.at(i));

    // Placeholder until function is replaced
    let val_1 = i33 { inner: 1_u32, sign: false };
    let exponential = *input.at(index) + val_1;

    exponential_arr.append(exponential);
    exponential_inner(ref exponential_arr, input, index + 1_usize, len)
}

fn exponential_sum_inner(
    ref exp_sum: i33, exponential_arr: Array::<i33>, index: usize, len: usize
) -> i33 {
    if index == len {
        return (i33 { inner: 0_u32, sign: true });
    }

    // Replace exp by the pow function in quaireaux
    exp_sum = exp_sum + *exponential_arr.at(index);
    exponential_sum_inner(ref exp_sum, exponential_arr, index + 1_usize, len)
}

fn divide_exponentials_inner(
    ref divided_exponentials: Array::<i33>, exp_sum: i33, index: usize, len: usize
) -> Array::<i33> {
    if index == len {
        return (i33 { inner: 0_u32, sign: true });
    }

    let divided_exp = *divided_exponentials.at(index) / exp_sum;
    divided_exponentials.append(divided_exp)
}
