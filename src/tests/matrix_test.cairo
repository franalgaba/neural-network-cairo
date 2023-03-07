use array::ArrayTrait;
use option::OptionTrait;

use neural_network::matrix::Matrix;
use neural_network::matrix::MatrixTrait;

#[test]
#[available_gas(2000000)]
fn matrix_new_test() {
    let mut arr = ArrayTrait::<felt>::new();
    arr.append(1);
    arr.append(2);
    arr.append(3);
    arr.append(4);
    arr.append(5);
    arr.append(6);
    arr.append(7);
    arr.append(8);

    let mut matrix = MatrixTrait::<felt>::new(4_usize, 4_usize, arr);    
    let result_len = MatrixTrait::<felt>::len(matrix);

    assert(result_len == 8_usize, 'correct length');
}

#[test]
#[available_gas(2000000)]
fn matrix_get_test() {

    let mut arr = ArrayTrait::<felt>::new();
    arr.append(1);
    arr.append(2);
    arr.append(3);
    arr.append(4);
    arr.append(5);
    arr.append(6);
    arr.append(7);
    arr.append(8);

    let mut matrix = MatrixTrait::<felt>::new(4_usize, 4_usize, arr);
    let result = MatrixTrait::<felt>::get(matrix, 0_usize, 0_usize);

    assert(result == 1, 'correct result');
}