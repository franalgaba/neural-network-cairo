use array::ArrayTrait;
use traits::Into;
use option::OptionTrait;

use neural_network::matrix::Matrix;
use neural_network::matrix::MatrixTrait;

#[test]
#[available_gas(2000000)]
fn matrix_new_test() {
    let mut arr = ArrayTrait::<u256>::new();
    let val_1: u256 = 1.into();
    let val_2: u256 = 2.into();
    let val_3: u256 = 3.into();
    let val_4: u256 = 4.into();

    arr.append(val_1);
    arr.append(val_2);
    arr.append(val_3);
    arr.append(val_4);

    let mut matrix = MatrixTrait::new(2_usize, 2_usize, arr);
    let result_len = matrix.len();

    assert(result_len == 4_usize, 'correct length');
}

#[test]
#[available_gas(2000000)]
fn matrix_get_test() {

    let mut arr = ArrayTrait::<u256>::new();
    let val_1: u256 = 1.into();
    let val_2: u256 = 2.into();
    let val_3: u256 = 3.into();
    let val_4: u256 = 4.into();

    arr.append(val_1);
    arr.append(val_2);
    arr.append(val_3);
    arr.append(val_4);

    let mut matrix = MatrixTrait::new(2_usize, 2_usize, arr);

    match matrix.get(0_usize, 0_usize) {
        Option::Some(result) => {
            assert(result == val_1, 'wrong result');
        },
        Option::None(_) => {
            assert(0 == 1, 'should return value');
        },
    };
}