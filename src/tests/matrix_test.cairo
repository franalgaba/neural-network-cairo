use array::ArrayTrait;
use traits::Into;
use option::OptionTrait;

use neural_network::math::matrix::Matrix;
use neural_network::math::matrix::MatrixTrait;
use neural_network::math::int33;
use neural_network::math::int33::i33;


#[test]
#[available_gas(2000000)]
fn matrix_new_test() {
    let mut arr = ArrayTrait::<i33>::new();
    let val_1 = i33 { inner: 1_u32, sign: false };
    let val_2 = i33 { inner: 2_u32, sign: false };
    let val_3 = i33 { inner: 3_u32, sign: false };
    let val_4 = i33 { inner: 4_u32, sign: false };

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

    let mut arr = ArrayTrait::<i33>::new();
    let val_1 = i33 { inner: 1_u32, sign: false };
    let val_2 = i33 { inner: 2_u32, sign: true };
    let val_3 = i33 { inner: 3_u32, sign: false };
    let val_4 = i33 { inner: 4_u32, sign: true };

    arr.append(val_1);
    arr.append(val_2);
    arr.append(val_3);
    arr.append(val_4);

    let mut matrix = MatrixTrait::new(2_usize, 2_usize, arr);
    let result = matrix.get(0_usize, 0_usize);

    assert(result.inner == 1_u32, 'result[0] == -1');
    assert(result.sign == false, 'result[0] == -1');

    let result = matrix.get(1_usize, 1_usize);

    assert(result.inner == 4_u32, 'result[8] == -4');
    assert(result.sign == true, 'result[8] == -4');
}