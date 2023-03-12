use array::ArrayTrait;
use traits::Into;
use option::OptionTrait;

use neural_network::math::matrix::Matrix;
use neural_network::math::matrix::MatrixTrait;

use neural_network::math::int33;
use neural_network::math::int33::i33;


fn relu(z: @Matrix) -> Matrix {
        let mut arr = ArrayTrait::<i33>::new();
        
        relu_inner(ref arr, z.data, 0_usize, z.data.len());
        MatrixTrait::new(*z.rows, *z.cols, arr)
    }

fn relu_inner(ref arr: Array::<i33>, input: @Array::<i33>, index: usize, len: usize){

    // TODO: Remove when automatically handled by compiler.
    match gas::get_gas() {
        Option::Some(_) => {},
        Option::None(_) => {
            let mut data = array_new::<felt>();
            array_append::<felt>(ref data, 'OOG');
            panic(data);
        },
    }
    
    if index == len {
        return ();
    }

    let val_0 = (i33 { inner: 0_u32, sign: true });

    // if x > 0 -> x
    if *input.at(index) > val_0 {
        arr.append(*input.at(index));
    }
    // if x < 0 -> 0
    else {
        arr.append(val_0);
    }
    relu_inner(ref arr, input, index + 1_usize, len);
}
