use array::ArrayTrait;
use option::OptionTrait;

use neural_network::onnx_cairo::operators::math::int33;
use neural_network::onnx_cairo::operators::math::int33::i33;

use neural_network::onnx_cairo::operators::math::matrix::Matrix;
use neural_network::onnx_cairo::operators::math::matrix::MatrixTrait;

fn load_b1(rows: usize, cols: usize) -> Matrix {
    let mut tensor = ArrayTrait::new();

    tensor.append(i33 { inner: 20_u32, sign: true });
    tensor.append(i33 { inner: 2_u32, sign: true });
    tensor.append(i33 { inner: 486_u32, sign: false });
    tensor.append(i33 { inner: 94_u32, sign: false });
    tensor.append(i33 { inner: 291_u32, sign: false });
    tensor.append(i33 { inner: 1_u32, sign: true });
    tensor.append(i33 { inner: 2_u32, sign: true });
    tensor.append(i33 { inner: 1_u32, sign: true });
    tensor.append(i33 { inner: 1_u32, sign: true });
    tensor.append(i33 { inner: 2_u32, sign: true });

    MatrixTrait::new(rows, cols, tensor)
}
