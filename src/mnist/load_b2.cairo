use array::ArrayTrait;
use option::OptionTrait;

use neural_network::onnx_cairo::operators::math::int33;
use neural_network::onnx_cairo::operators::math::int33::i33;

use neural_network::onnx_cairo::operators::math::matrix::Matrix;
use neural_network::onnx_cairo::operators::math::matrix::MatrixTrait;

fn load_b2(rows: usize, cols: usize) -> Matrix {
    let mut tensor = ArrayTrait::new();

    tensor.append(i33 { inner: 190_u32, sign: true });
    tensor.append(i33 { inner: 58_u32, sign: true });
    tensor.append(i33 { inner: 37_u32, sign: false });
    tensor.append(i33 { inner: 59_u32, sign: true });
    tensor.append(i33 { inner: 31_u32, sign: false });
    tensor.append(i33 { inner: 111_u32, sign: true });
    tensor.append(i33 { inner: 36_u32, sign: false });
    tensor.append(i33 { inner: 309_u32, sign: false });
    tensor.append(i33 { inner: 156_u32, sign: true });
    tensor.append(i33 { inner: 67_u32, sign: false });

    MatrixTrait::new(rows, cols, tensor)
}
