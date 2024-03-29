use array::ArrayTrait;
use option::OptionTrait;

use neural_network::onnx_cairo::operators::math::int33;
use neural_network::onnx_cairo::operators::math::int33::i33;

use neural_network::onnx_cairo::operators::math::matrix::Matrix;
use neural_network::onnx_cairo::operators::math::matrix::MatrixTrait;

fn load_image(rows: usize, cols: usize) -> Matrix {
    let mut image = ArrayTrait::new();

    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 3_u32, sign: false });
    image.append(i33 { inner: 18_u32, sign: false });
    image.append(i33 { inner: 18_u32, sign: false });
    image.append(i33 { inner: 18_u32, sign: false });
    image.append(i33 { inner: 126_u32, sign: false });
    image.append(i33 { inner: 120_u32, sign: true });
    image.append(i33 { inner: 81_u32, sign: true });
    image.append(i33 { inner: 26_u32, sign: false });
    image.append(i33 { inner: 90_u32, sign: true });
    image.append(i33 { inner: 1_u32, sign: true });
    image.append(i33 { inner: 9_u32, sign: true });
    image.append(i33 { inner: 127_u32, sign: false });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 30_u32, sign: false });
    image.append(i33 { inner: 36_u32, sign: false });
    image.append(i33 { inner: 94_u32, sign: false });
    image.append(i33 { inner: 102_u32, sign: true });
    image.append(i33 { inner: 86_u32, sign: true });
    image.append(i33 { inner: 3_u32, sign: true });
    image.append(i33 { inner: 3_u32, sign: true });
    image.append(i33 { inner: 3_u32, sign: true });
    image.append(i33 { inner: 3_u32, sign: true });
    image.append(i33 { inner: 3_u32, sign: true });
    image.append(i33 { inner: 31_u32, sign: true });
    image.append(i33 { inner: 84_u32, sign: true });
    image.append(i33 { inner: 3_u32, sign: true });
    image.append(i33 { inner: 14_u32, sign: true });
    image.append(i33 { inner: 61_u32, sign: true });
    image.append(i33 { inner: 64_u32, sign: false });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });
    image.append(i33 { inner: 0_u32, sign: true });

    MatrixTrait::new(rows, cols, image)
}
