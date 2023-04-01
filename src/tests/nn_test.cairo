use array::ArrayTrait;
use option::OptionTrait;

use neural_network::onnx_cairo::operators::math::int33;
use neural_network::onnx_cairo::operators::math::int33::i33;

use neural_network::nn::neural_network::NNTrait;
use neural_network::mnist::load_w1::load_W1;
use neural_network::mnist::load_w2::load_W2;
use neural_network::mnist::load_b1::load_b1;
use neural_network::mnist::load_b2::load_b2;
use neural_network::mnist::load_image::load_image;

#[test]
#[available_gas(99999999999999999)]
fn main() -> Array::<usize> {
    let image = load_image(196_usize, 1_usize);

    let W1 = load_W1(10_usize, 196_usize);
    let b1 = load_b1(1_usize, 10_usize);
    let W2 = load_W2(10_usize, 10_usize);
    let b2 = load_b2(1_usize, 10_usize);

    let mut nn = NNTrait::new(W1, W2, b1, b2);

    nn.predict(@image)
}
