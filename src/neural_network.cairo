use array::ArrayTrait;
use option::OptionTrait;

use neural_network::math::matrix::Matrix;
use neural_network::math::matrix::MatrixTrait;

#[derive(Drop)]
struct NeuralNetwork {
    W1: Matrix,
    b1: Matrix,
    W2: Matrix,
    b2: Matrix,
}

trait NNTrait {
    fn new(W1: Matrix, b1: Matrix, W2: Matrix, b2: Matrix) -> NeuralNetwork;
// fn forward_prop(self: @Matrix, data: Matrix) -> Option::<Matrix> ;
// fn ReLU(self: @Matrix, z: Matrix) -> Option::<Matrix>;
// fn softmax(self: @Matrix, z: Matrix) -> Option::<Matrix>;
}

impl NNImpl of NNTrait {
    #[inline(always)]
    fn new(W1: Matrix, b1: Matrix, W2: Matrix, b2: Matrix) -> NeuralNetwork {
        nn_new(W1, b1, W2, b2)
    }
// fn forward_prop(self: @Matrix, data: Matrix) -> Option::<Matrix> {
// Z1 = W1.dot(X) + b1
// A1 = ReLU(Z1)
// Z2 = W2.dot(A1) + b2
// A2 = softmax(Z2)
// }

// def ReLU(Z):
//     return np.maximum(Z, 0)

// def softmax(Z):
//     A = np.exp(Z) / sum(np.exp(Z))
//     return A

}

fn nn_new(W1: Matrix, b1: Matrix, W2: Matrix, b2: Matrix) -> NeuralNetwork {
    NeuralNetwork { W1: W1, b1: b1, W2: W2, b2: b2,  }
}
