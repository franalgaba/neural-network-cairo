use array::ArrayTrait;
use option::OptionTrait;

use neural_network::math::matrix::Matrix;
use neural_network::math::matrix::MatrixTrait;
use neural_network::activations::relu::relu;

#[derive(Drop)]
struct NeuralNetwork {
    W1: Matrix,
    b1: Matrix,
    W2: Matrix,
    b2: Matrix,
}

trait NNTrait {
    fn new(W1: Matrix, b1: Matrix, W2: Matrix, b2: Matrix) -> NeuralNetwork;
    fn forward_prop(self: @NeuralNetwork, X: @Matrix) -> Matrix;
    fn predict(self: @NeuralNetwork, X: @Matrix) -> Matrix;
}

impl NNImpl of NNTrait {
    #[inline(always)]
    fn new(W1: Matrix, b1: Matrix, W2: Matrix, b2: Matrix) -> NeuralNetwork {
        nn_new(W1, b1, W2, b2)
    }

    fn forward_prop(self: @NeuralNetwork, X: @Matrix) -> Matrix {
        let mut Z1_temp = self.W1.dot(X);
        let mut Z1 = Z1_temp.add(self.b1);
        let mut A1 = relu(@Z1);
        let mut Z2_temp = self.W2.dot(@A1);
        let mut Z2 = Z2_temp.add(self.b2);
        // A2 = softmax(Z2)
        Z2
        // A2
    }

    fn predict(self: @NeuralNetwork, X: @Matrix) -> Matrix {
        let mut A2 = forward_prop(self, X);
        // calculate the argmax of the final layer
        // A2 = argmax(A2)
        A2
    }

}

fn nn_new(W1: Matrix, b1: Matrix, W2: Matrix, b2: Matrix) -> NeuralNetwork {
    NeuralNetwork { W1: W1, b1: b1, W2: W2, b2: b2,  }
}
