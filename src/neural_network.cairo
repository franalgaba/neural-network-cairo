use array::ArrayTrait;
use option::OptionTrait;

use neural_network::math::int33;
use neural_network::math::int33::i33;
use neural_network::math::matrix::Matrix;
use neural_network::math::matrix::MatrixTrait;
use neural_network::activations::relu::relu;
use neural_network::activations::softmax::softmax;


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
    fn predict(self: @NeuralNetwork, X: @Matrix) -> Array::<usize>;
}

impl NNImpl of NNTrait {

    fn new(W1: Matrix, b1: Matrix, W2: Matrix, b2: Matrix) -> NeuralNetwork {
        nn_new(W1, b1, W2, b2)
    }

    fn forward_prop(self: @NeuralNetwork, X: @Matrix) -> Matrix {
        let mut Z1_temp = self.W1.dot(X);
        let mut Z1 = Z1_temp.add(self.b1);
        let mut A1 = relu(@Z1);
        let mut Z2_temp = self.W2.dot(@A1);
        let mut Z2 = Z2_temp.add(self.b2);
        let mut A2 = softmax(@Z2);
        A2
    }

    fn predict(self: @NeuralNetwork, X: @Matrix) -> Array::<usize> {
        let mut A2 = self.forward_prop(X);
        // calculate the argmax of the final layer
        let mut predictions = A2.argmax();
        predictions
    }

}

fn nn_new(W1: Matrix, b1: Matrix, W2: Matrix, b2: Matrix) -> NeuralNetwork {
    NeuralNetwork { W1: W1, b1: b1, W2: W2, b2: b2,  }
}
