<p align="center">
  <a href="https://github.com/auditless/cairo-template/actions/workflows/test.yaml">
    <img src="https://github.com/auditless/cairo-template/actions/workflows/test.yaml/badge.svg?event=push" alt="CI Badge"/>
  </a>
</p>

# Neural Network for MNIST in Cairo 1.0

Implementation of a Neural Network from scratch using Cairo 1.0 for MNIST predictions.

The NN has a simple two-layer architecture: 
  - Input layer 𝑎[0] will have 784 units corresponding to the 784 pixels in each 28x28 input image. 
  - A hidden layer 𝑎[1] will have 10 units with ReLU activation.
  - Output layer 𝑎[2] will have 10 units corresponding to the ten digit classes with softmax activation.

Functionalities implemented in Cairo 1:
 - Matrix implementation.
 - 8-bit weight quantization.
 - ReLU layer imaplementation.
 - Softmax activation.

[Built with **`auditless/cairo-template`**](https://github.com/auditless/cairo-template)

## Working with the project

Currently supports building and testing contracts.

### Build

Build the contracts.

```bash
$ make build
```

### Test

Run the tests in `src/test`:

```bash
$ make test
```

### Format

Format the Cairo source code (using Scarb):

```bash
$ make fmt
```

### Credits

- [cairo-ml](https://github.com/raphaelDkhn/cairo_ml) by [Raphael Doukhan](https://twitter.com/raphael_dkhn).
