<p align="center">
  <a href="https://github.com/auditless/cairo-template/actions/workflows/test.yaml">
    <img src="https://github.com/auditless/cairo-template/actions/workflows/test.yaml/badge.svg?event=push" alt="CI Badge"/>
  </a>
</p>

# Neural Network for NIST in Cairo 1.0

Implementation of a Neural Network from scratch using Cairo 1.0 for MNIST predictions.

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