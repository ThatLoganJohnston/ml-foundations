# ML Foundations from Scratch

Small, tested implementations of foundational machine-learning algorithms using NumPy. The project is independent portfolio work inspired by concepts I am studying; it does not contain course prompts, datasets, notebooks, or submitted solutions.

## Implemented models

- k-nearest-neighbors classification with Euclidean or Manhattan distance
- ordinary least-squares linear regression with a stable pseudoinverse solution
- binary logistic regression trained with gradient descent
- multiclass perceptron classification

Each estimator follows a compact `fit` / `predict` interface and validates its inputs. The tests cover expected behavior, error handling, deterministic training, and comparison against simple known datasets.

## Quick start

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
pytest
```

## Why this repository exists

Libraries make models easy to call; implementing the core algorithms makes their assumptions and failure modes visible. This project focuses on readable numerical code, reproducible tests, and explanations that connect the math to the implementation.

## Roadmap

- learning-curve and decision-boundary visualizations
- regularization for the linear and logistic models
- a small neural network with backpropagation
- benchmark reports against scikit-learn on public datasets

