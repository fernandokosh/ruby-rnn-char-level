# Ruby From-Scratch Language Model (RNN Prototype)

This project is a minimal, fully transparent, from-scratch implementation of a toy language model written in pure Ruby.
The goal is **educational clarity over performance**: every operation (matrix multiplication, activation functions, forward pass, backward pass, training loop) is implemented manually, without external numeric libraries.

The project demonstrates the conceptual building blocks behind modern language models (LLMs) using a small **character-level RNN**.
It is intentionally slow and intentionally simple — a learning platform, not a production engine.

---

## Vision

The project aims to demystify how models like GPT ultimately work under the hood:

- Manual matrix operations
- Backpropagation through time (BPTT)
- Softmax and cross-entropy implementation
- Embeddings without external libraries
- A simple RNN cell built from primitive components
- End-to-end training on plain text

By exposing each step explicitly, the repository helps developers understand the full lifecycle of a language model.

---

## Project Structure

```

.
├── README.md
├── Gemfile
├── Rakefile
├── src
│   ├── matrix.rb
│   ├── layers.rb
│   ├── loss.rb
│   ├── rnn.rb
│   └── trainer.rb
└── spec
    ├── matrix_spec.rb
    ├── layers_spec.rb
    ├── loss_spec.rb
    └── rnn_spec.rb

```

---

## Core Components

### `MatrixSimple`
A tiny pure-Ruby matrix implementation supporting:
- Addition
- Multiplication
- Transposition
- Random initialization

This class is intentionally naive to highlight how numeric engines work internally.

### `Linear`
Single fully-connected layer with:
- Forward pass
- Backward pass
- Trainable weights and biases

### `Softmax + CrossEntropy`
Complete, stable softmax function + loss and gradient implementation.

### `RNNCell`
A single tanh-based recurrent unit:
```

h_t = tanh(x_t * Wxh + h_{t-1} * Whh + b)

````

### `Trainer`
Runs full BPTT through the sequence, updates all parameters via SGD.

---

## Requirements

- Ruby 3.0+
- Bundler (for tests only; the model itself has zero external dependencies)

Install test deps:

```bash
bundle install
````

Run test suite:

```bash
rake
```

---

## Usage Example

Train on a small text file:

irb


```ruby

require_relative "src/trainer"

trainer = Trainer.new("data/input.txt")
trainer.train(epochs: 50, learning_rate: 0.01)
```

Generate text:

```ruby
output = trainer.generate("t", length: 200)
puts output
```

---

## Roadmap

* Add multi-layer RNN
* Replace tanh cell with GRU
* Implement a miniature attention layer
* Build a 1-layer micro-Transformer
* Add visualization of gradients
* Add export/import of weights

---

## Disclaimer

This project is intentionally slow and completely unsuitable for real-world NLP tasks.
Its purpose is **pedagogical transparency**: everything is built from first principles.

Enjoy exploring the machinery behind neural language models!

