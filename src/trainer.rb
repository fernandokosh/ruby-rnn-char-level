require "json"

require_relative "rnn"
require_relative "loss"
require_relative "layers"
require_relative "matrix"

class Trainer
  attr_reader :vocab, :vocab_map

  def initialize(path)
    text = File.read(path)
    @vocab = text.chars.uniq.sort
    @vocab_map = {}
    @vocab.each_with_index { |c, i| @vocab_map[c] = i }

    @indices = text.chars.map { |c| @vocab_map[c] }

    @model = RNNCell.new(@vocab.size, 64, @vocab.size)
  end

  def one_hot(idx)
    m = MatrixSimple.new(1, @vocab.size, 0.0)
    m[0, idx] = 1.0
    m
  end

  def train(epochs:, learning_rate:)
    puts "Training for #{epochs} epochs..."

    epochs.times do |ep|
      loss_sum = 0.0
      # tiny sequence loop — not optimized
      (0...@indices.length - 2).each do |i|
        prev_h = MatrixSimple.new(1, 64, 0.0)

        x = one_hot(@indices[i])
        target = @indices[i + 1]

        h, logits = @model.step(x, prev_h)

        probs = softmax_rows(logits)

        loss, grad_out = cross_entropy_loss_and_grad(probs, [target])
        loss_sum += loss

        # Backprop only through output layer (toy mode)
        d_h = grad_out.dot(@model.Why.W.transpose)
        # ignoring full BPTT for minimal version

        # update Why / by
        @model.Why.grad_W = h.transpose.dot(grad_out)
        @model.Why.grad_b = sum_rows(grad_out)
        @model.Why.step!(learning_rate)

      end

      puts "Epoch #{ep+1} | loss=#{loss_sum.round(4)}"
    end
  end

  def generate(seed, length:)
    prev_h = MatrixSimple.new(1, 64, 0.0)
    output = seed.dup
    x = one_hot(@vocab_map[seed])

    length.times do
      h, logits = @model.step(x, prev_h)
      probs = softmax_rows(logits)

      idx = sample(probs)
      char = @vocab[idx]
      output << char

      x = one_hot(idx)
      prev_h = h
    end

    output
  end

  private

  def sample(probs)
    r = rand
    cum = 0.0
    probs.cols.times do |i|
      cum += probs[0, i]
      return i if r <= cum
    end
    probs.cols - 1
  end

  def sum_rows(d)
    out = MatrixSimple.new(1, d.cols)
    d.rows.times do |i|
      d.cols.times { |j| out[0, j] += d[i, j] }
    end
    out
  end

  def save_checkpoint(path)
    params = {
      Wxh: @Wxh.W.data,
      Whh: @Whh.W.data,
      Why: @Why.W.data,
      bh:  @bh.data,
      by:  @by.data
    }

    File.write(path, JSON.pretty_generate(params))
  end

  def load_checkpoint(path)
    params = JSON.parse(File.read(path))

    @Wxh.W.data = params["Wxh"]
    @Whh.W.data = params["Whh"]
    @Why.W.data = params["Why"]
    @bh.data    = params["bh"]
    @by.data    = params["by"]
  end
end
