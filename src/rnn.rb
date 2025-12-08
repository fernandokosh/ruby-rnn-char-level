require_relative "matrix"
require_relative "layers"
require_relative "loss"

class RNNCell
  attr_accessor :Wxh, :Whh, :bh, :Why, :by

  def initialize(input_dim, hidden_dim, output_dim)
    @Wxh = Linear.new(input_dim, hidden_dim)
    @Whh = Linear.new(hidden_dim, hidden_dim)
    @Why = Linear.new(hidden_dim, output_dim)

    @bh = MatrixSimple.new(1, hidden_dim, 0.0)
    @by = MatrixSimple.new(1, output_dim, 0.0)
  end

  # forward pass for one step
  def step(x, prev_h)
    h_linear = x.dot(@Wxh.W) + prev_h.dot(@Whh.W)
    h_linear = add_bias(h_linear, @bh)

    h = tanh_forward(h_linear)

    y = h.dot(@Why.W)
    y = add_bias(y, @by)

    [h, y]
  end

  private

  def add_bias(x, b)
    out = MatrixSimple.new(x.rows, x.cols)
    x.rows.times do |i|
      x.cols.times { |j| out[i, j] = x[i, j] + b[0, j] }
    end
    out
  end
end
