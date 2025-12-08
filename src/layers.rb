require_relative "matrix"

class Linear
  attr_accessor :W, :b, :grad_W, :grad_b, :X

  def initialize(in_dim, out_dim)
    @W = MatrixSimple.random(in_dim, out_dim, 0.1)
    @b = MatrixSimple.new(1, out_dim, 0.0)
    @grad_W = MatrixSimple.new(in_dim, out_dim, 0.0)
    @grad_b = MatrixSimple.new(1, out_dim, 0.0)
  end

  def forward(x)
    @x = x
    out = x.dot(@W)
    out = add_bias(out, @b)
    out
  end

  def backward(dout)
    @grad_W = @X.transpose.dot(dout)
    @grad_b = sum_rows(dout)
    dX = dout.dot(@W.transpose)
    dX
  end

  def step!(lr)
    @W.rows.times do |i|
      @W.cols.times do |j|
        @W[i, j] -= lr * @grad_W[i, j]
      end
    end
    @b.cols.times { |j| @b[0, j] -= lr * @grad_b[0, j] }
  end

  private

  def add_bias(x, b)
    out = MatrixSimple.new(x.rows, x.cols)
    x.rows.times do |i|
      x.cols.times do |j|
        out[i, j] = x[i, j] + b[0, j]
      end
    end
    out
  end

  def sum_rows(d)
    out = MatrixSimple.new(1, d.cols)
    d.rows.times do |i|
      d.cols.times { |j| out[0, j] += d[i, j] }
    end
    out
  end
end

def tanh_forward(x)
  out = MatrixSimple.new(x.rows, x.cols)
  x.rows.times do |i|
    x.cols.times { |j| out[i, j] = Math.tanh(x[i, j]) }
  end
  out
end

def tanh_backward(out, dout)
  dx = MatrixSimple.new(out.rows, out.cols)
  out.rows.times do |i|
    out.cols.times do |j|
      dx[i, j] = dout[i, j] * (1.0 - out[i, j]**2)
    end
  end
  dx
end
