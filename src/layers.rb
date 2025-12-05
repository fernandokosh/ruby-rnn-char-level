require_relative 'matrix'

class Linear
  attr_accessor :W, :b, :grad_W, :grad_b
  def initialize(in_dim, out_dim)
    @W = MatrixSimple.random(in_dim, out_dim, 0.1)
    @b = MatrixSimple.new(1, out_dim, 0.0)
    @grad_W = MatrixSimple.new(in_dim, out_dim, 0.0)
    @grad_b = MatrixSimple.new(1, out_dim, 0.0)
  end

  def forward(X) # X: MatrixSimple (batch, in_dim)
    @X = X
    out = X.dot(@W) + broadcast_bias(X.rows, @b)
    out
  end

  def backward(dout) # dout: (batch, out_dim)
    # dW = X^T * dout
    @grad_W = @X.transpose.dot(dout)
    # db = sum rows of dout
    @grad_b = MatrixSimple.new(1, dout.cols)
    dout.rows.times { |i| dout.cols.times { |j| @grad_b[0,j] += dout[i,j] } }
    # dX = dout * W^T
    dX = dout.dot(@W.transpose)
    dX
  end

  def step!(lr)
    @W.rows.times { |i| @W.cols.times { |j| @W[i,j] -= lr * @grad_W[i,j] } }
    @b.cols.times { |j| @b[0,j] -= lr * @grad_b[0,j] }
  end

  private
  def broadcast_bias(batch, b)
    m = MatrixSimple.new(batch, b.cols)
    batch.times { |i| b.cols.times { |j| m[i,j] = b[0,j] } }
    m
  end
end

# Tanh activation (elementwise)
def tanh_forward(X)
  r = MatrixSimple.new(X.rows, X.cols)
  X.rows.times { |i| X.cols.times { |j| r[i,j] = Math.tanh(X[i,j]) } }
  r
end

def tanh_backward(X, dout) # X is forward output (tanh)
  dx = MatrixSimple.new(X.rows, X.cols)
  X.rows.times { |i| X.cols.times { |j| dx[i,j] = dout[i,j] * (1 - X[i,j]**2) } }
  dx
end
