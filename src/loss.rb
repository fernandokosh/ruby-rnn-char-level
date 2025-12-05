require_relative 'matrix'

def softmax_rows(X)
  out = MatrixSimple.new(X.rows, X.cols)
  X.rows.times do |i|
    max = X.data[i].max
    exps = X.data[i].map { |v| Math.exp(v - max) }
    s = exps.sum
    out.data[i].each_index { |j| out[i,j] = exps[j] / s }
  end
  out
end

# Y: probs (batch, C), targets: array of ints (batch)
def cross_entropy_loss_and_grad(Y, targets)
  batch = Y.rows; C = Y.cols
  loss = 0.0
  grad = MatrixSimple.new(batch, C)
  batch.times do |i|
    t = targets[i]
    p = Y[i,t]
    loss -= Math.log([p,1e-15].max)
    C.times { |j| grad[i,j] = Y[i,j] }
    grad[i,t] -= 1.0
  end
  [loss / batch.to_f, grad] # grad is dL/dz where z are logits after softmax
end
