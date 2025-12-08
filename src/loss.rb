require_relative "matrix"

def softmax_rows(x)
  out = MatrixSimple.new(x.rows, x.cols)
  x.rows.times do |i|
    row = x.data[i]
    max_v = row.max
    exp = row.map { |v| Math.exp(v - max_v) }
    sum = exp.sum
    exp.each_with_index { |v, j| out[i, j] = v / sum }
  end
  out
end

def cross_entropy_loss_and_grad(probs, targets)
  batch = probs.rows
  classes = probs.cols
  loss = 0.0
  grad = MatrixSimple.new(batch, classes)

  batch.times do |i|
    t = targets[i]
    p = probs[i, t]
    p = 1e-15 if p <= 0
    loss -= Math.log(p)

    classes.times { |j| grad[i, j] = probs[i, j] }
    grad[i, t] -= 1.0
  end

  [loss / batch.to_f, grad]
end
