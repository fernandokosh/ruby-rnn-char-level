require 'json'

class MatrixSimple
  attr_accessor :rows, :cols, :data
  def initialize(rows, cols, fill=0.0)
    @rows = rows; @cols = cols
    @data = Array.new(rows) { Array.new(cols, fill.to_f) }
  end
  def [](i,j); @data[i][j]; end
  def []=(i,j,val); @data[i][j]=val.to_f; end
  def self.random(rows, cols, scale=0.1)
    m = MatrixSimple.new(rows, cols)
    rows.times { |i| cols.times { |j| m[i,j] = (rand - 0.5) * 2 * scale } }
    m
  end
  def dot(other)
    raise unless @cols == other.rows
    r = MatrixSimple.new(@rows, other.cols)
    @rows.times do |i|
      other.cols.times do |j|
        s = 0.0
        @cols.times { |k| s += self[i,k] * other[k,j] }
        r[i,j] = s
      end
    end
    r
  end
  def transpose
    r = MatrixSimple.new(@cols, @rows)
    @rows.times { |i| @cols.times { |j| r[j,i] = self[i,j] } }
    r
  end
end

def softmax_row(arr)
  mx = arr.max
  exps = arr.map { |v| Math.exp(v - mx) }
  s = exps.sum
  exps.map { |e| e / s }
end

puts "Prepare data"

# Prepare data
text = File.read('corpus.txt') rescue "hello world\nthis is a tiny corpus for demo\n"
chars = text.each_char.to_a.uniq.sort
char_to_i = Hash[chars.map.with_index { |c,i| [c,i] }]
i_to_char = char_to_i.invert

vocab = chars.length
context = 8 # window size
embed_dim = 16
hidden = 128
lr = 0.1
epochs = 2000

# Initialize parameters: embeddings, linear layers
# Embedding matrix: vocab x embed_dim
E = MatrixSimple.random(vocab, embed_dim, 0.1)
# MLP: input dim = context * embed_dim
W1 = MatrixSimple.random(context*embed_dim, hidden, 0.1)
b1 = Array.new(hidden, 0.0)
W2 = MatrixSimple.random(hidden, vocab, 0.1)
b2 = Array.new(vocab, 0.0)

# helpers
def mat_row_dot_vector(mat, vec) # mat: 1 x D, vec: D
  res = Array.new(mat.cols, 0.0)
  mat.cols.times { |j|
    s = 0.0
    mat.rows.times { |i| s += mat[i,j] * vec[i] }
    res[j] = s
  }
  res
end

puts "Training: stochastic over windows"

# Training: stochastic over windows
seq = text.each_char.map { |c| char_to_i[c] }
pairs = []
(0...(seq.length - context)).each do |i|
  print "."
  input = seq[i, context]
  target = seq[i+context]
  pairs << [input, target]
end

puts "done"

pairs.shuffle!

puts "Training: running epochs"

epochs.times do |ep|
  print "."
  loss = 0.0
  pairs.each do |input, target|
    # forward
    # build embedding vector (1 x context*embed_dim)
    x = []
    input.each do |idx|
      e_row = E.data[idx] # array of embed_dim
      x.concat(e_row)
    end
    # hidden = tanh(x W1 + b1)
    h = Array.new(hidden, 0.0)
    (0...hidden).each do |j|
      s = 0.0
      (0...x.length).each { |k| s += x[k] * W1[k,j] }
      s += b1[j]
      h[j] = Math.tanh(s)
    end
    # logits = h W2 + b2
    logits = Array.new(vocab, 0.0)
    (0...vocab).each do |j|
      s = 0.0
      (0...hidden).each { |k| s += h[k] * W2[k,j] }
      s += b2[j]
      logits[j] = s
    end
    probs = softmax_row(logits)
    loss += -Math.log([probs[target], 1e-15].max)

    # backward (compute grads, vanilla SGD)
    # grad_logits = probs; grad_logits[target] -= 1
    grad_logits = probs
    grad_logits[target] -= 1.0

    # grads for W2, b2, grad h
    grad_W2 = Array.new(hidden) { Array.new(vocab, 0.0) }
    grad_b2 = Array.new(vocab, 0.0)
    grad_h = Array.new(hidden, 0.0)
    (0...vocab).each do |j|
      grad_b2[j] = grad_logits[j]
      (0...hidden).each { |k| grad_W2[k][j] += h[k] * grad_logits[j]; grad_h[k] += W2[k,j] * grad_logits[j] }
    end

    # back through tanh
    (0...hidden).each { |k| grad_h[k] *= (1 - h[k]**2) }

    # grads for W1, b1, and x
    grad_W1 = Array.new(x.length) { Array.new(hidden, 0.0) }
    grad_b1 = Array.new(hidden, 0.0)
    grad_x = Array.new(x.length, 0.0)
    (0...hidden).each do |j|
      grad_b1[j] += grad_h[j]
      (0...x.length).each do |i_x|
        grad_W1[i_x][j] += x[i_x] * grad_h[j]
        grad_x[i_x] += W1[i_x,j] * grad_h[j]
      end
    end

    # grads to embeddings E
    # x is concatenation of embeddings for each char in context
    (0...context).each do |ci|
      emb_idx = input[ci]
      start = ci * embed_dim
      (0...embed_dim).each do |k|
        E[emb_idx, k] -= lr * grad_x[start + k]
      end
    end

    # update W2, b2
    (0...hidden).each { |k| (0...vocab).each { |j| W2[k,j] -= lr * grad_W2[k][j] } }
    (0...vocab).each { |j| b2[j] -= lr * grad_b2[j] }
    # update W1, b1
    (0...x.length).each { |i_x| (0...hidden).each { |j| W1[i_x,j] -= lr * grad_W1[i_x][j] } }
    (0...hidden).each { |j| b1[j] -= lr * grad_b1[j] }
  end

  if ep % 100 == 0
    puts "done"
    puts "ep #{ep} loss #{loss / pairs.length}"
    # sample text
    seed = pairs.sample[0]
    out = seed.map { |i| i_to_char[i] }.join
    cur = seed.dup
    100.times do
      # forward same as above to get probs
      x = []
      cur.each { |idx| x.concat(E.data[idx]) }
      h = Array.new(hidden, 0.0)
      (0...hidden).each do |j|
        s = 0.0
        (0...x.length).each { |k| s += x[k] * W1[k,j] }
        s += b1[j]
        h[j] = Math.tanh(s)
      end
      logits = Array.new(vocab, 0.0)
      (0...vocab).each do |j|
        s = 0.0
        (0...hidden).each { |k| s += h[k] * W2[k,j] }
        s += b2[j]
        logits[j] = s
      end
      probs = softmax_row(logits)
      # greedy
      idx = probs.each_with_index.max[1]
      out += i_to_char[idx]
      cur.shift; cur << idx
    end
    puts "sample: #{out}"
  end
end
puts "done"

puts "Training: done"
