require_relative 'matrix'
require_relative 'layers'
require_relative 'loss'

# tiny helper: one-hot vector as MatrixSimple row
def one_hot(index, size)
  m = MatrixSimple.new(1, size)
  m[0,index] = 1.0
  m
end

vocab_size = chars.length # define the vocabulary here.
embed_dim = 16
hidden = 64
lr = 0.01

# Embedding as simple Linear (vocab_size -> embed_dim)
embed = Linear.new(vocab_size, embed_dim)
# RNN weights as Linears used in loop
Wxh = Linear.new(embed_dim, hidden) # we will treat like layer but only matrix usage
Whh = Linear.new(hidden, hidden)
Why = Linear.new(hidden, vocab_size)

# training loop (BPTT naive)
epochs.times do |ep|
  total_loss = 0.0
  seqs.each do |seq| # seq = array of indices
    # forward pass
    hs = []
    h_prev = MatrixSimple.new(1, hidden, 0.0)
    logits = []
    seq[0..-2].each_with_index do |ch, t|
      x_one = one_hot(ch, vocab_size)
      x_embed = embed.forward(x_one) # (1, embed_dim)
      h_linear = x_embed.dot(Wxh.W) + h_prev.dot(Whh.W) + Wxh.broadcast_bias(1, Wxh.b)
      h = tanh_forward(h_linear)
      y = h.dot(Why.W) + Why.broadcast_bias(1, Why.b)
      hs << h
      logits << y
      h_prev = h
    end

    # compute loss across sequence targets (shifted)
    probs_seq = logits.map { |z| softmax_rows(z) } # each is 1 x vocab
    targets = seq[1..-1] # next chars
    # aggregate loss and grad through time — naive
    seq_loss = 0.0
    # accumulate gradients placeholders (zero)

    # Backprop: compute grads at each time step, then backprop through h
    # (Implementing full BPTT is verbose — start with truncated bptt or single-step)
    # ...
  end
end
