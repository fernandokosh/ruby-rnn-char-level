class MatrixSimple
  attr_accessor :rows, :cols, :data

  def initialize(rows, cols, fill = 0.0)
    @rows = rows
    @cols = cols
    @data = Array.new(rows) { Array.new(cols, fill.to_f) }
  end

  def [](i, j)
    @data[i][j]
  end

  def []=(i, j, v)
    @data[i][j] = v.to_f
  end

  def self.random(rows, cols, scale = 0.01)
    m = MatrixSimple.new(rows, cols)
    rows.times do |i|
      cols.times do |j|
        m[i, j] = (rand - 0.5) * 2 * scale
      end
    end
    m
  end

  def +(other)
    raise "Dimension mismatch" unless same_shape?(other)
    r = MatrixSimple.new(@rows, @cols)
    @rows.times do |i|
      @cols.times do |j|
        r[i, j] = self[i, j] + other[i, j]
      end
    end
    r
  end

  def dot(other)
    raise "Dimension mismatch" unless @cols == other.rows
    r = MatrixSimple.new(@rows, other.cols)
    @rows.times do |i|
      other.cols.times do |j|
        sum = 0.0
        @cols.times { |k| sum += self[i, k] * other[k, j] }
        r[i, j] = sum
      end
    end
    r
  end

  def transpose
    r = MatrixSimple.new(@cols, @rows)
    @rows.times { |i| @cols.times { |j| r[j, i] = self[i, j] } }
    r
  end

  def same_shape?(other)
    @rows == other.rows && @cols == other.cols
  end
end
