class MatrixSimple
  attr_accessor :rows, :cols, :data
  def initialize(rows, cols, fill=0.0)
    @rows = rows; @cols = cols
    @data = Array.new(rows) { Array.new(cols, fill.to_f) }
  end

  def [](i,j); @data[i][j]; end
  def []=(i,j,val); @data[i][j]=val.to_f; end

  def self.random(rows, cols, scale=0.01)
    m = MatrixSimple.new(rows, cols)
    rows.times { |i| cols.times { |j| m[i,j] = (rand - 0.5) * 2 * scale } }
    m
  end

  def +(other)
    raise unless @rows==other.rows && @cols==other.cols
    r = MatrixSimple.new(@rows, @cols)
    @rows.times { |i| @cols.times { |j| r[i,j] = self[i,j] + other[i,j] } }
    r
  end

  def dot(other) # this * other
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
