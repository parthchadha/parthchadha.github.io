require 'nn'
require 'mattorch'

data = mattorch.load('a.mat')
data = data.a:transpose(1,2)

size = data:size(1)

