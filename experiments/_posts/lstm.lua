require 'nn'
require 'nngraph'


local inputs = {}
-- nn.Identity used as a placeholder for input
table.insert(inputs,nn.Identity()())  -- input
table.insert(inputs,nn.Identity()())  -- c at t-1
table.insert(inputs,nn.Identity()())  -- h at t-1

local input = inputs[1]
local prev_c = inputs[2]
local prev_h = inputs[3]

local i2h =  nn.Linear(input_size, 4*rnn_size)(input) 
-- Input to hidden layer. 4 parts would be used for 4 different gates
-- Linear transformation to the incoming data, input. y=Ax+b.

local h2h = nn.Linear(rnn_size,4*rnn_size)(prev_h)
-- hidden to hidden layer. 4 parts would be used for 4 different gates

local preactivations = nn.CAddTable()({i2h,h2h}) 
-- i2h + h2h
-- Portions of this preactivations are for different gates

-- narrow(dim,index,size)
-- narrow  returns a Tensor with the dimension dim is narrowed from index to index+size-1
local pre_sigmoid_chunk = nn.Narrow(2,1,3*rnn_size)(preactivations)
-- we just chose 3 parts of preactivations on which we will apply sigmoid
local in_chunk = nn.Narrow(2,3*rnn_size+1,rnn_size)(preactivations)
-- we just chose 1 part of preactivation on which we will apply tanh (input preactivations)

local all_gates = nn.Sigmoid()(pre_sigmoid_chunk)
local in_transform = nn.Tanh()(in_chunk)

-- seperating 3 gates on which we applied sigmoid
local in_gate = nn.Narrow(2,1,rnn_size)(all_gates)
local forget_gate = nn.Narrow(2,rnn_size+1,rnn_size)(all_gates)
local out_gate = nn.Narrow(2,2*rnn_size+1,rnn_size)(all_gates)


-- next_c equation implementation
local c_forget = nn.CMulTable()({forget_gate,prev_c})
local c_input = nn.CMulTable()({in_gate,in_transform})

local next_c = nn.CAddTable()({c_forget,c_input})

-- next_h equation implementation 
local c_transform = nn.Tanh()(next_c)
local next_h = nn.CMulTable()({out_gate,c_transform})


outputs = {}
table.insert(outputs,next_c)
table.insert(outputs,next_h)

return nn.gModule(inputs,outputs)










