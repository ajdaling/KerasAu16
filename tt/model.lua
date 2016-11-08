----------------------------------------------------------------------
-- Create CNN and loss to optimize.
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'   -- torch
require 'image'   -- to visualize the dataset
require 'nn'      -- provides all sorts of trainable modules/layers
--require 'Dropout' -- Hinton dropout technique

if opt.type == 'cuda' then
   nn.SpatialConvolutionMM = nn.SpatialConvolution
end

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> define parameters')

-- 2-class problem: faces!
local noutputs = 2

-- input dimensions: faces!
local nfeats = numChannels   -- this is the number of channels
local width = 32
local height = 32

-- hidden units, filter sizes (for ConvNet only):
local nstates = {15,30}
local filtsize = {5, 13}
local poolsize = 2

local convDepth_L1 = 10  -- this is the number of neurons in a conv layer that connects to the same region in the input image volume (depth x width x height)
local receptiveFieldWidth_L1 = 5
local receptiveFieldHeight_L1 = 5

local convDepth_L2 = 5  -- this is the number of neurons in a conv layer that connects to the same region in the input image volume (depth x width x height)
local receptiveFieldWidth_L2 = 5  
local receptiveFieldHeight_L2 = 5  

local outputWdith_L2 = 5
local outputWdith_L3 = 10


----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> construct CNN')

local CNN = nn.Sequential()

-- stage 1: conv+max
CNN:add(nn.SpatialConvolutionMM(nfeats, convDepth_L1,receptiveFieldWidth_L1,receptiveFieldHeight_L1))  
-- Since the default stride of the receptive field is 1, then 
-- (assuming receptiveFieldWidth_L1 = receptiveFieldHeight_L1 = 5)  the number of receptive fields is (32-5+1)x(32-5+1) or 28x28
-- so the output volume is (convDepth_L1 X 28 X 28) or 10 x 28 x 28

--CNN:add(nn.Threshold())
CNN:add(nn.ReLU())
CNN:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize)) 
-- if poolsize=2, then the output of this is 10x14x14

-- stage 2: conv+max
CNN:add(nn.SpatialConvolutionMM(convDepth_L1,convDepth_L2, receptiveFieldWidth_L2,receptiveFieldHeight_L2))  
-- Since the default stride of the receptive field is 1, then 
-- (assuming receptiveFieldWidth_L2 = receptiveFieldHeight_L2 = 5)  the number of receptive fields is (14-5+1)x(14-5+1) or 10x10
-- so the output volume is (convDepth_L2 x 10 x 10) or 5x10x10


--CNN:add(nn.Threshold())
CNN:add(nn.ReLU())
CNN:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize)) 
-- if poolsize=2, then the output of this is 5x5x5

local classifier = nn.Sequential()
-- stage 3: linear

classifier:add(nn.View(convDepth_L2*outputWdith_L2*outputWdith_L2))
--classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(convDepth_L2*outputWdith_L2*outputWdith_L2,outputWdith_L3))
classifier:add(nn.Threshold())
classifier:add(nn.Linear(outputWdith_L3,noutputs))
-- stage 4 : log probabilities
classifier:add(nn.LogSoftMax())

for _,layer in ipairs(CNN.modules) do
   if layer.bias then
      layer.bias:fill(.2)
      if i == #CNN.modules-1 then
         layer.bias:zero()
      end
   end
end

local model = nn.Sequential()
model:add(CNN)
model:add(classifier)

-- Loss: NLL
loss = nn.ClassNLLCriterion()


----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> here is the CNN:')
print(model)

if opt.type == 'cuda' then
   model:cuda()
   loss:cuda()
end

-- return package:
return {
   model = model,
   loss = loss,
}

