require 'torch'
require 'optim'

require 'nngraph'
require 'cunn'
require 'image'

opt = {
   dataset = 'coco',       -- imagenet / lsun / folder
   batchSize = 64,
   loadSize = 128,
   scale_coarse = 128, 
   scale_flow = 32, 
   nz = 100,               -- #  of dim for Z
   ngf = 64,               -- #  of gen filters in first conv layer
   ndf = 64,               -- #  of discrim filters in first conv layer
   nThreads = 3,           -- #  of data loading threads to use
   niter = 40,             -- #  of iter at starting learning rate
   lr = 0.0002,            -- initial learning rate for adam
   beta1 = 0.5,            -- momentum term of adam
   ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
   display = 0,            -- display samples while training. 0 = false
   display_id = 10,        -- display window id.
   gpu = 1,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
   name = 'train_ucf_pred_5frame_32s',
   noise = 'normal',       -- uniform / normal
   classnum = 81, 
   save_epoch = 2, 
   mse_lamda = 1000,

}

opt.flag = 1

opt.network = ''
opt.geometry = {3, opt.loadSize, opt.loadSize}
opt.condDim = {3, opt.loadSize, opt.loadSize}

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then opt.display = false end

opt.manualSeed = torch.random(1, 10000) -- fix seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

-- create data loader
local DataLoader = paths.dofile('data/data.lua')
local data = DataLoader.new(opt.nThreads, opt.dataset, opt)
print("Dataset: " .. opt.dataset, " Size: ", data:size())
----------------------------------------------------------------------------
local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
--      m:noBias()
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end



if opt.network == '' then

  local nplanes = 64
  local inputsize = 128
  dx_I = nn.Identity()()
  dx_C = nn.Identity()()

  -- I and C
  dc1 = nn.JoinTable(2, 2)({dx_I, dx_C})  -- rgb + flowxy
  dh1 = nn.SpatialConvolution(3 + 2, nplanes, 5, 5, 2, 2, 2, 2)(dc1)  -- size / 2 64 
  -- db1 = nn.SpatialBatchNormalization(nplanes)(dh1)
  dr1 = nn.LeakyReLU(0.2, true)(dh1)

  dh2 = nn.SpatialConvolution(nplanes, nplanes * 2, 5, 5, 2, 2, 2, 2)(dr1) -- size / 2 32
  db2 = nn.SpatialBatchNormalization(nplanes * 2)(dh2)
  dr2 = nn.LeakyReLU(0.2, true)(db2)

  dh3 = nn.SpatialConvolution(nplanes * 2, nplanes * 4, 3, 3, 2, 2, 1, 1)(dr2) -- size / 2 16
  db3 = nn.SpatialBatchNormalization(nplanes * 4)(dh3)
  dr3 = nn.LeakyReLU(0.2, true)(db3)

  dh4 = nn.SpatialConvolution(nplanes * 4, nplanes * 8, 3, 3, 2, 2, 1, 1)(dr3) -- size / 2
  db4 = nn.SpatialBatchNormalization(nplanes * 8)(dh4)
  dr4 = nn.LeakyReLU(0.2, true)(db4)

  dh5 = nn.SpatialConvolution(nplanes * 8, nplanes * 2, 3, 3, 1, 1, 1, 1)(dr4) -- same size
  db5 = nn.SpatialBatchNormalization(nplanes * 2)(dh5)
  dr5 = nn.LeakyReLU(0.2, true)(db5)

  local outputsize3 =  nplanes * 2 * (inputsize / 16) * (inputsize / 16) 
  rshp = nn.Reshape(outputsize3)(dr5)
  dh6 = nn.Linear(outputsize3, 1)(rshp)
  dout = nn.Sigmoid()(dh6)
  netD = nn.gModule({dx_I, dx_C}, {dout})

  netD:apply(weights_init)
  

  ----------------------------------------------------------------------
  -- define G network to train
  -- my G network
  local nplanes = 64
  x_I = nn.Identity()()  -- 100-d
  x_C = nn.Identity()()  -- input frame 
  x_C2 = nn.Identity()() -- flow 

  hi1 = nn.SpatialFullConvolution(opt.nz, nplanes * 2 , 8, 8)(x_I) 
  bi1 = nn.SpatialBatchNormalization(nplanes * 2)(hi1)
  ri1 = nn.ReLU(true)(bi1)

  hi1_1 = nn.SpatialFullConvolution(nplanes * 2, nplanes * 2, 4, 4, 2, 2, 1, 1)(ri1) 
  bi1_1 = nn.SpatialBatchNormalization(nplanes * 2)(hi1_1)
  ri1_1 = nn.ReLU(true)(bi1_1)


  c1 = nn.JoinTable(2, 2)({ x_C, x_C2 })

  h1 = nn.SpatialConvolution(3 + 2, nplanes, 5, 5, 2, 2, 2, 2)(c1)  -- 64 * 64
  b1 = nn.SpatialBatchNormalization(nplanes)(h1) 
  r1 = nn.ReLU(true)(b1) 

  h2 = nn.SpatialConvolution( nplanes, nplanes * 2, 5, 5, 2, 2, 2, 2)(r1) -- size / 2  32 * 32
  b2 = nn.SpatialBatchNormalization(nplanes * 2)(h2) 
  r2 = nn.ReLU(true)(b2)

  h2_2 = nn.SpatialConvolution(nplanes * 2, nplanes * 4, 5, 5, 2, 2, 2, 2)(r2)  -- 16 * 16 
  b2_2 = nn.SpatialBatchNormalization(nplanes * 4)(h2_2) 
  r2_2 = nn.ReLU(true)(b2_2)

  c2 = nn.JoinTable(2, 2)({ri1_1, r2_2})  -- 16 * 16 

  h3 = nn.SpatialFullConvolution(nplanes*6, nplanes * 4, 4, 4, 2, 2, 1, 1)(c2) -- size * 2
  b3 = nn.SpatialBatchNormalization(nplanes * 4)(h3) 
  r3 = nn.ReLU(true)(b3)

  h6 = nn.SpatialFullConvolution(nplanes*4, nplanes * 2, 4, 4, 2, 2, 1, 1)(r3) -- size * 2
  b6 = nn.SpatialBatchNormalization(nplanes * 2)(h6) 
  r6 = nn.LeakyReLU(true)(b6)

  h6_2 = nn.SpatialConvolution(nplanes*2, nplanes * 2, 3, 3, 1, 1, 1, 1)(r6) -- same size
  b6_2 = nn.SpatialBatchNormalization(nplanes * 2)(h6_2) 
  r6_2 = nn.LeakyReLU(true)(b6_2)

  h7 = nn.SpatialFullConvolution(nplanes*2, nplanes, 4, 4, 2, 2, 1, 1)(r6_2) -- size * 2
  b7 = nn.SpatialBatchNormalization(nplanes)(h7) 
  r7 = nn.LeakyReLU(0.2, true)(b7)

  h9 = nn.SpatialConvolution(nplanes, 3 , 3, 3, 1, 1, 1, 1)(r7) -- same size

  gout = nn.Tanh()(h9)

  netG = nn.gModule({x_I, x_C, x_C2}, {gout})

  netG:apply(weights_init)


else
  print('<trainer> reloading previously trained network: ' .. opt.network)
  tmp = torch.load(opt.network)
  netD = tmp.D
  netG = tmp.G
end

-- loss function: negative log-likelihood
criterion = nn.BCECriterion()
criterion2 = nn.MSECriterion() 


---------------------------------------------------------------------------
optimStateG = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
optimStateD = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
----------------------------------------------------------------------------
-- local input = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
-- local noise = torch.Tensor(opt.batchSize, nz, 1, 1)
-- local label = torch.Tensor(opt.batchSize)



local inputs = torch.Tensor(opt.batchSize, opt.geometry[1], opt.geometry[2], opt.geometry[3])
local targets = torch.Tensor(opt.batchSize)
local targets_fake = torch.Tensor(opt.batchSize)

local noise_inputs = torch.Tensor(opt.batchSize, opt.nz, 1, 1)
local cond_inputs_flow = torch.Tensor(opt.batchSize, 2, opt.loadSize, opt.loadSize)
local cond_inputs_coarse = torch.Tensor(opt.batchSize, 3, opt.scale_coarse, opt.scale_coarse)
local gt_inputs= torch.Tensor(opt.batchSize, opt.geometry[1], opt.geometry[2], opt.geometry[3])

local errD, errG
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()
----------------------------------------------------------------------------
if opt.gpu > 0 then
   require 'cunn'
   cutorch.setDevice(opt.gpu)
   inputs = inputs:cuda();  targets = targets:cuda();  noise_inputs = noise_inputs:cuda()
   cond_inputs_flow = cond_inputs_flow:cuda(); targets_fake = targets_fake:cuda()
   cond_inputs_coarse = cond_inputs_coarse:cuda(); gt_inputs = gt_inputs:cuda()

   if pcall(require, 'cudnn') then
      require 'cudnn'
      cudnn.benchmark = true
      cudnn.convert(netG, cudnn)
      cudnn.convert(netD, cudnn)
   end
   netD:cuda();           netG:cuda();           criterion:cuda()
   criterion2:cuda()
end

local parametersD, gradParametersD = netD:getParameters()
local parametersG, gradParametersG = netG:getParameters()

if opt.display then disp = require 'display' end

noise_vis = noise_inputs:clone()
if opt.noise == 'uniform' then
    noise_vis:uniform(-1, 1)
elseif opt.noise == 'normal' then
    noise_vis:normal(0, 1)
end

-- create closure to evaluate f(X) and df/dX of discriminator
local fDx = function(x)
   gradParametersD:zero()

   -- train with real
   data_tm:reset(); data_tm:resume()
   local inputs_all = data:getBatch()
   data_tm:stop()


   inputs:copy(inputs_all[1])
   cond_inputs_coarse:copy(inputs_all[2])
   cond_inputs_flow:copy(inputs_all[3])

   targets:fill(1)
   local output = netD:forward( {inputs, cond_inputs_flow} )
   local errD_real = criterion:forward(output, targets)
   local df_do = criterion:backward(output, targets)
   netD:backward({inputs, cond_inputs_flow}, df_do)

   -- train with fake
   if opt.noise == 'uniform' then -- regenerate random noise
       noise_inputs:uniform(-1, 1)
   elseif opt.noise == 'normal' then
       noise_inputs:normal(0, 1)
   end

   local inputs_all2 = data:getBatch()

   inputs:copy(inputs_all2[1])
   cond_inputs_coarse:copy(inputs_all2[2])
   cond_inputs_flow:copy(inputs_all2[3])

   gt_inputs:copy(inputs)

   local fake = netG:forward({noise_inputs, cond_inputs_coarse, cond_inputs_flow}) 
   inputs:copy(fake)
   targets_fake:fill(0)

   local output = netD:forward({inputs, cond_inputs_flow})
   local errD_fake = criterion:forward(output, targets_fake)
   local df_do = criterion:backward(output, targets_fake)
   netD:backward({inputs, cond_inputs_flow}, df_do)

   errD = errD_real + errD_fake

   return errD, gradParametersD
end

-- create closure to evaluate f(X) and df/dX of generator
local fGx = function(x)
   gradParametersG:zero()

   --[[ the three lines below were already executed in fDx, so save computation
   noise:uniform(-1, 1) -- regenerate random noise
   local fake = netG:forward(noise)
   input:copy(fake) ]]--
   -- label:fill(real_label) -- fake labels are real for generator cost

   targets:fill(1) 

   local output = netD.output -- netD:forward(input) was already executed in fDx, so save computation
   errG = criterion:forward(output, targets)
   local df_do = criterion:backward(output, targets)
   netD:updateGradInput({inputs, cond_inputs_flow}, df_do)
   local df_dg = netD.gradInput[1] 

   local g_output = netG.output

   local mse_err = criterion2:forward(g_output, gt_inputs) 
   local mse_grad = criterion2:backward(g_output, gt_inputs) 
   
   mse_err = mse_err * opt.mse_lamda
   mse_grad = mse_grad * opt.mse_lamda

   print('mse error: ' .. mse_err)


   print(df_dg:norm())
   print(mse_grad:norm())

   df_dg = mse_grad + df_dg


   netG:backward({noise_inputs, cond_inputs_coarse, cond_inputs_flow}, df_dg)
   return errG, gradParametersG
end

-- train
for epoch = 1, opt.niter do
   epoch_tm:reset()
   local counter = 0
   for i = 1, math.min(data:size(), opt.ntrain), opt.batchSize do
      tm:reset()
      -- (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
      optim.adam(fDx, parametersD, optimStateD)

      -- (2) Update G network: maximize log(D(G(z)))
      optim.adam(fGx, parametersG, optimStateG)

      -- display
      counter = counter + 1
      if counter % 10 == 0 and opt.display then
          local fake = netG:forward(noise_vis)
          local real = data:getBatch()
          disp.image(fake, {win=opt.display_id, title=opt.name})
          disp.image(real, {win=opt.display_id * 3, title=opt.name})
      end

      -- logging
      if ((i-1) / opt.batchSize) % 1 == 0 then
         print(('Epoch: [%d][%8d / %8d]\t Time: %.3f  DataTime: %.3f  '
                   .. '  Err_G: %.4f  Err_D: %.4f'):format(
                 epoch, ((i-1) / opt.batchSize),
                 math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize),
                 tm:time().real, data_tm:time().real,
                 errG and errG or -1, errD and errD or -1))
      end
   end

   if epoch % opt.save_epoch == 0 then 

--       paths.mkdir('/nfs.yoda/xiaolonw/torch_projects/models_coco/' .. opt.name .. '/')
       paths.mkdir('/scratch/xiaolonw/models_ucf/' .. opt.name .. '/')
       parametersD, gradParametersD = nil, nil -- nil them to avoid spiking memory
       parametersG, gradParametersG = nil, nil
--       torch.save('/nfs.yoda/xiaolonw/torch_projects/models_coco/' .. opt.name .. '/' .. epoch .. '_net_G.t7', netG:clearState())
--       torch.save('/nfs.yoda/xiaolonw/torch_projects/models_coco/' .. opt.name .. '/' .. epoch .. '_net_D.t7', netD:clearState())
       torch.save('/scratch/xiaolonw/models_ucf/' .. opt.name .. '/' .. epoch .. '_net_G.t7', netG:clearState())
       torch.save('/scratch/xiaolonw/models_ucf/' .. opt.name .. '/' .. epoch .. '_net_D.t7', netD:clearState())
       parametersD, gradParametersD = netD:getParameters() -- reflatten the params and get them
       parametersG, gradParametersG = netG:getParameters()
       print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
                epoch, opt.niter, epoch_tm:time().real))

   end

end
