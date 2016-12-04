require 'torch'
require 'optim'

require 'nngraph'
require 'cunn'
require 'image'
require 'cudnn'

opt = {
   dataset = 'coco',       -- imagenet / lsun / folder
   batchSize = 128,
   loadSize = 128,
   scale_coarse = 128, 
   scale_flow = 128, 
   nz = 100,               -- #  of dim for Z
   ngf = 64,               -- #  of gen filters in first conv layer
   ndf = 64,               -- #  of discrim filters in first conv layer
   nThreads = 3,           -- #  of data loading threads to use
   niter = 40,             -- #  of iter at starting learning rate
   lr = 0.01,            -- initial learning rate for adam
   beta1 = 0.5,            -- momentum term of adam
   ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
   display = 0,            -- display samples while training. 0 = false
   display_id = 10,        -- display window id.
   gpu = 1,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
   name = 'train_hmdb_cls_frame5',
   noise = 'normal',       -- uniform / normal
   classnum = 51, 
   save_epoch = 2, 
   mse_lamda = 1000,
   weightDecay = 0.0005,
   momentum = 0.9

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
   if name:find('Linear')  then 
      if m.weight then m.weight:normal(0, 0.01) end
      if m.bias then m.bias:fill(0) end
   end
end


opt.pretrain = '/home/xiaolonw/ruslan/models_ucf/train_ucf_pred_5frame/32_net_D.t7'


if opt.network == '' then

  local nplanes = 64
  local inputsize = 128
  local outputsize3 =  nplanes * 2 * (inputsize / 16) * (inputsize / 16) 

  pretrain_netD = torch.load(opt.pretrain) 

  netD = nn.Sequential()

  for i = 3, 18 do 
    netD:add(pretrain_netD.modules[i])
  end

  netD:add(nn.Linear(outputsize3, 2046))
  netD:add(nn.BatchNormalization(2046))
  netD:add(nn.LeakyReLU(0.2, true))
  netD:add(nn.Linear(2046, opt.classnum))
  netD:add(nn.LogSoftMax())

  netD:apply(weights_init)

else
  print('<trainer> reloading previously trained network: ' .. opt.network)
  tmp = torch.load(opt.network)
  netD = tmp.D



end

-- loss function: negative log-likelihood
criterion = nn.ClassNLLCriterion()


optimStateD = {
    learningRate = opt.lr,
    learningRateDecay = 0.0,
    momentum = opt.momentum,
    dampening = 0.0,
    weightDecay = opt.weightDecay
}


local inputs = torch.Tensor(opt.batchSize, opt.geometry[1], opt.geometry[2], opt.geometry[3])
local targets = torch.Tensor(opt.batchSize)
local targets_fake = torch.Tensor(opt.batchSize)

local noise_inputs = torch.Tensor(opt.batchSize, opt.nz, 1, 1)
local cond_inputs_flow = torch.Tensor(opt.batchSize, 2, opt.scale_flow, opt.scale_flow)
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
      cudnn.convert(netD, cudnn)
   end
   netD:cuda();           criterion:cuda()
end

local parametersD, gradParametersD = netD:getParameters()

if opt.display then disp = require 'display' end


-- create closure to evaluate f(X) and df/dX of discriminator
local fDx = function(x)
   gradParametersD:zero()

   -- train with real
   data_tm:reset(); data_tm:resume()
   local inputs_all = data:getBatch()
   data_tm:stop()


   targets:copy(inputs_all[1])
   cond_inputs_coarse:copy(inputs_all[2])
   cond_inputs_flow:copy(inputs_all[3])

   local output = netD:forward( {cond_inputs_coarse, cond_inputs_flow} )
   local errD_real = criterion:forward(output, targets)
   local df_do = criterion:backward(output, targets)
   netD:backward({cond_inputs_coarse, cond_inputs_flow}, df_do)

   errD = errD_real

   return errD, gradParametersD
end

-- train
for epoch = 1, opt.niter do
   epoch_tm:reset()
   local counter = 0
   for i = 1, math.min(data:size(), opt.ntrain), opt.batchSize do
      tm:reset()
      -- (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
      optim.sgd(fDx, parametersD, optimStateD)

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
                   .. ' Err_D: %.4f'):format(
                 epoch, ((i-1) / opt.batchSize),
                 math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize),
                 tm:time().real, data_tm:time().real, errD and errD or -1))
      end
   end

   if epoch % opt.save_epoch == 0 then 

--       paths.mkdir('/nfs.yoda/xiaolonw/torch_projects/models_coco/' .. opt.name .. '/')
       paths.mkdir('/scratch/xiaolonw/models_ucf/' .. opt.name .. '/')
       parametersD, gradParametersD = nil, nil -- nil them to avoid spiking memory
       torch.save('/scratch/xiaolonw/models_ucf/' .. opt.name .. '/' .. epoch .. '_net_D.t7', netD:clearState())
       parametersD, gradParametersD = netD:getParameters() -- reflatten the params and get them
       print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
                epoch, opt.niter, epoch_tm:time().real))

   end

end
