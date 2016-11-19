require 'torch'
require 'optim'

require 'nngraph'
require 'cunn'
require 'image'

opt = {
   dataset = 'coco',       -- imagenet / lsun / folder
   batchSize = 64,
   loadSize = 128,
   scale_coarse = 64, 
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
   name = 'train_coco_res_dcgan_ori64_sr_ori',
   noise = 'normal',       -- uniform / normal
   classnum = 81, 
   save_epoch = 5, 
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
