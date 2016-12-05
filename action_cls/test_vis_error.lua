require 'torch'
require 'nngraph'
require 'cunn'
require 'optim'
require 'image'
require 'pl'
require 'paths'
require 'cudnn'

-- model path
-- dataset
-- class number
-- clean folder


ok, disp = pcall(require, 'display')
if not ok then print('display not found. unable to plot') end

opt = {
   dataset = 'coco',       -- imagenet / lsun / folder
   batchSize = 100,
   loadSize = 128,
   scale_coarse = 128, 
   scale_flow = 128, 
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
   name = 'train_coco_res_dcgan_ori',
   noise = 'normal',       -- uniform / normal
   classnum = 51, 
   save_epoch = 3, 

}


opt.network = ''
opt.geometry = {3, opt.loadSize, opt.loadSize}
opt.condDim = {3, opt.loadSize, opt.loadSize}
-- opt.noiseDim = {opt.nz, 1, 1}
opt.noiseDim = {1, opt.scale_coarse, opt.scale_coarse}

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then opt.display = false end

opt.manualSeed = 1 -- torch.random(1, 10000) -- fix seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.CudaTensor')

-- create data loader
-- local DataLoader = paths.dofile('data/data.lua')
-- local data = DataLoader.new(opt.nThreads, opt.dataset, opt)
-- print("Dataset: " .. opt.dataset, " Size: ", data:size())



model_G = torch.load('/scratch/xiaolonw/models_ucf/train_hmdb_cls_frame10/26_net_D.t7')
model_G:evaluate()


opt.div_num = 127.5
opt.sub_num = -1


paths.dofile('data/donkey_coco.lua')

local resultfile = '/home/xiaolonw/ruslan/results/test_cls.txt' 
local file = torch.DiskFile(resultfile, "w")


local accuracy_sum = 0 

local video_len = 25


function getSamples(dataset, N, beg)
  local resultpath = '/home/xiaolonw/ruslan/results/'
  os.execute('mkdir -p '.. resultpath)
  local N = N or 8
  local noise_inputs = torch.Tensor(N, opt.nz, 1, 1)
  local diff_input = torch.Tensor(N, opt.condDim[1], opt.condDim[2], opt.condDim[3])
  local cond_inputs_flow = torch.Tensor(opt.batchSize, 2, opt.scale_flow, opt.scale_flow)
  local cond_inputs_coarse = torch.Tensor(N, 3, opt.scale_coarse, opt.scale_coarse)
  local targets = torch.Tensor(N)

  -- Generate samples
  noise_inputs:normal(0, 1)
  batch_data = makeData_video_flow(trainLoader:get(beg + 1, beg + N ))
  -- batch_data = makeData_res(trainLoader:get(beg + 1, beg + N ) )

  targets:copy(batch_data[1])
  cond_inputs_coarse:copy(batch_data[2])
  cond_inputs_flow:copy(batch_data[3]) 

  assert(N % video_len == 0)


  local samples = model_G:forward({cond_inputs_coarse:cuda(), cond_inputs_flow:cuda() }) 
  local batch_accuracy = 0
  local batch_size = N / video_len
  samples = samples:float()

  for i=1, batch_size do
      
      prop = torch.Tensor(opt.classnum):fill(0)
      for j = 1, video_len do 
        -- print(samples[(i - 1) * video_len + j]:size() )
        -- print(prop:size())
        prop = prop + samples[(i - 1) * video_len + j] 
      end
      maxs, indices = torch.max(prop, 1)
      indx = indices[1]
      if indx == targets[(i - 1) * video_len + 1]  then
        batch_accuracy = batch_accuracy + 1
      end

  end
  accuracy_sum   = accuracy_sum + batch_accuracy
  print(batch_accuracy)

  
end


batchnum = 363
-- batchnum = 623

for i = 1,batchnum do 
  print(i)
  getSamples(trainData, opt.batchSize, (i - 1) * opt.batchSize )
end


accuracy_sum = accuracy_sum / (363 * 4)
print(accuracy_sum)
print(accuracy_sum)


-- torch.save('to_plot.t7', to_plot)

--disp.image(to_plot, {win=opt.window, width=700, title=opt.save})


file:close()





