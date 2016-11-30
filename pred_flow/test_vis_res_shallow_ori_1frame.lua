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
   classnum = 81, 
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



model_G = torch.load('/scratch/hongyuz/models_coco/train_coco_res_dcgan_ori64_sr_ori/40_net_G.t7')

opt.div_num = 127.5
opt.sub_num = -1


paths.dofile('data/donkey_coco.lua')

local resultfile = '/scratch/hongyuz/models_coco/train_coco_res_dcgan_ori64_sr_ori/res_cls.txt' 
local file = torch.DiskFile(resultfile, "w")


function getSamples(dataset, N, beg)
  local resultpath = '/scratch/hongyuz/models_coco/train_coco_res_dcgan_ori64_sr_ori/'
  os.execute('mkdir -p '.. resultpath)
  local N = N or 8
  local noise_inputs = torch.Tensor(N, opt.noiseDim[1], opt.noiseDim[2], opt.noiseDim[3])
  local cond_inputs = torch.Tensor(N, opt.condDim[1], opt.condDim[2], opt.condDim[3])
  local cond_inputs2 = torch.Tensor(N, opt.classnum, 1, 1)
  local diff_input = torch.Tensor(N, opt.condDim[1], opt.condDim[2], opt.condDim[3])
  local cond_inputs_coarse = torch.Tensor(N, 3, opt.scale_coarse, opt.scale_coarse)
  local label_ids = torch.Tensor(N)

  -- Generate samples
  noise_inputs:normal(0, 1)
  batch_data = makeData_video(trainLoader:sample(N))
  -- batch_data = makeData_res(trainLoader:get(beg + 1, beg + N ) )

  diff_input:copy(batch_data[1])
  -- cond_inputs:copy(batch_data[2])
  -- cond_inputs2:copy(batch_data[3])
  -- label_ids:copy(batch_data[4])
  cond_inputs_coarse:copy(batch_data[3])
  -- print(batch_data[1]:size())
  -- print(batch_data[3]:size())

  local samples = model_G:forward({noise_inputs:cuda(), cond_inputs_coarse:cuda()}) 

  gt_imgs = diff_input:clone()

  for i=1,N do
      coarse_name = paths.concat(resultpath, string.format('%04d_coarse.jpg',i + beg))
      output_name = paths.concat(resultpath, string.format('%04d_diff.jpg',i + beg))
      img_name = paths.concat(resultpath, string.format('%04d_imgs.jpg',i + beg))
      ori_name = paths.concat(resultpath, string.format('%04d_ori.jpg',i + beg))

      output_img = samples[i]:float():clone() -- torch.add(cond_inputs[i]:float(), samples[i]:float())
      indx = output_img:gt(1):byte()
      indx2 = output_img:lt(-1):byte()
      output_img:maskedCopy(indx, cond_inputs[i]:float():maskedSelect(indx))
      output_img:maskedCopy(indx2, cond_inputs[i]:float():maskedSelect(indx2))

      output_img = (output_img + 1 ) * opt.div_num
      output_img = output_img:byte()
      image.save(img_name, output_img)

      -- diff_img = (samples[i] + 1 ) * opt.div_num
      -- diff_img = diff_img:byte():clone()
      -- image.save(output_name, diff_img)

      coarse_img = (cond_inputs_coarse[i] + 1 ) * opt.div_num
      coarse_img = coarse_img:byte():clone()
      image.save(coarse_name, coarse_img)


      ori_img = (gt_imgs[i] + 1 ) * opt.div_num
      ori_img = ori_img:byte():clone()
      image.save(ori_name, ori_img)

      file:writeFloat( label_ids[i] )

  end


  -- noise_inputs:normal(0, 1)
  -- local samples2 = model_G:forward({noise_inputs:cuda(), cond_inputs_coarse:cuda()}) 

  -- for i=1,N do
  --     output_name = paths.concat(resultpath, string.format('%04d_diff2.jpg',i + beg))
  --     img_name = paths.concat(resultpath, string.format('%04d_imgs2.jpg',i + beg))

  --     output_img = samples2[i]:float():clone() -- torch.add(cond_inputs[i]:float(), samples2[i]:float())
  --     indx = output_img:gt(1):byte()
  --     indx2 = output_img:lt(-1):byte()
  --     output_img:maskedCopy(indx, cond_inputs[i]:float():maskedSelect(indx))
  --     output_img:maskedCopy(indx2, cond_inputs[i]:float():maskedSelect(indx2))
  --     output_img = (output_img + 1 ) * opt.div_num
  --     output_img = output_img:byte()

  --     image.save(img_name, output_img)

  --     -- diff_img = (samples2[i] + 1 ) * opt.div_num
  --     -- diff_img = diff_img:byte():clone()
  --     -- image.save(output_name, diff_img)


  -- end
  

end





for i = 1,5 do 
  print(i)
  getSamples(trainData, opt.batchSize, (i - 1) * opt.batchSize )
end



-- torch.save('to_plot.t7', to_plot)

--disp.image(to_plot, {win=opt.window, width=700, title=opt.save})


file:close()





