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
   scale_flow = 16, 
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



model_G = torch.load('/scratch/xiaolonw/models_ucf/train_ucf_pred_5frame_16s/32_net_G.t7')

opt.div_num = 127.5
opt.sub_num = -1


paths.dofile('data/donkey_coco.lua')

local resultfile = '/home/xiaolonw/ruslan/results/res_cls.txt' 
local file = torch.DiskFile(resultfile, "w")


function getSamples(dataset, N, beg)
  local resultpath = '/home/xiaolonw/ruslan/results/'
  os.execute('mkdir -p '.. resultpath)
  local N = N or 8
  local noise_inputs = torch.Tensor(N, opt.nz, 1, 1)
  local diff_input = torch.Tensor(N, opt.condDim[1], opt.condDim[2], opt.condDim[3])
  local cond_inputs_flow = torch.Tensor(opt.batchSize, 2, opt.condDim[2], opt.condDim[3])
  local cond_inputs_coarse = torch.Tensor(N, 3, opt.scale_coarse, opt.scale_coarse)
  local label_ids = torch.Tensor(N)

  -- Generate samples
  noise_inputs:normal(0, 1)
  batch_data = makeData_video_flow(trainLoader:sample(N))
  -- batch_data = makeData_res(trainLoader:get(beg + 1, beg + N ) )

  diff_input:copy(batch_data[1])
  cond_inputs_coarse:copy(batch_data[2])
  cond_inputs_flow:copy(batch_data[3]) 



  local samples = model_G:forward({noise_inputs:cuda(), cond_inputs_coarse:cuda(), cond_inputs_flow:cuda() }) 

  gt_imgs = diff_input:clone()

  for i=1,N do
      coarse_name = paths.concat(resultpath, string.format('%04d_input.jpg',i + beg))
      output_name = paths.concat(resultpath, string.format('%04d_pred.jpg',i + beg))
      img_name = paths.concat(resultpath, string.format('%04d_imgs.jpg',i + beg))
      ori_name = paths.concat(resultpath, string.format('%04d_gt.jpg',i + beg))

      output_img = samples[i]:float():clone() -- torch.add(cond_inputs[i]:float(), samples[i]:float())

      output_img = (output_img + 1 ) * opt.div_num
      output_img = output_img:byte()
      image.save(output_name, output_img)


      coarse_img = (cond_inputs_coarse[i] + 1 ) * opt.div_num
      coarse_img = coarse_img:byte():clone()
      image.save(coarse_name, coarse_img)


      ori_img = (gt_imgs[i] + 1 ) * opt.div_num
      ori_img = ori_img:byte():clone()
      image.save(ori_name, ori_img)

      file:writeFloat( label_ids[i] )

  end

  

end





for i = 1,5 do 
  print(i)
  getSamples(trainData, opt.batchSize, (i - 1) * opt.batchSize )
end



-- torch.save('to_plot.t7', to_plot)

--disp.image(to_plot, {win=opt.window, width=700, title=opt.save})


file:close()





