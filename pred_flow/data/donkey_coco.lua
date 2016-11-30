--[[
    Copyright (c) 2015-present, Facebook, Inc.
    All rights reserved.

    This source code is licensed under the BSD-style license found in the
    LICENSE file in the root directory of this source tree. An additional grant
    of patent rights can be found in the PATENTS file in the same directory.
]]--

require 'image'
require 'struct'
require 'image'
require 'string'

paths.dofile('dataset.lua')

-- This file contains the data-loading logic and details.
-- It is run by each data-loader thread.
------------------------------------------
-------- COMMON CACHES and PATHS
-- a cache file of the training metadata (if doesnt exist, will be created)
-- local cache = "cache_coco"
local cache = "cache_pred"
os.execute('mkdir -p '..cache)
local trainCache = paths.concat(cache, 'trainCache.t7')
--local trainCache = paths.concat(cache, 'testCache.t7')
local testCache = paths.concat(cache, 'testCache.t7')
local meanstdCache = paths.concat(cache, 'meanstdCache.t7')

-- Check for existence of opt.data
opt.data = os.getenv('DATA_ROOT') or '../logs'
--------------------------------------------------------------------------------------------
if not os.execute('cd ' .. opt.data) then
    error(("could not chdir to '%s'"):format(opt.data))
end

local loadSize   = {3, opt.loadSize}
local sampleSize = {3, opt.loadSize}

local function loadImage(path)
   local input = image.load(path, 3, 'float')
   input = image.scale(input, opt.loadSize, opt.loadSize, 'bilinear')
   input = input * 255
   return input
end


local function loadImage_flow(path)
   local input = image.load(path, 1, 'float')
   input = image.scale(input, opt.loadSize, opt.loadSize, 'bilinear')
   input = input * 255
   return input
end


local savepath = '/home/xiaolonw/t_imgs/'

function saveData(img, imgname)
  img = (img + 1 ) * 127.5
  img = img:byte()
  image.save(imgname, img )
end


function makeData(fine, classes)

   local coarse_size = opt.scale_coarse
   local sample_num  = (#(fine))[1]
   local channel_num = (#(fine))[2]
   local coarse_input0 = torch.Tensor(sample_num, channel_num, coarse_size, coarse_size)
   local coarse_input = torch.Tensor(sample_num, channel_num, opt.loadSize, opt.loadSize) 
   local classesids = torch.Tensor(sample_num) 

   for i = 1, sample_num do 

     local now_fine = fine[i]:clone() 
     local t_coarse_input = image.scale(now_fine, coarse_size, coarse_size):clone()
     local t_coarse_input2  = image.scale(t_coarse_input, opt.loadSize, opt.loadSize):clone()
     coarse_input0[i] = t_coarse_input:clone()
     coarse_input[i] = t_coarse_input2:clone()
     local maxs, indices = torch.max(classes[i], 1)
     classesids[i] = indices[1]

       if opt.flag == 1  then 
        local hpatch_name = paths.concat(savepath, string.format('%04d_coarse_input.jpg',i ))
        local coarse_input_name = paths.concat(savepath, string.format('%04d_ori.jpg',i ))
        saveData(fine[i]:clone(), hpatch_name)
        saveData(coarse_input[i]:clone(), coarse_input_name)
       end



   end

   opt.flag  = 0
   return {fine, coarse_input, classes, classesids, coarse_input0}
end



function makeData_video(fine, fine2)

   local coarse_size = opt.scale_coarse
   local sample_num  = (#(fine))[1]
   local channel_num = (#(fine))[2]
   local coarse_input0 = torch.Tensor(sample_num, channel_num, coarse_size, coarse_size)
   local coarse_input = torch.Tensor(sample_num, channel_num, opt.loadSize, opt.loadSize)

   for i = 1, sample_num do

     local now_fine = fine[i]:clone()
     local t_coarse_input = image.scale(now_fine, coarse_size, coarse_size):clone()
     local t_coarse_input2  = image.scale(t_coarse_input, opt.loadSize, opt.loadSize):clone()
     coarse_input0[i] = t_coarse_input:clone()
     coarse_input[i] = t_coarse_input2:clone()

       if opt.flag == 1  then
        local hpatch_name = paths.concat(savepath, string.format('%04d_ori.jpg',i ))
        local coarse_input_name = paths.concat(savepath, string.format('%04d_coarse_input.jpg',i ))
        saveData(fine[i]:clone(), hpatch_name)
        saveData(coarse_input[i]:clone(), coarse_input_name)
       end



   end

   opt.flag  = 0
   return {fine2, coarse_input,  coarse_input0}
end





function makeData_video_flow(fine, fine2, flowx, flowy)

   local coarse_size = opt.scale_coarse
   local flow_size   = opt.scale_flow
   local sample_num  = (#(fine))[1]
   local channel_num = (#(fine))[2]
   local coarse_input0 = torch.Tensor(sample_num, channel_num, coarse_size, coarse_size)
   local flow_input = torch.Tensor(sample_num, 2, flow_size, flow_size)

   for i = 1, sample_num do

     local now_fine = fine[i]:clone()
     local t_coarse_input = image.scale(now_fine, coarse_size, coarse_size):clone()
     coarse_input0[i] = t_coarse_input:clone()
     local now_flowx = flowx[i]:clone()
     local now_flowy = flowy[i]:clone()
     now_flowx = image.scale(now_flowx, flow_size, flow_size):clone()
     now_flowy = image.scale(now_flowy, flow_size, flow_size):clone()

     flow_input[{{i}, {1}, {}, {}}]:copy(now_flowx)
     flow_input[{{i}, {2}, {}, {}}]:copy(now_flowy) 


       if opt.flag == 1  then
        local hpatch_name = paths.concat(savepath, string.format('%04d_ori.jpg',i ))
        local coarse_input_name = paths.concat(savepath, string.format('%04d_coarse_input.jpg',i ))
        local flowx_name = paths.concat(savepath, string.format('%04d_flowx.jpg',i ))
        local flowy_name = paths.concat(savepath, string.format('%04d_flowy.jpg',i ))
        saveData(fine[i]:clone(), hpatch_name)
        saveData(coarse_input0[i]:clone(), coarse_input_name)
        saveData(flow_input[i][1]:clone(), flowx_name)
        saveData(flow_input[i][2]:clone(), flowy_name)
       end

   end

   opt.flag  = 0
   return {fine2,  coarse_input0, flow_input} 
end




function makeData_res(fine, classes)

   local coarse_size = opt.scale_coarse
   local sample_num  = (#(fine))[1]
   local channel_num = (#(fine))[2]
   local coarse_input0 = torch.Tensor(sample_num, channel_num, coarse_size, coarse_size)
   local coarse_input = torch.Tensor(sample_num, channel_num, opt.loadSize, opt.loadSize)  
   local diff_input   = torch.Tensor(sample_num, channel_num, opt.loadSize, opt.loadSize) 
   local classesids = torch.Tensor(sample_num) 

   for i = 1, sample_num do 

     local now_fine = fine[i]:clone() 
     local t_coarse_input = image.scale(now_fine, coarse_size, coarse_size, 'bilinear'):clone()
     local t_coarse_input2  = image.scale(t_coarse_input, opt.loadSize, opt.loadSize, 'bilinear'):clone()
     coarse_input0[i] = t_coarse_input:clone()
     coarse_input[i] = t_coarse_input2:clone()
     diff_input[i] = torch.add(fine[i], -1, coarse_input[i]):clone()
     
     local maxs, indices = torch.max(classes[i], 1)
     classesids[i] = indices[1]

       if opt.flag == 1  then 
        local hpatch_name = paths.concat(savepath, string.format('%04d_ori.jpg',i ))
        local coarse_input_name = paths.concat(savepath, string.format('%04d_coarse_input.jpg',i ))
        local diff_name = paths.concat(savepath, string.format('%04d_diff.jpg',i ))
        saveData(fine[i]:clone(), hpatch_name)
        saveData(coarse_input[i]:clone(), coarse_input_name)
        saveData(diff_input[i]:clone(), diff_name)
       end

   end

   opt.flag  = 0
   return {diff_input, coarse_input, classes, classesids, coarse_input0}
end




-- channel-wise mean and std. Calculate or load them from disk later in the script.
local div_num, sub_num
div_num = 127.5
sub_num = -1
--------------------------------------------------------------------------------
-- Hooks that are used for each image that is loaded

-- function to load the image, jitter it appropriately (random crops etc.)
local trainHook = function(self, imgpath, lblnum, flowxpath, flowypath)
   collectgarbage()
   local img = loadImage(imgpath)
   local lbl = loadImage(lblnum)
   local flowx = loadImage_flow(flowxpath)
   local flowy = loadImage_flow(flowypath)
   img:div(div_num)
   img:add(sub_num)
   
   lbl:div(div_num)
   lbl:add(sub_num)

   flowx:div(div_num)
   flowx:add(sub_num)

   flowy:div(div_num)
   flowy:add(sub_num)

   return img, lbl, flowx, flowy 

end

--------------------------------------
-- trainLoader
if paths.filep(trainCache) then
   print('Loading train metadata from cache')
   trainLoader = torch.load(trainCache)
   trainLoader.sampleHookTrain = trainHook
   trainLoader.loadSize = {3, opt.loadSize, opt.loadSize}
   trainLoader.sampleSize = {3, sampleSize[2], sampleSize[2]}
else
   print('Creating train metadata')
   trainLoader = dataLoader{
      paths = {paths.concat(opt.data, 'train')},
      loadSize = {3, loadSize[2], loadSize[2]},
      sampleSize = {3, sampleSize[2], sampleSize[2]},
      split = 100,
      verbose = true
   }
   print(trainLoader)
   torch.save(trainCache, trainLoader)
   trainLoader.sampleHookTrain = trainHook
end
collectgarbage()


