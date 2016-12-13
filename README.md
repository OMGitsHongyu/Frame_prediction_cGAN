# Frame Prediction using cGAN
Interpolate and extrapolate video frames using cGAN

Authors: Hongyu Zhu, Xiaolong Wang and Siddha Ganju.

## Network architecture
### The GAN
<img src="https://github.com/OMGitsHongyu/Frame_prediction_cGAN/blob/master/figures/approach.png" width="500">
### The generator
<img src="https://github.com/OMGitsHongyu/Frame_prediction_cGAN/blob/master/figures/model_archi.png">
## Models and inputs

| Models | Generator (*G*) |  Discriminator (*D*) |
|------- | --------------- | ---------------------|
|Baseline | 1<sup>st</sup> frame + noise | future frame|
|*FlowGAN*| 1<sup>st</sup> frame + flow (128x128) + noise | flow (128x128) + future frame |
|*FlowGAN-comp*| 1<sup>st</sup> frame + flow (128x128) + noise | 1<sup>st</sup> frame + flow (128x128) + future frame |
|*FlowGAN-sim*| 1<sup>st</sup> frame + flow (1x40) + noise  | flow (1x40)  + future frame |

## Dataset and experiments
### Frame prediction -- UCF101
[The 6<sup>th</sup> frames](https://omgitshongyu.github.io/Frame_prediction_cGAN/html/train_ucf_pred_5frame_30/1.html)


[Every 5 frames recurrently](https://omgitshongyu.github.io/Frame_prediction_cGAN/html/train_ucf_pred_5frame_recurrent/1.html)
### Action Recognition -- HMDB51
### [Static Image Editing](https://omgitshongyu.github.io/Frame_prediction_cGAN/html/ms_coco_pred_5frame_30/1.html) -- MS COCO
  [GIFs with series of random flows](https://omgitshongyu.github.io/Frame_prediction_cGAN/html/ms_coco_pred_1frame_40flow/7.html)
