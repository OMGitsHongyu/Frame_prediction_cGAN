# Frame_prediction_cGAN
Interpolate and extrapolate video frames using cGAN

Authors: Hongyu Zhu, Xiaolong Wang and Siddha Ganju.

## Network architecture
### The GAN

### The generator

## Models and inputs

| Models | Generator (*G*) |  Discriminator (*D*) |
|------- | --------------- | ---------------------|
|Baseline | 1<sup>st</sup> frame + noise | future frame|
|*FlowGAN*| 1<sup>st</sup> frame + flow (128x128) + noise | flow (128x128) + future frame |
|*FlowGAN-comp*| 1<sup>st</sup> frame + flow (128x128) + noise | 1<sup>st</sup> frame + flow (128x128) + future frame |
|*FlowGAN-sim*| 1<sup>st</sup> frame + flow (1x40) + noise  | flow (1x40)  + future frame |

## Dataset and experiments
### Frame prediction -- UCF101

### Action Recognition -- HMDB51
### Static Image Editing -- MS COCO
