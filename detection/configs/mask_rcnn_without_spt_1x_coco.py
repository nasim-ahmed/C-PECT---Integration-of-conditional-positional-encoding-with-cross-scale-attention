_base_ = './mask_rcnn_pcpvt_s_fpn_1x_coco_pvt_setting.py'

model = dict(
    pretrained='pretrained/alt_gvt_small.pth',
    backbone=dict(
        type='tft_gvt_small',
        style='pytorch'),
    neck=dict(
        in_channels=[64, 128, 256, 512],
        out_channels=256))