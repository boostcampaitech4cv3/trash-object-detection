# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None) # in boostcamp default : dict(grad_clip=dict(max_keep_ckpts=3, interval=1))  
# learning policy
lr_config = dict(
    policy='CosineRestart', 
    periods=[3900 * 2 for _ in range(10)],
    restart_weights=[1 / (2 ** i) for i in range(10)],
    by_epoch = False,
    min_lr=1e-07)
runner = dict(type='EpochBasedRunner', max_epochs=20)