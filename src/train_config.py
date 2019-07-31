"""Model running configuration."""
import cv2
import torch
import albumentations as A
from model import create_model

# default config
config = dict(
    # the directory where the output and checkpoints will be saved
    run_dir='output/run1',
    # data input dir
    data_dir='data/mrnet',
    debug=False,
    seed=1234,
    use_gpu=True,
    learning_rate=1e-05,
    weight_decay=0,
    epochs=50,
    rgb=True,
    savedir=None,
    im_type='all',
    save_model=True,
)

model = create_model()

optimizer = torch.optim.Adam(
      model.parameters(),
      config['learning_rate'], weight_decay=config['weight_decay'],
      betas=(0.9, 0.80))

scheduler = torch.optim.lr_scheduler.MultiStepLR(
     optimizer, milestones=[10, 15, 20, 25, 30, 35, 40, 45], gamma=0.1)

img_aug = A.Compose(
    [
      A.Resize(160, 160),
      A.HorizontalFlip(p=0.5),
      A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.12,
                rotate_limit=25, p=0.75, border_mode=cv2.BORDER_REPLICATE),
    ], p=1)


config.update(dict(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    transform=img_aug,
))
