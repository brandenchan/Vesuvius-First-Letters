import pytorch_lightning as pl
import random
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.optim import  AdamW
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import torch.nn as nn
import torch
from warmup_scheduler import GradualWarmupScheduler
from tap import Tap
from PIL import Image
import numpy as np
import scipy.stats as st
from collections import OrderedDict
from i3dall import InceptionI3d




def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel."""

    x = np.linspace(
        start=-nsig,
        stop=nsig,
        num=kernlen+1
    )
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d / kern2d.sum()


def set_seed(seed=None, cudnn_deterministic=True):
    if seed is None:
        seed = 42
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = False
    return seed


def read_image_and_mask(
    fragment_id,
    tile_size,
    data_path,
    start_idx=18,
    end_idx=38,
    reverse=False,
    rotation=0
):

    images = []

    idxs = range(start_idx, end_idx)
    dataset_path=data_path
    for i in idxs:

        layer_name = f"{dataset_path}/{fragment_id}/layers/{i:02}.tif"
        if not os.path.exists(layer_name):
            raise Exception(f"No layer file found at {layer_name}")

        image = cv2.imread(layer_name, 0)

        pad0 = (tile_size - image.shape[0] % tile_size)
        pad1 = (tile_size - image.shape[1] % tile_size)

        image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)
        image=np.clip(image,0,200)

        images.append(image)
    images = np.stack(images, axis=2)

    if fragment_id in ['20230925002745','20230926164853'] or reverse:
        images=images[:,:,::-1]

    fragment_mask=None
    if os.path.exists(f'{dataset_path}/{fragment_id}/{fragment_id}_mask.png'):
        fragment_mask=cv2.imread(f"{dataset_path}/{fragment_id}/{fragment_id}_mask.png", 0)
        fragment_mask = np.pad(fragment_mask, [(0, pad0), (0, pad1)], constant_values=0)
        kernel = np.ones((16,16),np.uint8)
        fragment_mask = cv2.erode(fragment_mask,kernel,iterations = 1)
    return images, fragment_mask


def get_img_splits(
    fragment_id,
    s,
    e,
    tile_size,
    stride,
    data_path,
    rotation=0,
):
    images = []
    xyxys = []

    image, fragment_mask = read_image_and_mask(
        fragment_id=fragment_id,
        start_idx=s,
        end_idx=e,
        rotation=rotation,
        tile_size=tile_size,
        data_path=data_path
    )
    pred_shape = (image.shape[0], image.shape[1])

    # Get subsections of image and save the coords of subsections in xyxys
    x1_list = list(range(0, image.shape[1]-tile_size+1, stride))
    y1_list = list(range(0, image.shape[0]-tile_size+1, stride))
    for y1 in y1_list:
        for x1 in x1_list:
            y2 = y1 + tile_size
            x2 = x1 + tile_size

            # Only fetch subsections where there are no 0s
            # Presumably to not train on the black space around the segments
            if not np.any(fragment_mask[y1:y2, x1:x2] == 0):
                images.append(image[y1:y2, x1:x2])
                xyxys.append([x1, y1, x2, y2])

    return images, np.stack(xyxys), fragment_mask, pred_shape


class CustomDatasetTest(Dataset):
    def __init__(self, images, xyxys, transform=None):
        self.images = images
        self.transform = transform
        if transform:
            self.images = self.transform_batch(images)
        self.xyxys = xyxys

    def transform_batch(self, images):
        print("Applying transform function to dataset.")
        data = []
        for image in images:
            curr_image = self.transform(image=image)
            curr_image_data = curr_image["image"].unsqueeze(0)
            data.append(curr_image_data)
        print("Transform done!")
        return data

    def __len__(self):
        # return len(self.df)
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        xy = self.xyxys[idx]

        return image, xy


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m, mode='fan_out', nonlinearity='relu')
class Decoder(nn.Module):
    def __init__(self, encoder_dims, upscale):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(encoder_dims[i]+encoder_dims[i-1], encoder_dims[i-1], 3, 1, 1, bias=False),
                nn.BatchNorm2d(encoder_dims[i-1]),
                nn.ReLU(inplace=True)
            ) for i in range(1, len(encoder_dims))])

        self.logit = nn.Conv2d(encoder_dims[0], 1, 1, 1, 0)
        self.up = nn.Upsample(scale_factor=upscale, mode="bilinear")
        # for l in self.convs:
        #     for m in l._modules:
        #         init_weights(m)
    def forward(self, feature_maps):
        for i in range(len(feature_maps)-1, 0, -1):
            f_up = F.interpolate(feature_maps[i], scale_factor=2, mode="bilinear")
            f = torch.cat([feature_maps[i-1], f_up], dim=1)
            f_down = self.convs[i-1](f)
            feature_maps[i-1] = f_down

        x = self.logit(feature_maps[0])
        mask = self.up(x)
        return mask

 

def normalization(x):
    """input.shape=(batch,f1,f2,...)"""
    #[batch,f1,f2]->dim[1,2]
    dim=list(range(1,x.ndim))
    mean=x.mean(dim=dim,keepdim=True)
    std=x.std(dim=dim,keepdim=True)
    return (x-mean)/(std+1e-9)


class RegressionPLModel(pl.LightningModule):
    def __init__(self,pred_shape,size=224,enc='',with_norm=False):
        super(RegressionPLModel, self).__init__()

        self.save_hyperparameters()
        self.mask_pred = np.zeros(self.hparams.pred_shape)
        self.mask_count = np.zeros(self.hparams.pred_shape)
        # self.backbone=SegModel(model_depth=50)
        self.loss_func1 = smp.losses.DiceLoss(mode='binary')
        # self.loss_func2= smp.losses.FocalLoss(mode='binary',gamma=2)
        self.loss_func2= smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.15)
        # self.loss_func=nn.HuberLoss(delta=5.0)
        self.loss_func= lambda x,y:0.5 * self.loss_func1(x,y)+0.5*self.loss_func2(x,y)
        
        # self.backbone = generate_model(model_depth=50, n_input_channels=1,forward_features=True,n_classes=700)
        if self.hparams.enc=='resnet34':
            self.backbone = generate_model(model_depth=34, n_input_channels=1,forward_features=True,n_classes=700)
            state_dict=torch.load('./r3d34_K_200ep.pth')["state_dict"]
            conv1_weight = state_dict['conv1.weight']
            state_dict['conv1.weight'] = conv1_weight.sum(dim=1, keepdim=True)
            self.backbone.load_state_dict(state_dict,strict=False)
        elif self.hparams.enc=='resnest101':
            self.backbone = generate_model(model_depth=101, n_input_channels=1,forward_features=True,n_classes=1039)
            state_dict=torch.load('./r3d101_KM_200ep.pth')["state_dict"]
            conv1_weight = state_dict['conv1.weight']
            state_dict['conv1.weight'] = conv1_weight.sum(dim=1, keepdim=True)
            self.backbone.load_state_dict(state_dict,strict=False)
        elif self.hparams.enc=='2p1d':
            self.backbone = generate_2p1d(model_depth=34, n_input_channels=1,n_classes=700)
            state_dict=torch.load('./r2p1d34_K_200ep.pth')["state_dict"]
            conv1_weight = state_dict['conv1_s.weight']
            state_dict['conv1_s.weight'] = conv1_weight.sum(dim=1, keepdim=True)
            self.backbone.load_state_dict(state_dict,strict=False)
        elif self.hparams.enc=='wide50':
            self.backbone = generate_wide(model_depth=50, n_input_channels=1,n_classes=700,forward_features=True,k=2)
        elif self.hparams.enc=='i3d':
            self.backbone=InceptionI3d(in_channels=1,num_classes=512)
        elif self.hparams.enc=='resnext101':
            self.backbone=resnext101(sample_size=112,
                                  sample_duration=16,
                                  shortcut_type='B',
                                  cardinality=32,
                                  num_classes=600)
            state_dict = torch.load('./kinetics_resnext_101_RGB_16_best.pth')['state_dict']
            checkpoint_custom = OrderedDict()
            for key_model, key_checkpoint in zip(self.backbone.state_dict().keys(), state_dict.keys()):
                checkpoint_custom.update({f'{key_model}': state_dict[f'{key_checkpoint}']})

            self.backbone.load_state_dict(checkpoint_custom, strict=True)
            self.backbone.conv1 = nn.Conv3d(1, 64, kernel_size=(7, 7, 7), stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        else:
            self.backbone = generate_model(model_depth=50, n_input_channels=1,forward_features=True,n_classes=700)
            state_dict=torch.load('./r3d50_K_200ep.pth')["state_dict"]
            conv1_weight = state_dict['conv1.weight']
            state_dict['conv1.weight'] = conv1_weight.sum(dim=1, keepdim=True)
            self.backbone.load_state_dict(state_dict,strict=False)
        
        self.decoder = Decoder(encoder_dims=[x.size(1) for x in self.backbone(torch.rand(1,1,20,256,256))], upscale=1)

        if self.hparams.with_norm:
            self.normalization=nn.BatchNorm3d(num_features=1)

            
    def forward(self, x):
        if x.ndim==4:
            x=x[:,None]
        if self.hparams.with_norm:
            x=self.normalization(x)
        feat_maps = self.backbone(x)
        feat_maps_pooled = [torch.mean(f, dim=2) for f in feat_maps]
        pred_mask = self.decoder(feat_maps_pooled)
        
        return pred_mask

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        # outputs=torch.clip(outputs,min=-1,max=1)
        loss1 = self.loss_func(outputs, y)
        if torch.isnan(loss1):
            print("Loss nan encountered")
        # print(loss1)
        self.log("train/Arcface_loss", loss1.item(),on_step=True, on_epoch=True, prog_bar=True)
        # self.log("train/Accuracy_macro", acc_macro,on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss1}

    def validation_step(self, batch, batch_idx):
        x,y,xyxys= batch
        batch_size = x.size(0)
        outputs = self(x)
        loss1 = self.loss_func(outputs, y)
        y_preds = torch.sigmoid(outputs).to('cpu')
        for i, (x1, y1, x2, y2) in enumerate(xyxys):
            self.mask_pred[y1:y2, x1:x2] += F.interpolate(y_preds[i].unsqueeze(0).float(),scale_factor=4,mode='bilinear').squeeze(0).squeeze(0).numpy()
            self.mask_count[y1:y2, x1:x2] += np.ones((self.hparams.size, self.hparams.size))

        self.log("val/MSE_loss", loss1.item(),on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss1}
    
    def on_validation_epoch_end(self):
        self.mask_pred = np.divide(self.mask_pred, self.mask_count, out=np.zeros_like(self.mask_pred), where=self.mask_count!=0)
        wandb_logger.log_image(key="masks", images=[np.clip(self.mask_pred,0,1)], caption=["probs"])

        #reset mask
        self.mask_pred = np.zeros(self.hparams.pred_shape)
        self.mask_count = np.zeros(self.hparams.pred_shape)
    def configure_optimizers(self):

        optimizer = AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=CFG.lr)
    
        scheduler = get_scheduler(CFG, optimizer)
        return [optimizer]



class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    """
    https://www.kaggle.com/code/underwearfitting/single-fold-training-of-resnet200d-lb0-965
    """
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(
            optimizer, multiplier, total_epoch, after_scheduler)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

def get_scheduler(cfg, optimizer):
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, cfg.epochs, eta_min=1e-7)
    scheduler = GradualWarmupSchedulerV2(
        optimizer, multiplier=10, total_epoch=1, after_scheduler=scheduler_cosine)

    return scheduler


def predict_fn(
    test_loader,
    model,
    device,
    size,
    pred_shape
):
    mask_pred = np.zeros(pred_shape)
    mask_count = np.zeros(pred_shape)
    model.eval()
    kernel = gkern(size, 1)
    kernel = kernel / kernel.max()

    for step, (images, xys) in tqdm(enumerate(test_loader), total=len(test_loader)):
        images = images.to(device)
        with torch.no_grad():
            y_preds = model(images)

        y_preds = torch.sigmoid(y_preds).to('cpu')

        for i, (x1, y1, x2, y2) in enumerate(xys):

            interpolated = F.interpolate(
                y_preds[i].unsqueeze(0).float(),
                scale_factor=4,
                mode='bilinear'
            ).squeeze(0).squeeze(0).numpy()

            mask_pred[y1:y2, x1:x2] += np.multiply(
                interpolated,
                kernel
            )
            mask_count[y1:y2, x1:x2] += np.ones((size, size))

    mask_pred /= mask_count

    return mask_pred


class Config:
    def __init__(
        self,
        size,
        stride,
        batch_size,
        workers,
        data_path,
        segment_id,
        start_idx,
        model_path,
        out_path,
        device,
        warmup_factor=10,
        train_batch_size=None,
        *args,
        **kwargs
    ):

        self.data_path = data_path
        self.segment_id = segment_id
        self.model_path = model_path
        self.out_path = out_path

        # ============== comp exp name =============
        self.device = device
        comp_name = 'vesuvius'
        exp_name = 'pretraining_all'

        # ============== pred target =============
        self.target_size = 1

        # ============== model cfg =============
        self.model_name = 'Unet'
        self.backbone = 'efficientnet-b0'

        self.in_chans = 30  # 65
        self.encoder_depth = 5
        # ============== training cfg =============
        self.size = size
        self.tile_size = size
        self.stride = stride

        self.train_batch_size = batch_size  # 32
        self.valid_batch_size = self.train_batch_size
        self.use_amp = True

        self.scheduler = 'GradualWarmupSchedulerV2'
        self.epochs = 50  # 30

        self.warmup_factor = warmup_factor
        self.lr = 1e-4 / warmup_factor

        # ============== fold =============
        self.valid_id = 2

        self.metric_direction = 'maximize'  # maximize, 'minimize'

        # ============== fixed =============
        self.pretrained = True
        self.inf_weight = 'best'  # 'best'

        self.min_lr = 1e-6
        self.weight_decay = 1e-6
        self.max_grad_norm = 5

        self.print_freq = 50
        self.num_workers = workers
        self.start_idx = start_idx

        self.seed = set_seed()

        # ============== augmentation =============


        train_aug_list = [
            A.Resize(size, size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.75),
            A.ShiftScaleRotate(rotate_limit=90, shift_limit=0.1, scale_limit=0.1, p=0.5),
            A.OneOf([
                A.GaussNoise(var_limit=[10, 50]),
                A.GaussianBlur(),
                A.MotionBlur(),
            ], p=0.4),
            A.GridDistortion(num_steps=2, distort_limit=0.3, p=0.4),
            A.CoarseDropout(max_holes=5, max_width=int(size * 0.05), max_height=int(size * 0.05),
                            mask_fill_value=0, p=0.5),
            A.Normalize(
                mean=[0] * self.in_chans,
                std=[1] * self.in_chans
            ),
            ToTensorV2(transpose_mask=True),
        ]

        valid_aug_list = [
            A.Resize(size, size),
            A.Normalize(
                mean=[0] * self.in_chans,
                std=[1] * self.in_chans
            ),
            ToTensorV2(transpose_mask=True),
        ]






if __name__ == "__main__":

    class InferenceArgumentParser(Tap):
        """ Defines the command line arguments that are accepted.
        For an explanation of Tap see https://github.com/swansonk14/typed-argument-parser """
        segment_id: str = '20230504093154'
        data_path: str = './data'
        model_path: str = './models/valid_20230827161847_0_fr_i3depoch=7.ckpt'
        out_path: str = './inference_output'
        stride: int = 16
        start_idx: int = 15
        workers: int = os.cpu_count()
        batch_size: int = 512
        size: int = 64
        reverse: int = 0
        device: str = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = InferenceArgumentParser().parse_args().as_dict()
    config = Config(**args)

    images, xyxys, fragment_mask, pred_shape = get_img_splits(
        fragment_id=config.segment_id,
        s=config.start_idx,
        e=config.start_idx+30,
        rotation=0,
        tile_size=config.tile_size,
        stride=config.stride,
        data_path=config.data_path
    )

    transform_list = A.Compose(
        [
            A.Resize(
                config.size,
                config.size
            ),
            A.Normalize(
                mean=[0] * config.in_chans,
                std=[1] * config.in_chans
            ),
            ToTensorV2(transpose_mask=True),
        ]
    )

    test_dataset = CustomDatasetTest(
        images=images,
        xyxys=xyxys,
        transform=transform_list
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.valid_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False
    )

    # Load Model
    model = RegressionPLModel.load_from_checkpoint(
        checkpoint_path=config.model_path,
        strict=False
    )
    model.to(config.device)

    # Make prediction
    mask_pred = predict_fn(
        test_loader=test_loader,
        model=model,
        device=config.device,
        pred_shape=pred_shape,
        size=config.size
    )

    # Rescale predction masks so that they are in a range from 0-255
    mask_pred = np.clip(
        a=np.nan_to_num(mask_pred),
        a_min=0,
        a_max=1
    )
    mask_pred /= mask_pred.max()
    mask_pred = (mask_pred * 255).astype(np.uint8)
    mask_pred = Image.fromarray(mask_pred)

    # Save prediction mask
    if not os.path.exists(config.out_path):
        os.mkdir(config.out_path)
    filename = f"{config.segment_id}_{config.stride}_{config.start_idx}.png"
    mask_pred.save(fp=f'{config.out_path}/{filename}')