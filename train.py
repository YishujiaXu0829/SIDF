"""DualUCNN training script."""


import math
import json
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.utils as vutils
from PIL import Image
from tqdm import tqdm

from mosaic import PolarizationMosaicPipeline
from model import DualUCNN

try:
    import scipy.io as sio
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

try:
    import h5py as _h5py
    _HAS_H5PY = True
except ImportError:
    _HAS_H5PY = False



def _safe_tensor(x, nan=0.0, posinf=1.0, neginf=0.0):
    return torch.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)


def charb(a, b, eps=1e-6):
    a = _safe_tensor(a)
    b = _safe_tensor(b)
    return torch.mean(torch.sqrt((a - b) ** 2 + eps))


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size
        self.register_buffer('window', self._make_window(window_size))

    def _make_window(self, ws):
        g = torch.tensor([math.exp(-(x-(ws//2))**2/2.) for x in range(ws)], dtype=torch.float32)
        g /= g.sum()
        return (g.unsqueeze(1) * g.unsqueeze(0)).unsqueeze(0).unsqueeze(0)

    def _ssim_single(self, x, y):
        x = _safe_tensor(x)
        y = _safe_tensor(y)
        C1, C2 = 0.01**2, 0.03**2
        w = self.window.to(x.device)
        mu_x = F.conv2d(x, w, padding=self.window_size//2)
        mu_y = F.conv2d(y, w, padding=self.window_size//2)
        mu_x2, mu_y2, mu_xy = mu_x**2, mu_y**2, mu_x*mu_y
        sig_x  = F.conv2d(x*x, w, padding=self.window_size//2) - mu_x2
        sig_y  = F.conv2d(y*y, w, padding=self.window_size//2) - mu_y2
        sig_xy = F.conv2d(x*y, w, padding=self.window_size//2) - mu_xy
        num = (2*mu_xy+C1)*(2*sig_xy+C2)
        den = ((mu_x2+mu_y2+C1)*(sig_x+sig_y+C2)).clamp_min(1e-6)
        return torch.mean(_safe_tensor(num / den).clamp(-1.0, 1.0))

    def forward(self, pred, target):
                loss = 0.
        for i in range(pred.shape[1]):
            loss = loss + (1. - self._ssim_single(
                pred[:,i:i+1], target[:,i:i+1]))
        return loss / pred.shape[1]


def compute_polar_stokes_loss(pred_12ch, gt_12ch):
  
    pred_12ch = _safe_tensor(pred_12ch).clamp(0, 1)
    gt_12ch = _safe_tensor(gt_12ch).clamp(0, 1)
    loss = 0.
    for ci in range(3):
        I0   = pred_12ch[:, ci*4+0:ci*4+1]
        I45  = pred_12ch[:, ci*4+1:ci*4+2]
        I90  = pred_12ch[:, ci*4+2:ci*4+3]
        I135 = pred_12ch[:, ci*4+3:ci*4+4]
        G0   = gt_12ch[:,   ci*4+0:ci*4+1]
        G45  = gt_12ch[:,   ci*4+1:ci*4+2]
        G90  = gt_12ch[:,   ci*4+2:ci*4+3]
        G135 = gt_12ch[:,   ci*4+3:ci*4+4]
        s0p = (I0+I90)/2.;      s0g = (G0+G90)/2.
        s1p = (I0-I90+1.)/2.;   s1g = (G0-G90+1.)/2.
        s2p = (I45-I135+1.)/2.; s2g = (G45-G135+1.)/2.
        dop_p = (torch.sqrt((I0-I90)**2+(I45-I135)**2+1e-6) /
                 (I0+I90).clamp(1e-6)).clamp(0,1)
        dop_g = (torch.sqrt((G0-G90)**2+(G45-G135)**2+1e-6) /
                 (G0+G90).clamp(1e-6)).clamp(0,1)
        loss = loss + (charb(s0p,s0g)+charb(s1p,s1g)+
                       charb(s2p,s2g)+charb(dop_p,dop_g))/4.
    return loss / 3.


def compute_polar_physics_loss(pred_12ch, gt_12ch, eps=1e-6):
    """Polar auxiliary loss: Stokes + DoLP + AoP consistency."""
    pred_12ch = _safe_tensor(pred_12ch).clamp(0, 1)
    gt_12ch = _safe_tensor(gt_12ch).clamp(0, 1)
    loss = 0.
    for ci in range(3):
        p0 = pred_12ch[:, ci*4+0:ci*4+1]
        p45 = pred_12ch[:, ci*4+1:ci*4+2]
        p90 = pred_12ch[:, ci*4+2:ci*4+3]
        p135 = pred_12ch[:, ci*4+3:ci*4+4]
        g0 = gt_12ch[:, ci*4+0:ci*4+1]
        g45 = gt_12ch[:, ci*4+1:ci*4+2]
        g90 = gt_12ch[:, ci*4+2:ci*4+3]
        g135 = gt_12ch[:, ci*4+3:ci*4+4]

        ps0 = (p0 + p90).clamp(eps, 2.0)
        ps1 = p0 - p90
        ps2 = p45 - p135
        gs0 = (g0 + g90).clamp(eps, 2.0)
        gs1 = g0 - g90
        gs2 = g45 - g135

        pdolp = (torch.sqrt(ps1 ** 2 + ps2 ** 2 + eps) / ps0).clamp(0, 1)
        gdolp = (torch.sqrt(gs1 ** 2 + gs2 ** 2 + eps) / gs0).clamp(0, 1)

        pnorm = torch.sqrt(ps1 ** 2 + ps2 ** 2 + eps).clamp_min(eps)
        gnorm = torch.sqrt(gs1 ** 2 + gs2 ** 2 + eps).clamp_min(eps)
        pcos2a, psin2a = ps1 / pnorm, ps2 / pnorm
        gcos2a, gsin2a = gs1 / gnorm, gs2 / gnorm

        stokes_loss = charb(ps0, gs0) + charb(ps1, gs1) + charb(ps2, gs2)
        dolp_loss = charb(pdolp, gdolp)
        valid_aop = (gdolp.detach() > 0.02).float()
        aop_map = (torch.sqrt((pcos2a - gcos2a) ** 2 + eps) +
                   torch.sqrt((psin2a - gsin2a) ** 2 + eps))
        aop_loss = (aop_map * valid_aop).sum() / valid_aop.sum().clamp_min(1.0)
        loss = loss + stokes_loss + 0.5 * dolp_loss + 0.25 * aop_loss
    return _safe_tensor(loss / 3.)



_WATER_PARAMS = {
    'coastal':       {'airlight':[0.35,0.60,0.50],'factor_r':1.4,'factor_g':0.8, 'factor_b':1.0,'trans_ratio':0.15},
    'deep_ocean':    {'airlight':[0.20,0.35,0.65],'factor_r':2.0,'factor_g':1.2, 'factor_b':0.6,'trans_ratio':0.20},
    'clear_shallow': {'airlight':[0.25,0.50,0.60],'factor_r':1.5,'factor_g':0.9, 'factor_b':0.7,'trans_ratio':0.18},
    'turbid':        {'airlight':[0.45,0.50,0.40],'factor_r':1.2,'factor_g':0.9, 'factor_b':1.1,'trans_ratio':0.12},
}
_BETA_VALUES = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
_T1_WATER_TYPE = 't1_identity'
_ANGLES_RAD  = {'0':0.0, '45':math.pi/4, '90':math.pi/2, '135':3*math.pi/4}
_POL_DEGREE  = 0.3
_POL_ANGLE   = math.pi / 4.0


def get_random_params():
    water_type = np.random.choice(list(_WATER_PARAMS.keys()) + [_T1_WATER_TYPE])
    if water_type == _T1_WATER_TYPE:
        return {'water_type': water_type, 'beta': 0.0, 't_is_one': True}
    return {
        'water_type': water_type,
        'beta':       float(np.random.choice(_BETA_VALUES)),
        't_is_one':   False,
    }


def make_depth(H, W, device):
    y = torch.linspace(0,1,H,device=device)
    x = torch.linspace(0,1,W,device=device)
    yy,xx = torch.meshgrid(y,x,indexing='ij')
    d = torch.sqrt((xx-.5)**2+(yy-.5)**2)
    return (d/d.max()).unsqueeze(0).unsqueeze(0)


def degrade_batch(pol_0, pol_45, pol_90, pol_135, params, depth_map=None):
    if params.get('t_is_one', False):
        return pol_0, pol_45, pol_90, pol_135
    B,_,H,W = pol_0.shape
    device  = pol_0.device
    wp      = _WATER_PARAMS[params['water_type']]
    beta    = params['beta']
    if depth_map is not None:
        depth = F.interpolate(depth_map.to(device),(H,W),
                              mode='bilinear',align_corners=True).expand(B,1,H,W)
    else:
        depth = make_depth(H,W,device).expand(B,1,H,W)
    t_r = (torch.exp(-beta*wp['factor_r']*depth)*wp['trans_ratio']).clamp(0.01,1.)
    t_g = (torch.exp(-beta*wp['factor_g']*depth)*wp['trans_ratio']).clamp(0.01,1.)
    t_b = (torch.exp(-beta*wp['factor_b']*depth)*wp['trans_ratio']).clamp(0.01,1.)
    def _deg(img, ang):
        coeff = 0.5*(1.+_POL_DEGREE*math.cos(2.*(_ANGLES_RAD[ang]-_POL_ANGLE)))
        Br = max(0.,min(1.,wp['airlight'][0]*coeff))
        Bg = max(0.,min(1.,wp['airlight'][1]*coeff))
        Bb = max(0.,min(1.,wp['airlight'][2]*coeff))
        return torch.cat([img[:,0:1]*t_r+Br*(1-t_r),
                          img[:,1:2]*t_g+Bg*(1-t_g),
                          img[:,2:3]*t_b+Bb*(1-t_b)],1).clamp(0,1)
    return _deg(pol_0,'0'),_deg(pol_45,'45'),_deg(pol_90,'90'),_deg(pol_135,'135')


def physics_restore(uw_out, params, depth_map=None):
        if params.get('t_is_one', False):
        return uw_out
    B,_,H,W = uw_out.shape
    device  = uw_out.device
    wp      = _WATER_PARAMS[params['water_type']]
    beta    = params['beta']
    if depth_map is not None:
        depth = F.interpolate(depth_map.to(device),(H,W),
                              mode='bilinear',align_corners=True).expand(B,1,H,W)
    else:
        depth = make_depth(H,W,device).expand(B,1,H,W)
    t_r = (torch.exp(-beta*wp['factor_r']*depth)*wp['trans_ratio']).clamp(0.01,1.)
    t_g = (torch.exp(-beta*wp['factor_g']*depth)*wp['trans_ratio']).clamp(0.01,1.)
    t_b = (torch.exp(-beta*wp['factor_b']*depth)*wp['trans_ratio']).clamp(0.01,1.)
    out = torch.zeros_like(uw_out)
    for li, ang in enumerate(['0','45','90','135']):
        rad   = _ANGLES_RAD[ang]
        coeff = 0.5*(1.+_POL_DEGREE*math.cos(2.*(rad-_POL_ANGLE)))
        Br = max(0.,min(1.,wp['airlight'][0]*coeff))
        Bg = max(0.,min(1.,wp['airlight'][1]*coeff))
        Bb = max(0.,min(1.,wp['airlight'][2]*coeff))
        out[:,0*4+li:0*4+li+1] = ((uw_out[:,0*4+li:0*4+li+1]-Br*(1-t_r))/t_r).clamp(0,1)
        out[:,1*4+li:1*4+li+1] = ((uw_out[:,1*4+li:1*4+li+1]-Bg*(1-t_g))/t_g).clamp(0,1)
        out[:,2*4+li:2*4+li+1] = ((uw_out[:,2*4+li:2*4+li+1]-Bb*(1-t_b))/t_b).clamp(0,1)
    return out


def make_12ch(p0, p45, p90, p135):
    B,_,H,W = p0.shape
    out = torch.zeros(B,12,H,W,dtype=p0.dtype,device=p0.device)
    for ci in range(3):
        out[:,ci*4+0:ci*4+1] = p0[:,  ci:ci+1]
        out[:,ci*4+1:ci*4+2] = p45[:, ci:ci+1]
        out[:,ci*4+2:ci*4+3] = p90[:, ci:ci+1]
        out[:,ci*4+3:ci*4+4] = p135[:,ci:ci+1]
    return out


def build_rgb_input(p0, p45, p90, p135):
    """RGB branch input: intensity-only 12ch map, repeated over four angle slots."""
    avg = ((p0 + p45 + p90 + p135) / 4.0).clamp(0, 1)
    return make_12ch(avg, avg, avg, avg)


def build_polar_input_from_12ch(mosaic_12ch):
   
    _, _, H, W = mosaic_12ch.shape
    x_pool = F.max_pool2d(mosaic_12ch, kernel_size=2, stride=1, padding=1)
    x_pool = x_pool[:, :, :H, :W]
    x_down = F.interpolate(x_pool, scale_factor=0.5,
                           mode='bilinear', align_corners=False)
    return F.interpolate(x_down, size=(H, W),
                         mode='bilinear', align_corners=False).clamp(0, 1)


def extract_rgb_0deg(t12):
  
    return torch.cat([t12[:,0:1], t12[:,4:5], t12[:,8:9]], dim=1)



def _load_depth_mat(mat_path):
    depth_np = None
    if _HAS_SCIPY:
        try:
            depth_np = np.array(sio.loadmat(mat_path)['depth_normalized'], dtype=np.float32)
        except Exception:
            pass
    if depth_np is None and _HAS_H5PY:
        try:
            with _h5py.File(mat_path,'r') as f:
                depth_np = np.array(f['depth_normalized'], dtype=np.float32)
                if depth_np.ndim == 2: depth_np = depth_np.T
        except Exception:
            pass
    if depth_np is None:
        return None
    d_min, d_max = depth_np.min(), depth_np.max()
    return (depth_np-d_min)/(d_max-d_min+1e-8)


class PolarDataset(Dataset):
    def __init__(self, image_dir, sample_names=None, transform=None, depth_mat_dir=None):
        self.image_dir     = Path(image_dir)
        self.transform     = transform
        self.depth_mat_dir = Path(depth_mat_dir) if depth_mat_dir else None
        self.samples       = []
        all_dirs = sorted([d for d in self.image_dir.iterdir() if d.is_dir()])
        if sample_names is not None:
            name_set = set(sample_names)
            all_dirs = [d for d in all_dirs if d.name in name_set]
        _rules = [
            {'pol_0':'pol_0.png','pol_45':'pol_45.png','pol_90':'pol_90.png','pol_135':'pol_135.png'},
            {'pol_0':'pol_0.jpg','pol_45':'pol_45.jpg','pol_90':'pol_90.jpg','pol_135':'pol_135.jpg'},
            {'pol_0':'0.bmp','pol_45':'45.bmp','pol_90':'90.bmp','pol_135':'135.bmp'},
            {'pol_0':'0.png','pol_45':'45.png','pol_90':'90.png','pol_135':'135.png'},
        ]
        for d in all_dirs:
            for rule in _rules:
                paths = {k: d/v for k,v in rule.items()}
                if all(p.exists() for p in paths.values()):
                    paths['name'] = d.name
                    self.samples.append(paths)
                    break
        if not self.samples:
            raise ValueError(f"no {image_dir} ")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s    = self.samples[idx]
        imgs = [Image.open(s[k]).convert('RGB') for k in ('pol_0','pol_45','pol_90','pol_135')]
        if self.transform:
            imgs = [self.transform(im) for im in imgs]
        depth_tensor = None
        if self.depth_mat_dir:
            mp = self.depth_mat_dir / f"{s['name']}_depth.mat"
            if mp.exists():
                d = _load_depth_mat(str(mp))
                if d is not None:
                    depth_tensor = torch.from_numpy(d).unsqueeze(0).unsqueeze(0)
        return imgs[0], imgs[1], imgs[2], imgs[3], s['name'], depth_tensor


def collate_fn(batch):
    p0   = torch.stack([b[0] for b in batch])
    p45  = torch.stack([b[1] for b in batch])
    p90  = torch.stack([b[2] for b in batch])
    p135 = torch.stack([b[3] for b in batch])
    names  = [b[4] for b in batch]
    depths = [b[5] for b in batch]
    if all(d is None for d in depths):
        depth_batch = None
    else:
        ref = next(d for d in depths if d is not None)
        _,_,H,W = ref.shape
        depths = [d if d is not None else torch.zeros(1,1,H,W) for d in depths]
        depth_batch = torch.cat(depths, 0)
    return p0, p45, p90, p135, names, depth_batch


def split_dataset(image_dir, n_train=90, n_test=15):
    all_dirs = sorted([d.name for d in Path(image_dir).iterdir() if d.is_dir()])
    total = len(all_dirs)
    if total < n_train + n_test:
        n_train = total - n_test
    train_names = all_dirs[:n_train]
    test_names  = all_dirs[n_train:n_train+n_test]
   
    return train_names, test_names



class Trainer:
    def __init__(self, config):
        self.config   = config
        self.device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pipeline = PolarizationMosaicPipeline()

        print(f"\n{'='*55}")
        

        self.model = DualUCNN(
            rgb_base_ch   = config.get('rgb_base_ch',   32),
            polar_base_ch = config.get('polar_base_ch', 32),
            fusion_mid_ch = config.get('fusion_mid_ch', 32),
        ).to(self.device)

        total = sum(p.numel() for p in self.model.parameters())
        n_rgb = sum(p.numel() for p in self.model.rgb_params())
        n_pol = sum(p.numel() for p in self.model.polar_params())
        n_fus = sum(p.numel() for p in self.model.fusion_params())
      
        lr = config.get('lr', 2e-4)
        wd = config.get('weight_decay', 0.0)
        self.optimizer = optim.AdamW([
            {'params': self.model.rgb_params(),    'lr': lr},
            {'params': self.model.polar_params(),  'lr': config.get('polar_lr', lr)},
            {'params': self.model.fusion_params(), 'lr': config.get('fusion_lr', lr)},
        ], lr=lr, weight_decay=wd)
        warmup = config.get('warmup_epochs', 3)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['num_epochs'] - warmup,
            eta_min=lr * 0.01,
        )
        self.warmup_epochs = warmup

                self.w_rgb    = config.get('w_rgb',    0.4)
        self.w_polar_img = config.get('w_polar_img', 0.4)
        self.w_fused  = config.get('w_fused',  1.0)
        self.w_stokes = config.get('w_stokes', 0.2)
        self.w_ssim   = config.get('w_ssim',   0.2)
        self.w_air    = config.get('w_air',    0.3)          self.ssim_fn  = SSIMLoss(window_size=11).to(self.device)

       
        self.output_dir  = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir/'checkpoints').mkdir(exist_ok=True)
        self.samples_dir = Path(config.get('samples_dir',
                                           str(self.output_dir/'samples')))
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self.nan_stats_path = self.output_dir/'nan_stats.jsonl'
        self.nan_loss_skip_total = 0
        self.nan_grad_skip_total = 0

        self.current_epoch = 0
        self.best_psnr_fused = -float('inf')

        self._setup_data()
       

    def _setup_data(self):
        tf = transforms.Compose([
            transforms.Resize((self.config['image_size'], self.config['image_size'])),
            transforms.ToTensor(),
        ])
        train_names, test_names = split_dataset(
            self.config['data_dir'],
            n_train=self.config.get('n_train', 90),
            n_test =self.config.get('n_test',  15),
        )
        with open(self.output_dir/'test_sample_names.json','w') as f:
            json.dump(test_names, f, indent=2)
        train_ds = PolarDataset(
            self.config['data_dir'], train_names, tf,
            self.config.get('depth_mat_dir'),
        )
        self.train_loader = DataLoader(
            train_ds, batch_size=self.config['batch_size'],
            shuffle=True, num_workers=self.config.get('num_workers', 0),
            collate_fn=collate_fn, drop_last=True,
        )

    def save(self, epoch, is_best=False):
        ckpt = {
            'epoch':      epoch,
            'model':      self.model.state_dict(),
            'optimizer':  self.optimizer.state_dict(),
            'scheduler':  self.scheduler.state_dict(),
            'best_psnr':  self.best_psnr_fused,
        }
        torch.save(ckpt, self.output_dir/'checkpoints'/'latest.pth')
        if is_best:
            torch.save(ckpt, self.output_dir/'checkpoints'/'best.pth')
        if (epoch+1) % self.config.get('save_freq', 10) == 0:
            torch.save(ckpt, self.output_dir/'checkpoints'/f'epoch_{epoch+1:03d}.pth')

    def train_one_epoch(self, epoch):
        self.model.train()
        log = []
        nan_loss_skip = 0
        nan_grad_skip = 0
        pbar = tqdm(self.train_loader,
                    desc=f'Ep{epoch+1}/{self.config["num_epochs"]}')

        for bidx, (pol_0, pol_45, pol_90, pol_135, names, depth_batch) in enumerate(pbar):
            pol_0   = pol_0.to(self.device)
            pol_45  = pol_45.to(self.device)
            pol_90  = pol_90.to(self.device)
            pol_135 = pol_135.to(self.device)
            if depth_batch is not None:
                depth_batch = depth_batch.to(self.device)

                        params = get_random_params()
            d0,d45,d90,d135 = degrade_batch(
                pol_0,pol_45,pol_90,pol_135,params,depth_batch)

                        clean_target   = make_12ch(pol_0, pol_45, pol_90, pol_135).clamp(0,1)
            degrade_target = make_12ch(d0, d45, d90, d135)

                        _, mosaic_12ch, _ = self.pipeline(d0, d45, d90, d135)

                        rgb_input = mosaic_12ch
            polar_input = build_polar_input_from_12ch(mosaic_12ch)
            rgb_out, polar_out, fused_out = self.model(rgb_input, polar_input)

                                    zero_loss = fused_out.new_zeros(())
            loss_rgb = charb(rgb_out, degrade_target) if self.w_rgb > 0 else zero_loss
            loss_polar_img = charb(polar_out, degrade_target) if self.w_polar_img > 0 else zero_loss
            loss_fused = charb(fused_out, degrade_target) if self.w_fused > 0 else zero_loss
            loss_stokes = compute_polar_stokes_loss(polar_out, degrade_target) if self.w_stokes > 0 else zero_loss
            loss_ssim = self.ssim_fn(fused_out, degrade_target) if self.w_ssim > 0 else zero_loss

                        if self.w_air > 0:
                air_out_grad = physics_restore(fused_out.clamp(0,1), params, depth_batch)
                loss_air = charb(air_out_grad, clean_target)
            else:
                with torch.no_grad():
                    air_out_grad = physics_restore(fused_out.detach().clamp(0,1), params, depth_batch)
                loss_air = zero_loss

            total_loss = (self.w_rgb    * loss_rgb
                        + self.w_polar_img * loss_polar_img
                        + self.w_fused  * loss_fused
                        + self.w_stokes * loss_stokes
                        + self.w_ssim   * loss_ssim
                        + self.w_air    * loss_air)

            if not torch.isfinite(total_loss):
                nan_loss_skip += 1
                self.nan_loss_skip_total += 1
                print(f"[WARN] skip non-finite loss batch: "
                      f"wt={params['water_type']} beta={params['beta']} "
                      f"rgb={loss_rgb.item()} polimg={loss_polar_img.item()} "
                      f"fused={loss_fused.item()} stokes={loss_stokes.item()} "
                      f"ssim={loss_ssim.item()} air={loss_air.item()}")
                self.optimizer.zero_grad(set_to_none=True)
                continue

            self.optimizer.zero_grad()
            total_loss.backward()
            if self.config.get('grad_clip', 1.0) > 0:
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(),
                                                     self.config['grad_clip'])
                if not torch.isfinite(grad_norm):
                    nan_grad_skip += 1
                    self.nan_grad_skip_total += 1
                    print(f"[WARN] skip non-finite gradient batch: "
                          f"wt={params['water_type']} beta={params['beta']} grad_norm={grad_norm}")
                    self.optimizer.zero_grad(set_to_none=True)
                    continue
            self.optimizer.step()

                        with torch.no_grad():
                air_out = air_out_grad.detach()
                psnr_fused = 10*math.log10(
                    1./(torch.mean((fused_out-degrade_target)**2).item()+1e-8))
                psnr_air = 10*math.log10(
                    1./(torch.mean((air_out-clean_target)**2).item()+1e-8))

            log.append({
                'loss_rgb':    loss_rgb.item(),
                'loss_polar_img': loss_polar_img.item(),
                'loss_fused':  loss_fused.item(),
                'loss_stokes': loss_stokes.item(),
                'loss_ssim':   loss_ssim.item(),
                'loss_air':    loss_air.item(),
                'loss_total':  total_loss.item(),
                'psnr_fused':  psnr_fused,
                'psnr_air':    psnr_air,
            })
            pbar.set_postfix({
                'loss':      f"{total_loss.item():.4f}",
                'PSNR_fused':f"{psnr_fused:.1f}",
                'PSNR_air':  f"{psnr_air:.1f}",
                'wt':        params['water_type'][:3],
                'β':         f"{params['beta']:.1f}",
                'nan':       f"{nan_loss_skip + nan_grad_skip}",
            })

                        if bidx % self.config.get('sample_freq', 10) == 0:
                n = min(2, pol_0.size(0))
                with torch.no_grad():
                                        def row(t12):
                        return extract_rgb_0deg(t12[:n]).clamp(0,1)

                    rows = [
                        row(clean_target),                                                  row(degrade_target),                                                mosaic_12ch[:n,0:1].expand(-1,3,-1,-1).clamp(0,1),                          row(fused_out),                                                     row(air_out),                                                   ]
                    grid = []
                    for r in rows:
                        for bi in range(r.size(0)):
                            grid.append(r[bi:bi+1])

                try:
                    vutils.save_image(
                        torch.cat(grid, 0),
                        str(self.samples_dir /
                            f'ep{epoch+1:03d}_b{bidx:04d}'
                            f'_{params["water_type"]}_b{params["beta"]:.1f}.png'),
                        nrow=n, normalize=False, padding=2, pad_value=0.15,
                    )
                except OSError as e:
                    print(f"[WARN] sample image save failed, skip this sample: {e}")

        if not log:
            keys = ['loss_rgb','loss_polar_img','loss_fused','loss_stokes','loss_ssim','loss_air','loss_total','psnr_fused','psnr_air']
            avg = {k: float('nan') for k in keys}
        else:
            avg = {k: float(np.mean([l[k] for l in log])) for k in log[0]}
        avg['nan_loss_skip'] = nan_loss_skip
        avg['nan_grad_skip'] = nan_grad_skip
        avg['nan_skip_total'] = nan_loss_skip + nan_grad_skip
        avg['nan_loss_skip_total'] = self.nan_loss_skip_total
        avg['nan_grad_skip_total'] = self.nan_grad_skip_total
        avg['nan_skip_total_all'] = self.nan_loss_skip_total + self.nan_grad_skip_total
        return avg

    def train(self):
        if self.current_epoch >= self.config['num_epochs']:
           
            return
      
        for epoch in range(self.current_epoch, self.config['num_epochs']):
            avg = self.train_one_epoch(epoch)

            is_best = avg['psnr_fused'] > self.best_psnr_fused
            if is_best:
                self.best_psnr_fused = avg['psnr_fused']

            print(f"Ep{epoch+1:03d}  "
                  f"rgb={avg['loss_rgb']:.4f}  polimg={avg['loss_polar_img']:.4f}  "
                  f"stokes={avg['loss_stokes']:.4f}  "
                  f"ssim={avg['loss_ssim']:.4f}  total={avg['loss_total']:.4f}  "
                  f"PSNR_fused={avg['psnr_fused']:.2f}  PSNR_air={avg['psnr_air']:.2f}  "
                  f"nan_skip={avg['nan_skip_total']} (loss={avg['nan_loss_skip']}, grad={avg['nan_grad_skip']})  "
                  f"nan_total={avg['nan_skip_total_all']}"
                  f"{'  ★best' if is_best else ''}")

            with open(self.nan_stats_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({
                    'epoch': epoch + 1,
                    'nan_loss_skip': avg['nan_loss_skip'],
                    'nan_grad_skip': avg['nan_grad_skip'],
                    'nan_skip_total': avg['nan_skip_total'],
                    'nan_loss_skip_total': avg['nan_loss_skip_total'],
                    'nan_grad_skip_total': avg['nan_grad_skip_total'],
                    'nan_skip_total_all': avg['nan_skip_total_all'],
                }, ensure_ascii=False) + '\n')

            self.save(epoch, is_best)

            if epoch >= self.warmup_epochs:
                self.scheduler.step()

       


def main():
    config = {
                'data_dir':      './data',
        'image_size':    512,           'depth_mat_dir': './data',
        'n_train':       90,
        'n_test':        15,

                'rgb_base_ch':   32,
        'polar_base_ch': 32,
        'fusion_mid_ch': 32,

                'num_epochs':    200,
        'batch_size':    2,
        'lr':            2e-4,
        'polar_lr':      2e-4,
        'fusion_lr':     2e-4,
        'weight_decay':  0.0,
        'grad_clip':     1.0,
        'warmup_epochs': 3,

                'w_rgb':    0.4,           'w_polar_img': 0.4,          'w_fused':  1.0,           'w_stokes': 0.2,           'w_ssim':   0.2,           'w_air':    0.3,   
                'num_workers':  4,
        'output_dir':   './data',
        'samples_dir':  './outputs/samples',
        'save_freq':    10,
        'sample_freq':  50,
        'resume':       './outputs/checkpoints/latest.pth',
    }

    trainer = Trainer(config)

    resume = config.get('resume')
    if resume:
        rp = Path(resume)
        if rp.exists():
            print(f" go on: {rp}")
            ckpt = torch.load(rp, map_location=trainer.device, weights_only=False)
            try:
                trainer.model.load_state_dict(ckpt['model'])
                trainer.optimizer.load_state_dict(ckpt['optimizer'])
                trainer.scheduler.load_state_dict(ckpt['scheduler'])
                trainer.current_epoch = ckpt['epoch'] + 1
                trainer.best_psnr_fused = ckpt.get('best_psnr', -float('inf'))
                print(f"  {trainer.current_epoch+1}go on")
            except RuntimeError as e:
                print(f" checkpoint no: {e}")
        else:
            print("no checkpoint, go on")

    trainer.train()





if __name__ == "__main__":
    main()
