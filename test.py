"""DualUCNN training script."""


import math, json, csv, datetime, time
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from mosaic import PolarizationMosaicPipeline
from model import DualUCNN

try:
    import scipy.io as sio; _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

try:
    import h5py as _h5py; _HAS_H5PY = True
except ImportError:
    _HAS_H5PY = False



_WATER_PARAMS = {
    'coastal':       {'airlight':[0.35,0.60,0.50],'factor_r':1.4,'factor_g':0.8,'factor_b':1.0,'trans_ratio':0.15},
    'deep_ocean':    {'airlight':[0.20,0.35,0.65],'factor_r':2.0,'factor_g':1.2,'factor_b':0.6,'trans_ratio':0.20},
    'clear_shallow': {'airlight':[0.25,0.50,0.60],'factor_r':1.5,'factor_g':0.9,'factor_b':0.7,'trans_ratio':0.18},
    'turbid':        {'airlight':[0.45,0.50,0.40],'factor_r':1.2,'factor_g':0.9,'factor_b':1.1,'trans_ratio':0.12},
}
_BETA_VALUES = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
_T1_WATER_TYPE = 't1_identity'
_ANGLES_RAD  = {'0':0.0,'45':math.pi/4,'90':math.pi/2,'135':3*math.pi/4}
_POL_DEGREE  = 0.3
_POL_ANGLE   = math.pi / 4.0
_DOLP_GAMMA  = 0.3


def make_depth(H, W, device):
    y = torch.linspace(0,1,H,device=device)
    x = torch.linspace(0,1,W,device=device)
    yy,xx = torch.meshgrid(y,x,indexing='ij')
    d = torch.sqrt((xx-.5)**2+(yy-.5)**2)
    return (d/d.max()).unsqueeze(0).unsqueeze(0)


def degrade_batch(pol_0, pol_45, pol_90, pol_135, params, depth_map=None):
    if params.get('t_is_one', False):
        return pol_0, pol_45, pol_90, pol_135
    B,_,H,W = pol_0.shape; device = pol_0.device
    wp = _WATER_PARAMS[params['water_type']]; beta = params['beta']
    if depth_map is not None:
        depth = F.interpolate(depth_map.to(device),(H,W),mode='bilinear',align_corners=True).expand(B,1,H,W)
    else:
        depth = make_depth(H,W,device).expand(B,1,H,W)
    t_r = (torch.exp(-beta*wp['factor_r']*depth)*wp['trans_ratio']).clamp(0.01,1.)
    t_g = (torch.exp(-beta*wp['factor_g']*depth)*wp['trans_ratio']).clamp(0.01,1.)
    t_b = (torch.exp(-beta*wp['factor_b']*depth)*wp['trans_ratio']).clamp(0.01,1.)
    def _deg(img, ang):
        coeff = 0.5*(1.+_POL_DEGREE*math.cos(2.*(_ANGLES_RAD[ang]-_POL_ANGLE)))
        Br=max(0.,min(1.,wp['airlight'][0]*coeff))
        Bg=max(0.,min(1.,wp['airlight'][1]*coeff))
        Bb=max(0.,min(1.,wp['airlight'][2]*coeff))
        return torch.cat([img[:,0:1]*t_r+Br*(1-t_r),img[:,1:2]*t_g+Bg*(1-t_g),img[:,2:3]*t_b+Bb*(1-t_b)],1).clamp(0,1)
    return _deg(pol_0,'0'),_deg(pol_45,'45'),_deg(pol_90,'90'),_deg(pol_135,'135')


@torch.no_grad()
def physics_restore_pols(uw_0, uw_45, uw_90, uw_135, params, depth_map=None):
    if params.get('t_is_one', False):
        return uw_0, uw_45, uw_90, uw_135
    device=uw_0.device; B,_,H,W=uw_0.shape
    wp=_WATER_PARAMS[params['water_type']]; beta=params['beta']
    if depth_map is not None:
        depth=F.interpolate(depth_map.to(device),(H,W),mode='bilinear',align_corners=True).expand(B,1,H,W)
    else:
        depth=make_depth(H,W,device).expand(B,1,H,W)
    t_r=(torch.exp(-beta*wp['factor_r']*depth)*wp['trans_ratio']).clamp(0.01,1.)
    t_g=(torch.exp(-beta*wp['factor_g']*depth)*wp['trans_ratio']).clamp(0.01,1.)
    t_b=(torch.exp(-beta*wp['factor_b']*depth)*wp['trans_ratio']).clamp(0.01,1.)
    results=[]
    for uw,ang in [(uw_0,'0'),(uw_45,'45'),(uw_90,'90'),(uw_135,'135')]:
        rad=_ANGLES_RAD[ang]; coeff=0.5*(1.+_POL_DEGREE*math.cos(2.*(rad-_POL_ANGLE)))
        Br=max(0.,min(1.,wp['airlight'][0]*coeff))
        Bg=max(0.,min(1.,wp['airlight'][1]*coeff))
        Bb=max(0.,min(1.,wp['airlight'][2]*coeff))
        results.append(torch.cat([
            ((uw[:,0:1]-Br*(1-t_r))/t_r).clamp(0,1),
            ((uw[:,1:2]-Bg*(1-t_g))/t_g).clamp(0,1),
            ((uw[:,2:3]-Bb*(1-t_b))/t_b).clamp(0,1)],1))
    return results


def split_12ch_to_pols(t12):
    return (torch.cat([t12[:,0:1],t12[:,4:5],t12[:,8:9]],1),
            torch.cat([t12[:,1:2],t12[:,5:6],t12[:,9:10]],1),
            torch.cat([t12[:,2:3],t12[:,6:7],t12[:,10:11]],1),
            torch.cat([t12[:,3:4],t12[:,7:8],t12[:,11:12]],1))


def iter_test_params():
    for water_type in _WATER_PARAMS.keys():
        for beta in _BETA_VALUES:
            yield {'water_type': water_type, 'beta': beta, 't_is_one': False}
    yield {'water_type': _T1_WATER_TYPE, 'beta': 0.0, 't_is_one': True}



def make_12ch(p0, p45, p90, p135):
    B, _, H, W = p0.shape
    out = torch.zeros(B, 12, H, W, dtype=p0.dtype, device=p0.device)
    for ci in range(3):
        out[:, ci*4+0:ci*4+1] = p0[:, ci:ci+1]
        out[:, ci*4+1:ci*4+2] = p45[:, ci:ci+1]
        out[:, ci*4+2:ci*4+3] = p90[:, ci:ci+1]
        out[:, ci*4+3:ci*4+4] = p135[:, ci:ci+1]
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

def _t2np(t): return t.squeeze(0).permute(1,2,0).cpu().numpy().astype(np.float64)

def calc_psnr(pred, gt):
    mse = np.mean((pred-gt)**2); return float(10*np.log10(1./(mse+1e-8)))

def _ssim_1ch(p, g, win=11):
    from scipy.ndimage import uniform_filter
    C1,C2=(0.01)**2,(0.03)**2
    mu_p=uniform_filter(p.astype(np.float64),win); mu_g=uniform_filter(g.astype(np.float64),win)
    sig_p=uniform_filter(p*p,win)-mu_p**2; sig_g=uniform_filter(g*g,win)-mu_g**2
    sig_pg=uniform_filter(p*g,win)-mu_p*mu_g
    num=(2*mu_p*mu_g+C1)*(2*sig_pg+C2); den=(mu_p**2+mu_g**2+C1)*(sig_p+sig_g+C2)
    return float(np.mean(num/(den+1e-10)))

def calc_ssim(pred, gt):
    if pred.ndim==3: return float(np.mean([_ssim_1ch(pred[:,:,c],gt[:,:,c]) for c in range(pred.shape[2])]))
    return _ssim_1ch(pred,gt)

def calc_mae(pred, gt): return float(np.mean(np.abs(pred.astype(np.float64)-gt.astype(np.float64))))

def calc_stokes_rgb(pols):
    out={}
    for ch,ci in [('R',0),('G',1),('B',2)]:
        I0=pols['I0'][:,:,ci]; I45=pols['I45'][:,:,ci]
        I90=pols['I90'][:,:,ci]; I135=pols['I135'][:,:,ci]
        S0=I0+I90          S1=I0-I90; S2=I45-I135
        DoP=np.clip(np.sqrt(S1**2+S2**2)/(S0+1e-10),0,1)
        AoP=np.mod(0.5*np.arctan2(S2,S1)*180./np.pi,180)
        for k,v in [('S0',S0),('S1',S1),('S2',S2),('DoP',DoP),('AoP',AoP)]:
            out.setdefault(k,[]).append(v)
    return {k:np.stack(v,axis=2) for k,v in out.items()}


def evaluate_polarization(pred_pols, gt_pols):
    
    res={}
    for name in ['I0','I45','I90','I135']:
        p,g=pred_pols[name],gt_pols[name]
        res[f'{name}_PSNR']=calc_psnr(p,g)
        res[f'{name}_SSIM']=calc_ssim(p,g)
        res[f'{name}_MAE'] =calc_mae(p,g)
    res['avg_PSNR']=float(np.mean([res[f'{n}_PSNR'] for n in ['I0','I45','I90','I135']]))
    res['avg_SSIM']=float(np.mean([res[f'{n}_SSIM'] for n in ['I0','I45','I90','I135']]))
    res['avg_MAE'] =float(np.mean([res[f'{n}_MAE']  for n in ['I0','I45','I90','I135']]))
    stk_pred=calc_stokes_rgb(pred_pols)
    stk_gt  =calc_stokes_rgb(gt_pols)
    for sn in ['S0','S1','S2']:
        sp,sg=stk_pred[sn],stk_gt[sn]
        if sn=='S0':
                        sp_n,sg_n=sp/2.0,sg/2.0
        else:
                                    sp_n=(sp+1.0)/2.0
            sg_n=(sg+1.0)/2.0
        res[f'{sn}_PSNR']=calc_psnr(sp_n,sg_n)
        res[f'{sn}_SSIM']=calc_ssim(sp_n,sg_n)
        res[f'{sn}_MAE'] =calc_mae(sp_n,sg_n)
    DoP_p,DoP_g=stk_pred['DoP'],stk_gt['DoP']
    res['DoP_PSNR']=calc_psnr(DoP_p,DoP_g)
    res['DoP_SSIM']=calc_ssim(DoP_p,DoP_g)
    res['DoP_MAE'] =calc_mae(DoP_p,DoP_g)
    DoP_thresh=0.1; AoP_p,AoP_g=stk_pred['AoP'],stk_gt['AoP']
    mask=(DoP_g>DoP_thresh)&(DoP_p>DoP_thresh)
    aop_err=np.minimum(np.abs(AoP_p-AoP_g),180-np.abs(AoP_p-AoP_g))
    if mask.sum()>0:
        res['AoP_MeanErr']=float(np.mean(aop_err[mask]))
        res['AoP_StdErr'] =float(np.std(aop_err[mask]))
    else:
        res['AoP_MeanErr']=0.0; res['AoP_StdErr']=0.0
    res['AoP_ValidPx']   =int(mask.sum())
    res['AoP_ValidRatio']=float(mask.sum()/mask.size*100)
    res['AoP_Threshold'] =DoP_thresh
    return res





def _load_depth_mat(mat_path):
    depth_np=None
    if _HAS_SCIPY:
        try: depth_np=np.array(sio.loadmat(mat_path)['depth_normalized'],dtype=np.float32)
        except: pass
    if depth_np is None and _HAS_H5PY:
        try:
            with _h5py.File(mat_path,'r') as f:
                depth_np=np.array(f['depth_normalized'],dtype=np.float32)
                if depth_np.ndim==2: depth_np=depth_np.T
        except: pass
    if depth_np is None: return None
    d_min,d_max=depth_np.min(),depth_np.max()
    return (depth_np-d_min)/(d_max-d_min+1e-8)



def test_single(model, pipeline, device, tf, sample_dir, sample_name,
                params, depth_map, out_dir,
                run_python_restore=True):
    wp=_WATER_PARAMS.get(params['water_type'], {
        'airlight':[0.0,0.0,0.0],
        'factor_r':0.0,'factor_g':0.0,'factor_b':0.0,
        'trans_ratio':1.0,
    })
    suffix=f"{sample_name}_{params['water_type']}_beta{params['beta']}"

    _rules=[
        {'pol_0':'pol_0.png','pol_45':'pol_45.png','pol_90':'pol_90.png','pol_135':'pol_135.png'},
        {'pol_0':'pol_0.jpg','pol_45':'pol_45.jpg','pol_90':'pol_90.jpg','pol_135':'pol_135.jpg'},
        {'pol_0':'0.png','pol_45':'45.png','pol_90':'90.png','pol_135':'135.png'},
    ]
    img_paths=None
    for rule in _rules:
        paths={k:Path(sample_dir)/v for k,v in rule.items()}
        if all(p.exists() for p in paths.values()): img_paths=paths; break
    if img_paths is None: return None

    _orig_img=Image.open(img_paths['pol_0']).convert('RGB')
    _orig_H,_orig_W=_orig_img.height,_orig_img.width
    pol_0  =tf(_orig_img).unsqueeze(0).to(device)
    pol_45 =tf(Image.open(img_paths['pol_45']).convert('RGB')).unsqueeze(0).to(device)
    pol_90 =tf(Image.open(img_paths['pol_90']).convert('RGB')).unsqueeze(0).to(device)
    pol_135=tf(Image.open(img_paths['pol_135']).convert('RGB')).unsqueeze(0).to(device)

    with torch.no_grad():
        d0,d45,d90,d135=degrade_batch(pol_0,pol_45,pol_90,pol_135,params,depth_map)
        _,mosaic_12ch,_=pipeline(d0,d45,d90,d135)
        degrade_target=make_12ch(d0,d45,d90,d135)
        rgb_input=mosaic_12ch
        polar_input=build_polar_input_from_12ch(mosaic_12ch)
        _,_,fused_out=model(rgb_input,polar_input)
        fu_0,fu_45,fu_90,fu_135=split_12ch_to_pols(fused_out)

    res_uw=evaluate_polarization(
        {'I0':_t2np(fu_0),'I45':_t2np(fu_45),'I90':_t2np(fu_90),'I135':_t2np(fu_135)},
        {'I0':_t2np(d0),  'I45':_t2np(d45),  'I90':_t2np(d90),  'I135':_t2np(d135)})

    res_air=None
    if run_python_restore:
        with torch.no_grad():
            res_0,res_45,res_90,res_135=physics_restore_pols(fu_0,fu_45,fu_90,fu_135,params,depth_map)
        res_air=evaluate_polarization(
            {'I0':_t2np(res_0),'I45':_t2np(res_45),'I90':_t2np(res_90),'I135':_t2np(res_135)},
            {'I0':_t2np(pol_0),'I45':_t2np(pol_45),'I90':_t2np(pol_90),'I135':_t2np(pol_135)})
    return res_uw, res_air





def _to_float_or_keep(v):
    if isinstance(v, (int, float)):
        return v
    try:
        if v is None or v == '':
            return v
        return float(v)
    except (TypeError, ValueError):
        return v


def _result_key(sample, water_type, beta):
    try:
        beta_key = "{:.6g}".format(float(beta))
    except (TypeError, ValueError):
        beta_key = str(beta)
    return (str(sample), str(water_type), beta_key)


def _row_key(row):
    return _result_key(row.get('sample', ''), row.get('water_type', ''), row.get('beta', ''))


def _load_existing_result_rows(csv_path):
    if not csv_path.exists():
        return []
    rows = []
    try:
        with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('sample') == 'MEAN':
                    continue
                rows.append({k: _to_float_or_keep(v) for k, v in row.items()})
    except Exception as e:
        print("  [WARN] existing CSV read failed, retest this folder: {} ({})".format(csv_path, e))
        return []
    return rows


def _write_result_csv(csv_path, rows):
    if not rows:
        return
    skip_cols = {'sample', 'water_type', 'beta'}
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    num_cols = [k for k in fieldnames if k not in skip_cols]
    mean_row = {'sample': 'MEAN', 'water_type': '', 'beta': ''}
    for col in num_cols:
        vals = []
        for row in rows:
            val = _to_float_or_keep(row.get(col, ''))
            if isinstance(val, (int, float)) and np.isfinite(val):
                vals.append(float(val))
        mean_row[col] = round(float(np.mean(vals)), 6) if vals else ''
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(mean_row)
        writer.writerows(rows)


def _run_test(model, pipeline, device, tf, config, out_dir, test_names):
    data_dir = Path(config['data_dir'])
    depth_dir = Path(config['depth_mat_dir']) if config.get('depth_mat_dir') else None
    out_dir.mkdir(parents=True, exist_ok=True)
    duration_start = time.perf_counter()
    duration_started_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    test_params = list(iter_test_params())
    total = len(test_names) * len(test_params)
    csv_path = out_dir / 'test_results.csv'

    all_rows = _load_existing_result_rows(csv_path)
    done_keys = {_row_key(row) for row in all_rows}
    if all_rows:
        print("  resume test: loaded {} existing rows from {}".format(len(all_rows), csv_path))

    with tqdm(total=total, initial=min(len(done_keys), total), desc=f' {out_dir.name}') as pbar:
        for sample_name in test_names:
            sample_dir = data_dir / sample_name
            depth_map = None
            if depth_dir:
                mp = depth_dir / f"{sample_name}_depth.mat"
                if mp.exists():
                    d = _load_depth_mat(str(mp))
                    if d is not None:
                        depth_map = torch.from_numpy(d).unsqueeze(0).unsqueeze(0).to(device)
            for params in test_params:
                key = _result_key(sample_name, params['water_type'], params['beta'])
                if key in done_keys:
                    continue
                result = test_single(
                    model, pipeline, device, tf, sample_dir, sample_name,
                    params, depth_map, out_dir,
                    run_python_restore=config['run_python_restore'],
                    False=config['False'],
                )
                pbar.update(1)
                if result is None:
                    continue
                res_uw, res_air = result
                row = {
                    'sample': sample_name, 'water_type': params['water_type'], 'beta': params['beta'],
                    'uw_avg_PSNR': round(res_uw['avg_PSNR'], 4), 'uw_avg_SSIM': round(res_uw['avg_SSIM'], 4), 'uw_avg_MAE': round(res_uw['avg_MAE'], 6),
                    'uw_I0_PSNR': round(res_uw['I0_PSNR'], 4), 'uw_I0_SSIM': round(res_uw['I0_SSIM'], 4), 'uw_I0_MAE': round(res_uw['I0_MAE'], 6),
                    'uw_I45_PSNR': round(res_uw['I45_PSNR'], 4), 'uw_I45_SSIM': round(res_uw['I45_SSIM'], 4), 'uw_I45_MAE': round(res_uw['I45_MAE'], 6),
                    'uw_I90_PSNR': round(res_uw['I90_PSNR'], 4), 'uw_I90_SSIM': round(res_uw['I90_SSIM'], 4), 'uw_I90_MAE': round(res_uw['I90_MAE'], 6),
                    'uw_I135_PSNR': round(res_uw['I135_PSNR'], 4), 'uw_I135_SSIM': round(res_uw['I135_SSIM'], 4), 'uw_I135_MAE': round(res_uw['I135_MAE'], 6),
                    'uw_S0_PSNR': round(res_uw['S0_PSNR'], 4), 'uw_S0_SSIM': round(res_uw['S0_SSIM'], 4),
                    'uw_S1_PSNR': round(res_uw['S1_PSNR'], 4), 'uw_S1_SSIM': round(res_uw['S1_SSIM'], 4),
                    'uw_S2_PSNR': round(res_uw['S2_PSNR'], 4), 'uw_S2_SSIM': round(res_uw['S2_SSIM'], 4),
                    'uw_stokes_avg_PSNR': round(float(np.mean([res_uw['S0_PSNR'], res_uw['S1_PSNR'], res_uw['S2_PSNR']])), 4),
                    'uw_stokes_avg_SSIM': round(float(np.mean([res_uw['S0_SSIM'], res_uw['S1_SSIM'], res_uw['S2_SSIM']])), 4),
                    'uw_DoP_PSNR': round(res_uw['DoP_PSNR'], 4), 'uw_DoP_SSIM': round(res_uw['DoP_SSIM'], 4),
                    'uw_AoP_err': round(res_uw['AoP_MeanErr'], 4),
                }
                if res_air is not None:
                    row.update({
                        'air_avg_PSNR': round(res_air['avg_PSNR'], 4), 'air_avg_SSIM': round(res_air['avg_SSIM'], 4), 'air_avg_MAE': round(res_air['avg_MAE'], 6),
                        'air_I0_PSNR': round(res_air['I0_PSNR'], 4), 'air_I0_SSIM': round(res_air['I0_SSIM'], 4), 'air_I0_MAE': round(res_air['I0_MAE'], 6),
                        'air_I45_PSNR': round(res_air['I45_PSNR'], 4), 'air_I45_SSIM': round(res_air['I45_SSIM'], 4), 'air_I45_MAE': round(res_air['I45_MAE'], 6),
                        'air_I90_PSNR': round(res_air['I90_PSNR'], 4), 'air_I90_SSIM': round(res_air['I90_SSIM'], 4), 'air_I90_MAE': round(res_air['I90_MAE'], 6),
                        'air_I135_PSNR': round(res_air['I135_PSNR'], 4), 'air_I135_SSIM': round(res_air['I135_SSIM'], 4), 'air_I135_MAE': round(res_air['I135_MAE'], 6),
                        'air_S0_PSNR': round(res_air['S0_PSNR'], 4), 'air_S0_SSIM': round(res_air['S0_SSIM'], 4),
                        'air_S1_PSNR': round(res_air['S1_PSNR'], 4), 'air_S1_SSIM': round(res_air['S1_SSIM'], 4),
                        'air_S2_PSNR': round(res_air['S2_PSNR'], 4), 'air_S2_SSIM': round(res_air['S2_SSIM'], 4),
                        'air_stokes_avg_PSNR': round(float(np.mean([res_air['S0_PSNR'], res_air['S1_PSNR'], res_air['S2_PSNR']])), 4),
                        'air_stokes_avg_SSIM': round(float(np.mean([res_air['S0_SSIM'], res_air['S1_SSIM'], res_air['S2_SSIM']])), 4),
                        'air_DoP_PSNR': round(res_air['DoP_PSNR'], 4), 'air_DoP_SSIM': round(res_air['DoP_SSIM'], 4),
                        'air_AoP_err': round(res_air['AoP_MeanErr'], 4),
                    })
                all_rows.append(row)
                done_keys.add(key)
                _write_result_csv(csv_path, all_rows)

    if not all_rows:
        return all_rows

    _write_result_csv(csv_path, all_rows)
    print(f"\n{'='*55}")
    print(f" {len(all_rows)} / {total}  t=1")
    print(f"  avg_PSNR={np.mean([float(r['uw_avg_PSNR']) for r in all_rows]):.3f} dB"
          f"  avg_SSIM={np.mean([float(r['uw_avg_SSIM']) for r in all_rows]):.4f}"
          f"  AoP_err={np.mean([float(r['uw_AoP_err']) for r in all_rows]):.3f}")
    if config['run_python_restore'] and 'air_avg_PSNR' in all_rows[0]:
        print(f"  avg_PSNR={np.mean([float(r['air_avg_PSNR']) for r in all_rows if 'air_avg_PSNR' in r]):.3f} dB"
              f"  AoP_err={np.mean([float(r['air_AoP_err']) for r in all_rows if 'air_AoP_err' in r]):.3f}")
    print(f"  CSV  {csv_path}")
        
    else:
        print(" finish")    print(f"{'='*55}\n")
    return all_rows


