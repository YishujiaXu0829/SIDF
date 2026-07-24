"""DoFP polarization mosaic synthesis module."""


import torch
from typing import Tuple, Dict


class PolarizationMosaicSynthesis:
    """
    """

    @staticmethod
    def rgb_to_gray(img: torch.Tensor) -> torch.Tensor:
        
        return 0.2126 * img[:, 0:1] + 0.7152 * img[:, 1:2] + 0.0722 * img[:, 2:3]

    @staticmethod
    def create_dofp_mosaic_single_channel(
            ch0:   torch.Tensor,               ch45:  torch.Tensor,               ch90:  torch.Tensor,               ch135: torch.Tensor        ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        B, _, H, W = ch0.shape
        H = H - (H % 2)
        W = W - (W % 2)
        ch0   = ch0[:,   :, :H, :W]
        ch45  = ch45[:,  :, :H, :W]
        ch90  = ch90[:,  :, :H, :W]
        ch135 = ch135[:, :, :H, :W]

        mosaic = torch.zeros(B, 4, H, W, dtype=ch0.dtype, device=ch0.device)
        mask   = torch.zeros(B, 4, H, W, dtype=ch0.dtype, device=ch0.device)

                        mask[:, 0, 1::2, 1::2] = 1
                mask[:, 1, 0::2, 1::2] = 1
                mask[:, 2, 0::2, 0::2] = 1
                mask[:, 3, 1::2, 0::2] = 1

        mosaic[:, 0] = ch0[:,   0] * mask[:, 0]           mosaic[:, 1] = ch45[:,  0] * mask[:, 1]           mosaic[:, 2] = ch90[:,  0] * mask[:, 2]           mosaic[:, 3] = ch135[:, 0] * mask[:, 3]   
        return mosaic, mask

    @staticmethod
    def create_sparse_mosaic_for_viz(
            pol_0: torch.Tensor, pol_45: torch.Tensor,
            pol_90: torch.Tensor, pol_135: torch.Tensor
    ) -> torch.Tensor:
        
        synth = PolarizationMosaicSynthesis
        B, _, H, W = pol_0.shape
        H = H - (H % 2)
        W = W - (W % 2)
        p0   = pol_0[:,   :, :H, :W]
        p45  = pol_45[:,  :, :H, :W]
        p90  = pol_90[:,  :, :H, :W]
        p135 = pol_135[:, :, :H, :W]

        g0   = synth.rgb_to_gray(p0)            g45  = synth.rgb_to_gray(p45)
        g90  = synth.rgb_to_gray(p90)
        g135 = synth.rgb_to_gray(p135)

        viz = torch.zeros(B, 1, H, W, dtype=p0.dtype, device=p0.device)
        viz[:, :, 1::2, 1::2] = g0[:,   :, 1::2, 1::2]           viz[:, :, 0::2, 1::2] = g45[:,  :, 0::2, 1::2]           viz[:, :, 0::2, 0::2] = g90[:,  :, 0::2, 0::2]           viz[:, :, 1::2, 0::2] = g135[:, :, 1::2, 0::2]   
        return viz.repeat(1, 3, 1, 1)   
    @staticmethod
    def create_network_input_12ch(
            pol_0: torch.Tensor, pol_45: torch.Tensor,
            pol_90: torch.Tensor, pol_135: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
       
        synth = PolarizationMosaicSynthesis
        B, _, H, W = pol_0.shape
        H = H - (H % 2)
        W = W - (W % 2)
        p0   = pol_0[:,   :, :H, :W]
        p45  = pol_45[:,  :, :H, :W]
        p90  = pol_90[:,  :, :H, :W]
        p135 = pol_135[:, :, :H, :W]

        layers = []
        detail = {}
        for ci, cname in enumerate(['R', 'G', 'B']):
            ch0   = p0[:,   ci:ci+1]
            ch45  = p45[:,  ci:ci+1]
            ch90  = p90[:,  ci:ci+1]
            ch135 = p135[:, ci:ci+1]

            mosaic_4, mask_4 = synth.create_dofp_mosaic_single_channel(
                ch0, ch45, ch90, ch135
            )                                        layers.append(mosaic_4)
            detail[f'mosaic4_{cname}'] = mosaic_4
            detail[f'mask_{cname}']    = mask_4

        mosaic_12ch = torch.cat(layers, dim=1)           return mosaic_12ch, detail


class PolarizationMosaicPipeline:
    """
    """

    def __init__(self):
        self.synthesis = PolarizationMosaicSynthesis()

    def __call__(
            self,
            deg_pol_0:   torch.Tensor,
            deg_pol_45:  torch.Tensor,
            deg_pol_90:  torch.Tensor,
            deg_pol_135: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        
        synth = self.synthesis

        mosaic_image = synth.create_sparse_mosaic_for_viz(
            deg_pol_0, deg_pol_45, deg_pol_90, deg_pol_135
        )

        mosaic_stack, detail = synth.create_network_input_12ch(
            deg_pol_0, deg_pol_45, deg_pol_90, deg_pol_135
        )

        return mosaic_image, mosaic_stack, detail


