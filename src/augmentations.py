import torch
from torchvision import transforms


class SaltPepperNoise:
    def __init__(self, amount=0.05, ratio=0.5):
        self.amount = amount
        self.ratio = ratio

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        noisy = img.clone()
        C, H, W = img.shape
        rand = torch.rand(H, W, device=img.device)
        noisy[:, rand < self.amount * (1 - self.ratio)] = 0.0
        noisy[:, rand > 1 - self.amount * self.ratio]   = 1.0
        return noisy.clamp(0, 1)


# according to this paper, we should use small patches and large number of them
# https://arxiv.org/pdf/2208.07220
# some MRI brain papers suggest 75% cutout, 
# that is 432 patches of size 4x4 for 96x96 images in our case
# https://papers.miccai.org/miccai-2024/paper/2724_paper.pdf
# USE 288 patches (50%) AND 432 patches (75%)
class PatchCutout:
    def __init__(self, patch_size=4, n_patches=432, snp_fillup=False, p_fillup=False):
        self.patch_size = patch_size
        self.n_patches = n_patches
        self.p_fillup = p_fillup
        self.snp_fillup = snp_fillup

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        C, H, W = img.shape
        out = img.clone()
        if self.patch_size > H or self.patch_size > W:
            return out
        for _ in range(self.n_patches):
            y = torch.randint(0, H - self.patch_size + 1, (1,)).item()
            x = torch.randint(0, W - self.patch_size + 1, (1,)).item()
            if self.snp_fillup:
                patch = out[:, y:y + self.patch_size, x:x + self.patch_size]
                out[:, y:y + self.patch_size, x:x + self.patch_size] = SaltPepperNoise(amount=0.05, ratio=0.5)(patch)
            elif self.p_fillup:
                patch = out[:, y:y + self.patch_size, x:x + self.patch_size]
                out[:, y:y + self.patch_size, x:x + self.patch_size] = SaltPepperNoise(amount=0.05, ratio=0)(patch)
            else:
                out[:, y:y + self.patch_size, x:x + self.patch_size] = 0.0
        return out


# same as PatchCutout, but instead of dropping patches at random pixel positions,
# the image is split into a non-overlapping grid of patch_size x patch_size cells
# (e.g. 96x96 with patch_size=16 -> (96/16)*(96/16) = 36 cells) and n_patches of
# those grid cells are selected at random to be dropped.
class GridPatchCutout:
    def __init__(self, patch_size=16, n_patches=4, snp_fillup=False, p_fillup=False):
        self.patch_size = patch_size
        self.n_patches = n_patches
        self.p_fillup = p_fillup
        self.snp_fillup = snp_fillup

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        C, H, W = img.shape
        out = img.clone()
        if self.patch_size > H or self.patch_size > W:
            return out

        n_rows = H // self.patch_size
        n_cols = W // self.patch_size
        n_cells = n_rows * n_cols

        n_drop = min(self.n_patches, n_cells)
        cells = torch.randperm(n_cells)[:n_drop]

        for cell in cells.tolist():
            row = cell // n_cols
            col = cell % n_cols
            y = row * self.patch_size
            x = col * self.patch_size
            if self.snp_fillup:
                patch = out[:, y:y + self.patch_size, x:x + self.patch_size]
                out[:, y:y + self.patch_size, x:x + self.patch_size] = SaltPepperNoise(amount=0.05, ratio=0.5)(patch)
            elif self.p_fillup:
                patch = out[:, y:y + self.patch_size, x:x + self.patch_size]
                out[:, y:y + self.patch_size, x:x + self.patch_size] = SaltPepperNoise(amount=0.05, ratio=0)(patch)
            else:
                out[:, y:y + self.patch_size, x:x + self.patch_size] = 0.0
        return out


# same noise for all channels, should be grey-scale by default
class GaussianNoise_GreyScale:
    def __init__(self, variance=0.05):
        self.std = variance ** 0.5

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        _, H, W = img.shape
        noise = torch.randn(1, H, W, device=img.device).expand_as(img)
        return (img + self.std * noise).clamp(0, 1)


# allowing full rgb noise
class GaussianNoise_FullRGB:
    def __init__(self, variance=0.05):
        self.std = variance ** 0.5

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return (img + self.std * torch.randn_like(img)).clamp(0, 1)



# scales each pixel by a random factor drawn from N(1, variance)
# should be grey-scale by default
class MultiplicativeGaussianNoise:
    def __init__(self, variance=0.05):
        self.std = variance ** 0.5

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        _, H, W = img.shape
        noise = torch.randn(1, H, W, device=img.device).expand_as(img)
        return (img * (1 + self.std * noise)).clamp(0, 1)




AUGMENTATIONS = {
    "baseline":                      lambda: transforms.Lambda(lambda x: x),
    "snp_5%(default)":               lambda: SaltPepperNoise(amount=0.05, ratio=0.5),

    # random patch
    # "patch_16x16_1":                   lambda: PatchCutout(patch_size=16, n_patches=1),
    "patch_16x16_9":                   lambda: PatchCutout(patch_size=16, n_patches=9), # max 25 %

    "gaussian_blur":                 lambda: transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 5.0)),
    "gaussian_noise_greyscale":      lambda: GaussianNoise_GreyScale(variance=0.05),

    "patch_16x16_18_snp_fillup":     lambda: PatchCutout(patch_size=16, n_patches=18, snp_fillup=True), # fillup with snp up to 25%

    # GRID ALIGNED
    "grid_patch_16x16_4":            lambda: GridPatchCutout(patch_size=16, n_patches=9),
    "grid_patch_16x16_18_snp_fillup":     lambda: GridPatchCutout(patch_size=16, n_patches=18, snp_fillup=True), # fillup with snp up to 50%
    "grid_patch_16x16_9_snp_fillup":     lambda: GridPatchCutout(patch_size=16, n_patches=9, snp_fillup=True), # fillup with snp up to 25%


    # "snp_1%":                        lambda: SaltPepperNoise(amount=0.01, ratio=0.5),
    # "pepper-only_5%":                lambda: SaltPepperNoise(amount=0.05, ratio=0.0),
    # "salt-only_5%":                  lambda: SaltPepperNoise(amount=0.05, ratio=1.0),

   
    # "patch_4x4_432":                 lambda: PatchCutout(patch_size=4, n_patches=432), # 75%
    # "patch_2x2_1728":                lambda: PatchCutout(patch_size=2, n_patches=1728), # 75% to match number of erased area with default
    # "patch_2x2_432":                 lambda: PatchCutout(patch_size=2, n_patches=432), # 4x smaller = ~19% of total area
    
    # Both corresponds to half the image in a cutout and same old 5% noise added to patches
    # "patch_16x16_18_snp_fillup":     lambda: PatchCutout(patch_size=16, n_patches=18, snp_fillup=True), # fillup with snp
    # "patch_16x16_18_p_fillup":       lambda: PatchCutout(patch_size=16, n_patches=18, p_fillup=True), # fillup with pepper

    # "gaussian_noise_greyscale":      lambda: GaussianNoise_GreyScale(variance=0.05),
    # "gaussian_noise_greyscale_1%":   lambda: GaussianNoise_GreyScale(variance=0.01),

    # not using full rgb now
    # "gaussian_noise_fullrgb":        lambda: GaussianNoise_FullRGB(variance=0.05),

    # "gaussian_noise_mult":           lambda: MultiplicativeGaussianNoise(variance=0.05),
    # "gaussian_blur":                 lambda: transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 5.0)),
    # "random_erasing":                lambda: transforms.RandomErasing(p=1.0, scale=(0.02, 0.1)),
}