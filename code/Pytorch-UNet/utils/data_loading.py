import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm

_RESAMPLING = getattr(Image, "Resampling", Image)
RESAMPLE_NEAREST = getattr(_RESAMPLING, "NEAREST")
RESAMPLE_BICUBIC = getattr(_RESAMPLING, "BICUBIC")


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=RESAMPLE_NEAREST if is_mask else RESAMPLE_BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                # Drop alpha channel if present (e.g. RGBA)
                if img.shape[2] == 4:
                    img = img[:, :, :3]
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask')


class LineWithEndpointDataset(BasicDataset):
    def __init__(self, images_dir, text_dir, mask_dir, endpoint_dir, scale=1.0, mask_suffix=''):
        super().__init__(images_dir, mask_dir, scale, mask_suffix)
        self.endpoint_dir = Path(endpoint_dir)

    def __getitem__(self, idx):
        base = super().__getitem__(idx)  # 拿到原来的 image 和 mask
        name = self.ids[idx]

        endpoint_file = list(self.endpoint_dir.glob(name + '.*'))
        assert len(endpoint_file) == 1, f'Either no endpoint or multiple endpoints found for the ID {name}: {endpoint_file}'

        # 加载端点图，转换为 numpy float32，归一化到 [0, 1]
        endpoint_img = load_image(endpoint_file[0])  # PIL.Image
        endpoint_np = np.array(endpoint_img).astype(np.float32) / 255.0  # (H, W)

        # 缩放（如果需要）
        if self.scale != 1:
            new_size = (int(endpoint_np.shape[1] * self.scale), int(endpoint_np.shape[0] * self.scale))
            import cv2
            endpoint_np = cv2.resize(endpoint_np, new_size, interpolation=cv2.INTER_LINEAR)

        # 添加 channel 维度，变成 (1, H, W)
        endpoint_np = np.expand_dims(endpoint_np, axis=0)

        # 转为 float tensor
        base['endpoint'] = torch.from_numpy(endpoint_np).float().contiguous()

        return base
    

class LineWithEndpoint4BandDataset(BasicDataset):
    """
    输出：
        'image': 四波段 tensor (R,G,B,TextMask), shape (4,H,W)
        'mask': 原始 mask tensor, shape (H,W)
        'endpoint': 热力图 tensor, shape (1,H,W)
    """
    def __init__(self, images_dir, text_mask_dir, mask_dir, endpoint_dir, scale=1.0, mask_suffix=''):
        super().__init__(images_dir, mask_dir, scale, mask_suffix)
        self.text_mask_dir = Path(text_mask_dir)
        self.endpoint_dir = Path(endpoint_dir)

    def __getitem__(self, idx):
        # 1. 原始 RGB + mask
        base = super().__getitem__(idx)
        name = self.ids[idx]

        # 2. 加载文字掩膜
        text_mask_file = list(self.text_mask_dir.glob(name + '.*'))
        assert len(text_mask_file) == 1, f"No text mask or multiple found for {name}: {text_mask_file}"
        text_mask_img = load_image(text_mask_file[0])  # PIL.Image
        text_mask_np = np.array(text_mask_img).astype(np.float32)

        # 归一化到 0~1
        if text_mask_np.max() > 1.0:
            text_mask_np /= 255.0

        # 缩放文字掩膜
        if self.scale != 1:
            import cv2
            new_size = (int(text_mask_np.shape[1] * self.scale), int(text_mask_np.shape[0] * self.scale))
            text_mask_np = cv2.resize(text_mask_np, new_size, interpolation=cv2.INTER_LINEAR)

        # 转为 (1,H,W)
        if text_mask_np.ndim == 2:
            text_mask_np = np.expand_dims(text_mask_np, axis=0)
        else:
            text_mask_np = text_mask_np.transpose((2,0,1))[:1,:,:]  # 保留第一个通道

        # 3. 拼接四波段图像
        img = base['image']  # (3,H,W)
        img_4band = torch.cat([img, torch.from_numpy(text_mask_np).float()], dim=0)
        base['image'] = img_4band.contiguous()

        # 4. 加载端点热力图
        endpoint_file = list(self.endpoint_dir.glob(name + '.*'))
        assert len(endpoint_file) == 1, f"No endpoint or multiple endpoints for {name}: {endpoint_file}"
        endpoint_img = load_image(endpoint_file[0])
        endpoint_np = np.array(endpoint_img).astype(np.float32) / 255.0  # (H,W)

        # 二值化：大于0都设为1
        endpoint_np = (endpoint_np > 0).astype(np.float32)

        # 缩放
        if self.scale != 1:
            new_size = (img_4band.shape[2], img_4band.shape[1])  # 对齐 H,W
            import cv2
            endpoint_np = cv2.resize(endpoint_np, new_size, interpolation=cv2.INTER_NEAREST)

        # 转为 (1,H,W)
        if endpoint_np.ndim == 2:
            endpoint_np = np.expand_dims(endpoint_np, axis=0)

        base['point'] = torch.from_numpy(endpoint_np).float().contiguous()

        return base



