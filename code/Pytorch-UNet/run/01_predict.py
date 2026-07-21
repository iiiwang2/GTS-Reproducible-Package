import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import logging
import os
import cv2

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from utils.data_loading import BasicDataset
from unet.unet_model import UNet
from unet.dense_unet import DenseUNet
from unet.unetDVH import UnetDVHLinear
from unet.unetDVV import UnetDVVLinear

from utils.utils import plot_img_and_mask


"""Predict line/border masks for geological time scale images."""

_RESAMPLING = getattr(Image, "Resampling", Image)
RESAMPLE_BILINEAR = getattr(_RESAMPLING, "BILINEAR")


def enhance_lines(img: Image.Image) -> Image.Image:
    """Enhance thin line structures before segmentation."""
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8) 
    dilated = cv2.dilate(binary, kernel, iterations=1)
    dilated_rgb = cv2.cvtColor(255 - dilated, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(dilated_rgb)


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    """Run whole-image UNet prediction and return a binary mask."""
    net.eval()

    # 线条增强
    # full_img = enhance_lines(full_img)

    # Resize 到 512×512
    resized_img = full_img.resize((512, 512), RESAMPLE_BILINEAR)

    img = torch.from_numpy(BasicDataset.preprocess(None, resized_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        # 将输出 resize 回原图块大小
        output = F.interpolate(output, size=(full_img.size[1], full_img.size[0]), mode='bilinear')

        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    mask_np = mask[0].long().squeeze().numpy()

    return mask_np



def get_args():
    """Parse command-line arguments for line-mask prediction."""
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', type=str, default='./results', help='Directory to save output images', required=True)
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    
    return parser.parse_args()


def mask_to_image(mask: np.ndarray, mask_values):
    """Convert a predicted mask array to a PIL image."""
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)

def zero_border(mask: np.ndarray, border_size=5) -> np.ndarray:
    """Set a thin image border to zero to suppress edge artifacts."""
    mask[:border_size, :] = 0
    mask[-border_size:, :] = 0
    mask[:, :border_size] = 0
    mask[:, -border_size:] = 0
    return mask


def sliding_window_predict(net, img: Image.Image, device, scale_factor=1, out_threshold=0.5,
                           crop_size=(512, 512), stride=(341, 341)):
    """Predict a large image by tiled sliding-window inference."""
    width, height = img.size
    mask_pred = np.zeros((height, width), dtype=np.uint8)

    crop_h, crop_w = crop_size
    stride_h, stride_w = stride

    for top in range(0, height, stride_h):
        if top + crop_h > height:
            top = height - crop_h
        for left in range(0, width, stride_w):
            if left + crop_w > width:
                left = width - crop_w

            box = (left, top, left + crop_w, top + crop_h)
            crop = img.crop(box)
            pred_mask = predict_img(net, crop, device, scale_factor, out_threshold).astype(np.uint8)

            pred_mask = zero_border(pred_mask)

            mask_pred[top:top + crop_h, left:left + crop_w] = np.maximum(
                mask_pred[top:top + crop_h, left:left + crop_w],
                pred_mask
            )

    return mask_pred


def overlay_mask_on_image(image: Image.Image, mask: np.ndarray, alpha=0.5, color=(255, 0, 0)):
    """Overlay a binary mask on an RGB image for visualization."""
    image = image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))

    mask_img = Image.fromarray((mask > 0).astype(np.uint8) * 255)  # 二值掩膜
    color_layer = Image.new("RGBA", image.size, color + (int(255 * alpha),))
    overlay.paste(color_layer, mask=mask_img)

    return Image.alpha_composite(image, overlay)


def collect_input_files(input_path):
    """Collect image files from a file path or an input directory."""
    if os.path.isdir(input_path):
        exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.lower().endswith(exts)]
        files.sort()
        return files
    elif os.path.isfile(input_path):
        return [input_path]
    else:
        raise FileNotFoundError(f"{input_path} 不存在")


def get_output_filenames(input_files, output_dir):
    """Create output mask paths for all input files."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    out_files = []
    for f in input_files:
        base_name = os.path.basename(f)  # 取文件名
        name_no_ext, _ = os.path.splitext(base_name)
        out_name = f"{name_no_ext}.png"
        out_files.append(os.path.join(output_dir, out_name))
    return out_files


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    input_paths = []
    for p in args.input:
        input_paths.extend(collect_input_files(p))

    in_files = input_paths
    # 使用新的保存路径参数
    output_dir = args.output 
    out_files = get_output_filenames(in_files, output_dir)

    # net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    # net = DenseUNet(in_channels=3, n_classes=args.classes, growth_rate=16, num_layers=3)
    # net = UnetDVVLinear(n_channels=3, n_classes=args.classes)
    net = UnetDVHLinear(n_channels=3, n_classes=args.classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict, strict=False)
    logging.info('Model loaded!')

    for i, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...')
        img = Image.open(filename).convert('RGB')
        if img.width < 512 or img.height < 512:
            img = img.resize((max(512, img.width), max(512, img.height)), RESAMPLE_BILINEAR)

        mask_orig = sliding_window_predict(net, img, device,
                                           scale_factor=args.scale,
                                           out_threshold=args.mask_threshold,
                                           crop_size=(512, 512),
                                           stride=(341, 341))

        img_rot = img.rotate(90, expand=True)
        mask_rot = sliding_window_predict(net, img_rot, device,
                                          scale_factor=args.scale,
                                          out_threshold=args.mask_threshold,
                                          crop_size=(512, 512),
                                          stride=(341, 341))

        mask_rot_back = Image.fromarray(mask_rot.astype(np.uint8) * 255).rotate(-90, expand=True)
        mask_rot_back = np.array(mask_rot_back) > 127

        h, w = mask_orig.shape
        mask_rot_back = mask_rot_back[:h, :w]

        mask_combined = np.logical_or(mask_orig > 0, mask_rot_back).astype(np.uint8)

        if not args.no_save:
            out_filename = out_files[i]
            result = mask_to_image(mask_combined, mask_values)
            result.save(out_filename)
            logging.info(f'Mask saved to {out_filename}')

        if args.viz:
            overlay = overlay_mask_on_image(img, mask_combined, alpha=0.5, color=(255, 0, 0))
            overlay.show()


    # for i, filename in enumerate(in_files):
    #     logging.info(f'Predicting image {filename} ...')
    #     img = Image.open(filename).convert('RGB')
    #     if img.width < 512 or img.height < 512:
    #         img = img.resize((max(512, img.width), max(512, img.height)), RESAMPLE_BILINEAR)

    #     orig_size = img.size  # 保存原始大小
    #     # 直接对原图做滑动预测（不 resize）
    #     mask = sliding_window_predict(net, img, device,
    #                                 scale_factor=args.scale,
    #                                 out_threshold=args.mask_threshold,
    #                                 crop_size=512, stride=341)


    #     # 裁剪回原图大小
    #     # mask = mask[:orig_size[1], :orig_size[0]]

    #     if not args.no_save:
    #         out_filename = out_files[i]
    #         result = mask_to_image(mask, mask_values)
    #         result.save(out_filename)
    #         logging.info(f'Mask saved to {out_filename}')

    #     if args.viz:
    #         overlay = overlay_mask_on_image(img, mask, alpha=0.5, color=(255, 0, 0))
    #         overlay.show()
