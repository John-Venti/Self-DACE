import os
import os.path as osp
import matplotlib.pyplot as plt
import cv2
import json
import numpy as np
from copy import deepcopy

IMG_ROOT = 'data/jpeg/23'
ANN_ROOT = 'data/anno/23'
PDF_PATH = './vis.pdf'
JPG_PATH = './vis.jpg'

IMAGE_LIST   = [23, 27, 31, 65, 83, 97]
PREFIX_MAP   = {'original': '_ll', 'predict': '_hl_pred', 'gt': '_hl'}
IMAGE2POINTS = {"23": [438, 512], "27": [617, 534], "31": [680, 398], "65": [209, 677], "83": [781, 594], "97": [727, 616]}

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.8])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=175):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def read_mask(mask_path: str, h: int, w: int):
    ann       = json.load(open(mask_path, 'r'))
    poly_list = [np.array(_["points"], dtype=np.int32) for _ in ann]
    mask      = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, poly_list, 1)
    return mask

def calc_iou(mask1, mask2):
    assert mask1.shape == mask2.shape
    mask_inter = np.logical_and(mask1, mask2)
    mask_union = np.logical_or(mask1, mask2)
    iou        = np.sum(mask_inter) / np.sum(mask_union)
    return iou

def main():

    fig, axes = plt.subplots(3, 6, figsize=(18, 6))
    axes = axes.flatten()
    idx_ax = 0
    for image in IMAGE_LIST:
        ori_image_path = osp.join(IMG_ROOT, f'{image}{PREFIX_MAP["original"]}.png')
        enh_image_path = osp.join(IMG_ROOT, f'{image}{PREFIX_MAP["predict"]}.png')
        assert osp.exists(ori_image_path) and osp.exists(enh_image_path)
        ori_image = cv2.imread(ori_image_path)
        enh_image = cv2.imread(enh_image_path)

        h, w = ori_image.shape[:2]
        assert enh_image.shape[0] == h and enh_image.shape[1] == w

        ori_pred_mask_path = osp.join(ANN_ROOT, f'{image}{PREFIX_MAP["original"]}.json')
        enh_pred_mask_path = osp.join(ANN_ROOT, f'{image}{PREFIX_MAP["predict"]}.json')
        gt_mask_path = osp.join(ANN_ROOT, f'{image}{PREFIX_MAP["gt"]}.json')

        ori_pred_mask = read_mask(ori_pred_mask_path, h, w)
        enh_pred_mask = read_mask(enh_pred_mask_path, h, w)
        gt_mask       = read_mask(gt_mask_path, h, w)
        iou_ori_pred  = calc_iou(ori_pred_mask, gt_mask)
        iou_enh_pred  = calc_iou(enh_pred_mask, gt_mask)

        # Prediction w/o Self-DACE
        ax = axes[idx_ax]
        idx_ax += 1
        ax.imshow(cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB))
        ax.set_xticks([])
        ax.set_yticks([])
        show_mask(ori_pred_mask, ax)
        show_points(np.array([IMAGE2POINTS[str(image)]]), np.array([1]), ax)
        ax.text(10, 70, f'IoU: {iou_ori_pred:.2f}', color='red', fontsize=12, fontweight='bold')

        # Prediction w/ Self-DACE
        ax = axes[idx_ax]
        idx_ax += 1
        ax.imshow(cv2.cvtColor(enh_image, cv2.COLOR_BGR2RGB))
        ax.set_xticks([])
        ax.set_yticks([])
        show_mask(enh_pred_mask, ax)
        show_points(np.array([IMAGE2POINTS[str(image)]]), np.array([1]), ax)
        ax.text(10, 70, f'IoU: {iou_enh_pred:.2f}', color='red', fontsize=12, fontweight='bold')

        # GT
        ax = axes[idx_ax]
        idx_ax += 1
        ax.imshow(cv2.cvtColor(enh_image, cv2.COLOR_BGR2RGB))
        ax.set_xticks([])
        ax.set_yticks([])
        show_mask(gt_mask, ax)
        show_points(np.array([IMAGE2POINTS[str(image)]]), np.array([1]), ax)
        ax.text(10, 70, 'GT', color='red', fontsize=12, fontweight='bold')

    # 在第一行上方添加标题 [Original, Enhanced, GT]
    axes[0].set_title('Original', fontsize=14)
    axes[1].set_title('Enhanced', fontsize=14)
    axes[2].set_title('GT', fontsize=14)
    axes[3].set_title('Original', fontsize=14)
    axes[4].set_title('Enhanced', fontsize=14)
    axes[5].set_title('GT', fontsize=14)


    plt.tight_layout()
    plt.savefig(PDF_PATH)
    plt.savefig(JPG_PATH)

if __name__ == '__main__':
    main()
