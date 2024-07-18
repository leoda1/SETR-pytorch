# -*- coding: UTF-8 -*-
"""
===================================================================================
@author : Leoda
@Date   : 2024/07/18 11:39:20
@Project -> : SETR-pytorch$
==================================================================================
"""
import os
from PIL import Image
from tqdm import tqdm

from setr import setr
from utils.utils_metrics import compute_mIoU, show_results
if __name__ == "__main__":
    miou_mode = 0
    num_classes = 11
    name_classes = ["background","aeroplane", "bicycle", "bird", "boat", "bottle",
                    "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
                    "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
                    "tvmonitor"]
    VOCdevkit_path = 'VOCdevkit'

    image_ids = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"), 'r').read.splitline()
    gt_dir = os.path.join(VOCdevkit_path, "VOC2007/SegmenationClass/")
    miou_out_dir = "miou_out"
    pred_dir = os.path.join(miou_out_dir, 'detection-results')

    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)

        print("Load model:")
        setr = setr()
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/" + image_id + ".jpg")
            image = Image.open(image_path)
            image = setr.get_miou_png(image)
            image.save(os.path.join(pred_dir, image_id + ".png"))

    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes, name_classes)  # 执行计算mIoU的函数
        print("Get miou done.")
        show_results(miou_out_dir, hist, IoUs, PA_Recall, Precision, name_classes)


