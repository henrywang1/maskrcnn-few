# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
import torch
import torchvision
import json
import random
import math

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.structures.keypoint import PersonKeypoints
from maskrcnn_benchmark.utils.comm import is_main_process,synchronize

# RuntimeError: unable to open shared memory object XXXX in read-write mode
# OSError: [Errno 24] Too many open files

import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')

min_keypoints_per_image = 10


def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True
    return False


class COCODataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self, ann_file, root, remove_images_without_annotations, transforms=None, use_transfer=False
    ):
        self.use_transfer = use_transfer
        self.is_lvis = True if "lvis" in ann_file else False
        self.is_train = False
        self.label_set = {}
        if self.is_lvis:
            ann_path = ann_file[:ann_file.rfind("/")+1]
            label_set_file = ann_path + "label_set.json"
            self.cids = [*range(1,1231)]
            if not os.path.isfile(label_set_file) and is_main_process():
                with open(ann_file) as f_in:
                    anns = json.load(f_in)
                    cat_f = []
                    cat_c = []
                    cat_r = []
                    for a in anns["categories"]:
                        if a["frequency"] == "f":
                            cat_f.append(a["id"])
                        if a["frequency"] == "c":
                            cat_c.append(a["id"])
                        if a["frequency"] == "r":
                            cat_r.append(a["id"])
                    self.label_set = {"cat_f":cat_f, "cat_c":cat_c, "cat_r":cat_r}
                    with open(label_set_file, 'w') as f:
                        json.dump(self.label_set, f, indent=2)
            synchronize()
            with open(label_set_file, 'r') as f:
                self.label_set = json.load(f)
            if "train" in ann_file:
                self.is_train = True
            else:
                ann_file_new = ann_file + "_fix"
                if not os.path.isfile(ann_file_new) and is_main_process():
                    with open(ann_file) as f_in:
                        anns = json.load(f_in)

                    if "_" in anns["images"][0]["file_name"]:
                        replace_file_name = lambda x: (x["file_name"].split("_")[2])
                        _ = [i.update({"file_name": replace_file_name(i)}) for i in anns["images"]]

                    with open(ann_file_new, "w") as f_out:
                        json.dump(anns, f_out)
                ann_file = ann_file_new
        self.ann_file = ann_file
        synchronize()
        print("Use annotation {}".format(ann_file))
        super(COCODataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)
   
        # filter images without detection annotations
        class_fractions = {i: 0 for i in range(1, 1231)}
        img_cls = {i: [] for i in self.ids}
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    ids.append(img_id)
                    if self.is_train:
                        img_cids = list(set(ann["category_id"] for ann in anno))
                        img_cls[img_id] = img_cids
                        for c in img_cids:
                            class_fractions[c] += 1

            self.ids = ids
            if self.is_train:
                for k, v in class_fractions.items():
                    class_fractions[k] = v/len(ids)
                self.img_repeat_factor = {i: 0.0 for i in self.ids}
                for img_id in self.ids:
                    all_class_fractions = [class_fractions[i] for i in img_cls[img_id]]
                    all_repeat_factor = [math.sqrt(0.001/fc) for fc in all_class_fractions]
                    repeat_factor = max(all_repeat_factor)
                    repeat_factor = max(1, repeat_factor)
                    self.img_repeat_factor[img_id] = repeat_factor

        self.img_to_id_map = {k: v for v, k in enumerate(self.ids)}
        self.categories = {cat['id']: cat['name'] for cat in self.coco.cats.values()}

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self._transforms = transforms

    def __getitem__(self, idx):
        img, anno = super(COCODataset, self).__getitem__(idx)

        # filter crowd annotations
        # TODO might be better to add an extra field
        if not self.is_lvis: # coco
            anno = [obj for obj in anno if obj["iscrowd"] == 0]

        # if self.is_train and self.use_transfer:
        #     cls_label = random.sample(self.cids, 1)[0]
        #     img_by_class = self.coco.getImgIds(catIds=[cls_label])
        #     idx = random.sample(set(img_by_class), 1)[0]
        #     idx = self.img_to_id_map[idx]
        #     img, anno = super(COCODataset, self).__getitem__(idx)

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        if anno and "segmentation" in anno[0]:
            masks = [obj["segmentation"] for obj in anno]
            masks = SegmentationMask(masks, img.size, mode='poly')
            target.add_field("masks", masks)

        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = PersonKeypoints(keypoints, img.size)
            target.add_field("keypoints", keypoints)

        target = target.clip_to_image(remove_empty=True)

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data
