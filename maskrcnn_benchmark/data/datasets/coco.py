# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from collections import defaultdict
import json
import os
import torch
import torchvision

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.structures.keypoint import PersonKeypoints
from maskrcnn_benchmark.utils.comm import is_main_process, synchronize


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
        self,
        ann_file,
        root,
        remove_images_without_annotations,
        transforms=None,
        split=0,
        extract_feature=False,
        load_mask=False
    ):
        self.is_train = remove_images_without_annotations
        self.is_lvis = "lvis" in ann_file or False
        self.load_mask = load_mask
        if self.is_lvis:
            ann_file = self.preprocess_lvis(ann_file, extract_feature)
        elif split:
            ann_file = self.preprocess_coco(ann_file, split)

        synchronize()
        print("Use annotation {}".format(ann_file))
        self.ann_file = ann_file
        super(COCODataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)
        # json_category_id class fraction
        class_fractions = defaultdict(int)
        self.img_cls = defaultdict(list)
        self.cls_img = defaultdict(list)

        # filter images without detection annotations
        if remove_images_without_annotations:
            ids = []
            num_of_anno = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if len(anno) < 350 and has_valid_annotation(anno):
                    num_of_anno.append(len(anno))
                    ids.append(img_id)
                    img_cids = list(set(ann["category_id"] for ann in anno))
                    assert img_cids
                    self.img_cls[img_id] = img_cids
                    for c in img_cids:
                        class_fractions[c] += 1
                        self.cls_img[c].append(img_id)
            self.ids = ids
            for k, v in class_fractions.items():
                class_fractions[k] = v/len(ids)

            self.img_repeat_factor = defaultdict(float)
            for img_id in self.ids:
                all_class_fractions = [class_fractions[i] for i in self.img_cls[img_id]]
                repeat_factor = [1.0/fc for fc in all_class_fractions]
                repeat_factor = max(repeat_factor)
                repeat_factor = max(1.0, repeat_factor)
                self.img_repeat_factor[img_id] = repeat_factor

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self.img_to_id_map = {v: k for k, v in enumerate(self.ids)}
        self._transforms = transforms

    def preprocess_lvis(self, ann_file, extract_feature):
        suffix = "_freq" if self.is_train else "_common_rare"
        ann_file_new = ann_file + suffix
        if not os.path.isfile(ann_file_new) and is_main_process():
            with open(ann_file) as f_in:
                anns = json.load(f_in)
                cids_lvis_f = [a["id"] for a in anns["categories"] if a["frequency"] == "f"]
            if self.is_train:
                anns["annotations"] = [v for v in anns["annotations"] if v["category_id"] in cids_lvis_f]
            else:
                anns["annotations"] = [v for v in anns["annotations"] if v["category_id"] not in cids_lvis_f]
                # classes not exhaustively annotated should not be support examples of an image
                if not extract_feature:
                    img_ne = {d["id"]: d["not_exhaustive_category_ids"] for d in anns["images"]}
                    anns["annotations"] = [v for v in anns["annotations"] if v["category_id"] not in img_ne[v["image_id"]]]

                # the file_name field in the lvis val dataset is broken
                replace_file_name = lambda x: (x["file_name"].split("_")[2])
                if "_" in anns["images"][0]["file_name"]:
                    for img in anns["images"]:
                        img.update({"file_name": replace_file_name(img)})

            with open(ann_file_new, "w") as f_out:
                json.dump(anns, f_out)
        return ann_file_new

    def preprocess_coco(self, ann_file, split):
        if split == 5 and not self.is_train:
            return ann_file
        ann_file_new = ann_file + "_" + str(split)
        if not os.path.isfile(ann_file_new) and is_main_process():
            with open(ann_file) as f_in:
                anns = json.load(f_in)
            if split == 5: #voc non-voc
                voc_inds = (0, 1, 2, 3, 4, 5, 6, 8, 14, 15, 16, 17, 18, 19, 39, 56, 57, 58, 60, 62)
                split_cat = [a["id"] for i, a in enumerate(anns["categories"]) if not i in voc_inds]
            else:
                if not self.is_train:
                    split_cat = [a["id"] for i, a in enumerate(anns["categories"]) if i % 4 == (split-1)]
                else:
                    split_cat = [a["id"] for i, a in enumerate(anns["categories"]) if not i % 4 == (split-1)]
            anns["annotations"] = [v for v in anns['annotations'] if v["category_id"] in split_cat]
            with open(ann_file_new, "w") as f_out:
                json.dump(anns, f_out)

        return ann_file_new

    def __getitem__(self, idx):
        img, anno = super(COCODataset, self).__getitem__(idx)

        # filter crowd annotations
        # TODO might be better to add an extra field
        if not self.is_lvis:
            anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        if self.load_mask and anno and "segmentation" in anno[0]:
            masks = [obj["segmentation"] for obj in anno]
            masks = SegmentationMask(masks, img.size, mode='poly')
            target.add_field("masks", masks)

        if not self.is_train:
            img_ids = [ann["image_id"] for ann in anno]
            ann_ids = [ann["id"] for ann in anno]
            img_ids = torch.tensor(img_ids)
            ann_ids = torch.tensor(ann_ids)
            target.add_field("img_ids", img_ids)
            target.add_field("ann_ids", ann_ids)
        # if anno and "keypoints" in anno[0]:
        #     keypoints = [obj["keypoints"] for obj in anno]
        #     keypoints = PersonKeypoints(keypoints, img.size)
        #     target.add_field("keypoints", keypoints)

        target = target.clip_to_image(remove_empty=True)

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data
