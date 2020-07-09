import pdb
import json
from collections import Counter, defaultdict

num_of_point_in_mcg_json = defaultdict(int)
num_of_point_in_coco_json = defaultdict(int)
with open("datasets/coco/annotations/coco_train_mcg.json") as f:
    myjson = json.load(f)
    annotations = myjson["annotations"]
    print("There are totoal {0} annotations".format(len(annotations)))
    # error_idx = []
    for i, ann in enumerate(annotations):
        num = len(ann["segmentation"][0])
        num_of_point_in_mcg_json[num] += 1

        # if num < 6: # less than 6 point will cause error
        #     error_idx.append(i)


with open("datasets/coco/annotations/instances_train2017.json") as f:
    myjson = json.load(f)
    annotations = myjson["annotations"]
    print("There are totoal {0} annotations".format(len(annotations)))
    for i, ann in enumerate(annotations):
        if ann["iscrowd"] == 1:
            continue
        # if isinstance(ann["segmentation"], list):
        num = len(ann["segmentation"][0])
        num_of_point_in_coco_json[num] += 1
        # else:
        #     pdb.set_trace()
        #     print(ann)

    #print(dict(sorted(num_of_point_in_coco_json.items())))
import pprint
pp = pprint.PrettyPrinter(indent=4)

print("num_of_point_in_mcg_json:")
print(dict(sorted(num_of_point_in_mcg_json.items())))
pp.pprint(num_of_point_in_mcg_json)
print("num_of_point_in_coco_json:")
print(dict(sorted(num_of_point_in_coco_json.items())))
pp.pprint(num_of_point_in_coco_json)
# pdb.set_trace()