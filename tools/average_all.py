import argparse
import torch
import pathlib
import scipy
import scipy.stats
import numpy as np
from collections import defaultdict
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return round(m, 1), round(h, 1)

all_means = defaultdict(list)
all_std = defaultdict(list)
# AP = []
# AP50 = []
# AP75 = []
# APs = []
# APm = []
# APl = []

def main():
    # ap = []
    # ap50 = []
    # ap75 = []
    # aps = []
    # apm = []
    # apl = []
    # print(log_folder)
    
    for i in range(1,5):
        all_result = defaultdict(list)
        log_folder = "outputs/coco_{0}_mil12_aff005_1shot".format(i)
        print(log_folder)
        log_folder = pathlib.Path(log_folder)
    
        for path in log_folder.glob("*.pth"):
            log = torch.load(path)
            # import pdb; pdb.set_trace()
            # box_results.append(log.results["bbox"]["AP"])
            if "segm" in log.results.keys():
                # import pdb; pdb.set_trace()
                for key in log.results["segm"].keys():
                    all_result[key].append(log.results["segm"][key]*100)
                # all_result["AP"].append(log.results["segm"]["AP"]*100)
                # all_result["AP50"].append(log.results["segm"]["AP50"]*100)
                # all_result["AP75"].append(log.results["segm"]["AP75"]*100)
                # all_result["APs"].append(log.results["segm"]["APs"]*100)
                # all_result["APm"].append(log.results["segm"]["APm"]*100)
                # all_result["APl"].append(log.results["segm"]["APl"]*100)
        # all_means["AP"].append(mean_confidence_interval(ap))
        for key in all_result:
            m, h = mean_confidence_interval(all_result[key])
            all_means[key].append(m)
            all_std[key].append(h)
            # print(key, all_result[key])
    for key in all_means.keys():
        print(key, np.array(all_means[key]), np.array(all_std[key]))
        print(key, round(np.mean(all_means[key]), 1), round(np.mean(all_std[key]), 1))
        # print(key, np.mean(all_std[key]))
    # print(all_means)
    # print(all_std)
    return
        # AP50.append(mean_confidence_interval(ap50))
        # AP75.append(mean_confidence_interval(ap75))
        # APs.append(mean_confidence_interval(aps))
        # APm.append(mean_confidence_interval(apm))
        # APl.append(mean_confidence_interval(apl))
        # print("AP", mean_confidence_interval(ap))
        # print("AP50", mean_confidence_interval(ap50))
        # print("AP75", mean_confidence_interval(ap75))
        # print("APs", mean_confidence_interval(aps))
        # print("APm", mean_confidence_interval(apm))
        # print("APl", mean_confidence_interval(apl))
    # if box_results:
    #     print(mean_confidence_interval(box_results))
    # if ap:
    #     print("AP", mean_confidence_interval(ap))
    # if ap50:
    #     print(ap50)
    #     print("AP50", mean_confidence_interval(ap50))
    # if ap75:
    #     print("AP75", mean_confidence_interval(ap75))
    # if apc:
    #     print(mean_confidence_interval(apc))
    # if apr:
    #     print(mean_confidence_interval(apr))
    
    # AP = np.array(AP)
    print(all_means)
    

if __name__ == "__main__":
    main()
