import argparse
import torch
import pathlib
import scipy
import scipy.stats
import numpy as np

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return round(m, 1), round(h, 1)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate COCO result from file")

    parser.add_argument(
        "--folder",
        default="models/coco_1",
        help="folder",
        type=str,
        required=True
    )
    args = parser.parse_args()
    log_folder = pathlib.Path(args.folder)
    box_results = []
    segm_results = []

    ap = []
    ap50 = []
    ap75 = []
    aps = []
    apm = []
    apl = []
    # print(log_folder)
    for path in log_folder.glob("*.pth"):
        log = torch.load(path)
        # import pdb; pdb.set_trace()
        # box_results.append(log.results["bbox"]["AP"])
        if "segm" in log.results.keys():
            # import pdb; pdb.set_trace()
            ap.append(log.results["segm"]["AP"]*100)
            ap50.append(log.results["segm"]["AP50"]*100)
            ap75.append(log.results["segm"]["AP75"]*100)
            aps.append(log.results["segm"]["APs"]*100)
            apm.append(log.results["segm"]["APm"]*100)
            apl.append(log.results["segm"]["APl"]*100)


    print("AP", mean_confidence_interval(ap))
    print("AP50", mean_confidence_interval(ap50))
    print("AP75", mean_confidence_interval(ap75))
    print("APs", mean_confidence_interval(aps))
    print("APm", mean_confidence_interval(apm))
    print("APl", mean_confidence_interval(apl))
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

if __name__ == "__main__":
    main()
