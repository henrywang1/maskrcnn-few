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
    return m, h

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
    # apr = []
    # apc = []
    print(log_folder)
    for path in log_folder.glob("*.pth"):
        log = torch.load(path)
        # import pdb; pdb.set_trace()
        # box_results.append(log.results["bbox"]["AP"])
        if "segm" in log.results.keys():
            ap.append(log.results["segm"]["AP"])
            ap50.append(log.results["segm"]["AP50"])
            ap75.append(log.results["segm"]["AP75"])

            # if "APr" in log.results["segm"].keys():
            #     apr.append(log.results["segm"]["APr"])
            # if "APc" in log.results["segm"].keys():
            #     apc.append(log.results["segm"]["APc"])


    # if box_results:
    #     print(mean_confidence_interval(box_results))
    if ap:
        print("AP", mean_confidence_interval(ap))
    if ap50:
        print("AP50", mean_confidence_interval(ap50))
    if ap75:
        print("AP75", mean_confidence_interval(ap75))
    # if apc:
    #     print(mean_confidence_interval(apc))
    # if apr:
    #     print(mean_confidence_interval(apr))

if __name__ == "__main__":
    main()
