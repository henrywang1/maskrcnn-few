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
    for path in log_folder.glob("*.pth"):
        log = torch.load(path)
        # box_results.append(log.results["bbox"]["AP50"])
        if "segm" in log.results.keys():
            segm_results.append(log.results["segm"]["AP50"])

    if box_results:
        print(mean_confidence_interval(box_results))
    if segm_results:
        print(mean_confidence_interval(segm_results))

if __name__ == "__main__":
    main()
