import argparse
import torch

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate COCO result from file")

    parser.add_argument(
        "--output_folder",
        default="coco_1",
        help="output folder",
        type=str,
        required=True
    )
    args = parser.parse_args()
    output_folder = args.output_folder
    #print("output_folder=%s" % output_folder)
    # resFile = './%s/inference/coco_2017_val/coco_results.pth' % (output_folder)
    #voc_2012_instance_val_cocostyle
    resFile = './%s/inference/lvis_val_cocostyle/coco_results.pth' % (output_folder)
    result = torch.load(resFile)
    print(result)

if __name__ == "__main__":
    main()
