import argparse
import pathlib
import pandas as pd
import numpy as np
import scipy
from scipy import stats, optimize, interpolate
import math
pd.options.display.float_format = '{:,.3f}'.format

def mean_confidence_interval(data, confidence=0.95):
    #a = data
    #a = 1.0 * np.array(data)
    #n = len(data)
    #m, se = data.mean(), data.sem()
    #h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    #import pdb; pdb.set_trace()
    m, c, s = data.mean(), data.count(), data.std()
    h = 1.96*s/math.sqrt(c)
    return m, h #m-h, m+h

def main():
    parser = argparse.ArgumentParser(
        description="Calculate 0.95 confidence interval")

    parser.add_argument(
        "--log_folder",
        default="./output/1/bbox",
        help="log folder",
        type=str,
        required=True
    )
    args = parser.parse_args()
    log_folder = args.log_folder


    # define the path
    log_folder = pathlib.Path(log_folder)
    all_log_files = []
    for path in log_folder.glob("*.csv"):
        df = pd.read_csv(path, index_col=0)
        all_log_files.append(df)   


    all_log_files = pd.concat(all_log_files, axis=1)
    result = all_log_files.apply(mean_confidence_interval, axis=1)
    result = result.apply(pd.Series)
    print(result)

if __name__ == "__main__":
    main()

