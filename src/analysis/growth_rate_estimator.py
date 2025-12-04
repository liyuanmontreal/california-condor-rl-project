"""
Estimate growth rate from real data vs model.
Here focuses on real data.
"""

import pandas as pd
import numpy as np


def main():
    df = pd.read_csv("data/condor_real_data.csv").sort_values("Year")
    N_total = df["Total"].values
    years = df["Year"].values

    r = N_total[1:] / N_total[:-1]

    print("=== Real Data Growth ===")
    print("Years:", years[1:])
    print("r (N_{t+1}/N_t):", r)
    print("Mean r:", r.mean())

if __name__ == "__main__":
    main()
