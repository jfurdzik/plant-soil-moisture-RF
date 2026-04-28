import pandas as pd
from scipy.signal import find_peaks
import numpy as np

# global constants
d2 = 0.3 # reflector buried at depth of 30cm = 0.3m
c = 3e8 # speed of light in a vacuum

# get top 2 highest amps and their corresponding ToFs
def get_max_amps(df):
    # find peaks and get n and p across all channels for this signal
    # update: treat each channel as a sample for NN
    results_n = []
    results_p = []
    signal = df.drop(columns=["time"])

    for col in signal.columns:
        x = signal[col].values
        t = df["time"].values
        peaks, _ = find_peaks(abs(x), height=0.05)

        if len(peaks) >= 2:
            peaks = np.sort(peaks) # sort by time
            idx1 = peaks[0]
            idx2 = peaks[1]
            t1 = t[idx1]
            t2 = t[idx2]
            a1 = x[idx1]
            a2 = x[idx2]

            # calc this channel n and p
            n = calc_RI(t1, t2)
            results_n.append(n)

            p = calc_RAR(a1, a2)
            results_p.append(p)

    return results_n, results_p

# RI or n caclulation using formula defined in paper
def calc_RI(t1, t2):
    return (0.5*c*(t2-t1))/d2

# RAR or p calculation using formula defined in paper
def calc_RAR(a1, a2):
    return a2 / a1

def main():
    # read dry soil csvs (for all 10 signals)
    n_dry = []
    p_dry = []
    for i in range(10):
        print(f"------------CAPTURE {i}------------")
        df = pd.read_csv(f"./Walabot-Data-Saver/raw-signals/soil-capture_{i}_signals.csv")
        all_n, all_p = get_max_amps(df)
        n_dry += all_n
        p_dry += all_p
        
    # read wet soil csvs
    n_wet = []
    p_wet = []
    for i in range(10):
        print(f"------------CAPTURE {i}------------")
        df = pd.read_csv(f"./Walabot-Data-Saver/raw-signals/wet-soil-capture_{i}_signals.csv")
        all_n, all_p = get_max_amps(df)
        n_wet += all_n
        p_wet += all_p
    
    n_dry = np.asarray(n_dry)
    p_dry = np.asarray(p_dry)
    n_wet = np.asarray(n_wet)
    p_wet = np.asarray(p_wet)
    
    print("n dry soil: ", n_dry)
    print("p dry soil: ", p_dry)
    print("n wet soil: ", n_wet)
    print("p wet soil: ", p_wet)

if __name__ == '__main__':
    main()