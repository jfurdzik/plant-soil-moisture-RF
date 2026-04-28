import pandas as pd

# get top 2 highest amps and their corresponding ToFs
def get_max_amps(df):
    lst = []
    signal = df.drop(columns=["time"])
    max_amps = signal.abs().max().sort_values(ascending=False) # sort max amplitudes in descending order
    top2 = max_amps[0:2] # top 2 max amps
    # get tof for each of those
    for channel in top2.index:
        idx = signal[channel].abs().idxmax()
        tof = df.loc[idx, "time"] # lookup time for that peak from original df
        amp = df.loc[idx, channel]
        print(f"{channel} -> ToF={tof}, amplitude={amp}")
        lst.append((channel, tof, amp)) # add all to list as tuple
    return lst

def main():
    # read dry soil csvs
    for i in range(10):
        print(f"------------CAPTURE {i}------------")
        df = pd.read_csv(f"./Walabot-Data-Saver/raw-signals/soil-capture_{i}_signals.csv")
        lst = get_max_amps(df)
        print(lst)
    # read wet soil csvs
    for i in range(10):
        print(f"------------CAPTURE {i}------------")
        df = pd.read_csv(f"./Walabot-Data-Saver/raw-signals/wet-soil-capture_{i}_signals.csv")
        lst_w = get_max_amps(df)
        # print(lst_w)

if __name__ == '__main__':
    main()