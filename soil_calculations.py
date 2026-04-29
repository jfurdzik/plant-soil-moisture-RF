import pandas as pd
from scipy.signal import find_peaks
import numpy as np
import neural_net
import random
from torch.utils.data import random_split, Dataset, ConcatDataset, RandomSampler, DataLoader
from torch.nn import BCELoss
from torch.optim import SGD

# global constants
d2 = 0.3 # reflector buried at depth of 30cm = 0.3m
c = 3e8 # speed of light in a vacuum
K = 4 # sample size
EPOCHS = 20 # change later

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

class SoilDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# sample K data points and create + return dataloader
def sample_K(dataset):
    # need to get the same matching n and p elem
    idx = random.sample(range(len(dataset)))
    n_vec = [dataset[i][0] for i in idx] # n is first elem in tuple
    p_vec = [dataset[i][1] for i in idx] # p second elem in tuple
    x = np.concatenate([n_vec, p_vec])
    return x


def preprocessing_data(n_dry, p_dry, n_wet, p_wet):
    # --------- TRAINING + NN STUFF HERE ----------------
    # also split data 80/20 each wet/dry for training vs testing
    X_dry = np.column_stack((n_dry, p_dry))
    X_wet = np.column_stack((n_wet, p_wet)) # TODO REVERT THIS FLATTEN LATER after split
    y_dry = np.zeros(X_dry.shape[0])
    y_wet = np.ones(X_wet.shape[0])
    dataset_dry = SoilDataset(X_dry, y_dry)
    dataset_wet = SoilDataset(X_wet, y_wet)

    train_size = int(0.8*len(X_dry))
    test_size = len(X_dry) - train_size
    train_data_dry, test_data_dry = random_split(dataset_dry, [train_size, test_size])
    train_data_wet, test_data_wet = random_split(dataset_wet, [train_size, test_size]) # is wet same size as dry? bc channels?
    train_set = ConcatDataset([train_data_dry, train_data_wet])
    test_set = ConcatDataset([test_data_dry, test_data_wet])
    
    return train_set, test_set

# training loop for neural net
def train(dataloader):
    net = neural_net.SoilNet()
    loss_func = BCELoss()
    optimizer = SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(EPOCHS):
        print("EPOCH: ", epoch)
        net.train(True)
        running_loss = 0
        for data in dataloader:
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print("Running Loss This Epoch: ", running_loss)
    print("TRAINING DONE")
    
def evaluate(dataloader):
    return None # TODO

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

    # preprocess/split data, then sample K for net
    train_set, test_set = preprocessing_data(n_dry, p_dry, n_wet, p_wet) 
    K_samples = sample_K(train_set)
    dl = DataLoader(K_samples)
    train(dl)

    # TODO at some point concat and flatten (X is n and p)??
    # TODO need to normalize data since n and p dif scales

if __name__ == '__main__':
    main()