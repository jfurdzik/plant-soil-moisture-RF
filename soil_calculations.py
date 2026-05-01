import pandas as pd
from scipy.signal import find_peaks
import numpy as np
import neural_net
import random
from torch.utils.data import random_split, Dataset, ConcatDataset, RandomSampler, DataLoader
from torch.nn import MSELoss
from torch.optim import SGD
import torch
import matplotlib.pyplot as plt

# global constants
d2 = 0.3 # reflector buried at depth of 30cm = 0.3m
c = 3e8 # speed of light in a vacuum
K = 4 # sample size
EPOCHS = 100 # change later

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
    eps = 1e-8 # prevent divide by 0
    return a2 / (a1+eps)

class SoilDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def preprocessing_data(n_dry, p_dry, n_wet, p_wet):
    # --------- TRAINING + NN STUFF HERE ----------------
    
    X_dry = torch.stack((n_dry, p_dry), dim=1)
    X_wet = torch.stack((n_wet, p_wet), dim=1) # TODO REVERT THIS FLATTEN LATER after split

    # normalize Xs
    X_dry = (X_dry - X_dry.mean(dim=0)) / X_dry.std(dim=0)
    X_wet = (X_wet - X_wet.mean(dim=0)) / X_wet.std(dim=0)

    y_dry = torch.full((len(X_dry),), 0.1)
    y_wet = torch.full((len(X_wet),), 0.4)
    dataset_dry = SoilDataset(X_dry, y_dry)
    dataset_wet = SoilDataset(X_wet, y_wet)

    # split data 80/20 each wet/dry for training vs testing
    train_size_dry = int(0.8*len(X_dry))
    test_size_dry = len(X_dry) - train_size_dry
    train_size_wet = int(0.8*len(X_wet))
    test_size_wet = len(X_wet) - train_size_wet
    train_data_dry, test_data_dry = random_split(dataset_dry, [train_size_dry, test_size_dry])
    train_data_wet, test_data_wet = random_split(dataset_wet, [train_size_wet, test_size_wet]) # is wet same size as dry? bc channels?
    # recombine wet and dry to single dataset each for train, test
    train_set = ConcatDataset([train_data_dry, train_data_wet])
    test_set = ConcatDataset([test_data_dry, test_data_wet])
    
    return train_set, test_set

# training loop for neural net
def train(dataloader):
    net = neural_net.SoilNet()
    loss_func = MSELoss()
    optimizer = SGD(net.parameters(), lr=0.001, momentum=0.9)
    loss_arr = []

    for epoch in range(EPOCHS):
        print("EPOCH: ", epoch+1)
        net.train(True)
        running_loss = 0
        for data in dataloader:
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs.float())
            loss = loss_func(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print("Running Loss This Epoch: ", running_loss)
        loss_arr.append(running_loss) # loss for all epochs
    print("TRAINING DONE")
    return loss_arr, net #need the smae net for eval

def plot_loss(loss_arr):
    # plot total loss per epoch 
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(range(len(loss_arr)), loss_arr)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Loss Per Epoch")
    ax.set_title(f"Total Loss Across All Epochs")
    ax.grid(True)

    plt.show()

def evaluate_metrics(dataloader, net):
    net.eval()

    # for mae and rmse
    abs_errors = []
    sq_errors = []

    # for predicted vs actual plot
    preds_all = []
    y_all = []

    # for dry vs wet historgram
    dry_preds = []
    wet_preds = []

    with torch.no_grad():
        for x, y in dataloader:
            # pass data thru model
            preds = net(x.float()).squeeze()
            y = y.float()

            # mse, rmse
            abs_errors.append(torch.abs(preds - y))
            sq_errors.append((preds - y) ** 2)
            # pred vs actual
            preds_all.append(preds)
            y_all.append(y.float())

    # calc mae and rmse
    abs_errors = torch.tensor(abs_errors)
    sq_errors = torch.tensor(sq_errors)
    mae = abs_errors.mean().item()
    rmse = torch.sqrt(sq_errors.mean()).item()

    preds_all = np.asarray(preds_all)
    y_all = np.asarray(y_all)

    # plot predicitons vs actual
    plt.figure()
    plt.scatter(y_all, preds_all)
    # perfect predictions line:
    plt.plot([y_all.min(), y_all.max()], [y_all.min(), y_all.max()], linestyle="--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs. Actual Soil Moisture")
    plt.show()

    # plot historgram dry vs wet preds
    # histogram dry vs wet
    for p, label in zip(preds_all, y_all):
        if 0 <= label.item() <= 0.3:
            dry_preds.append(p.item())
        else:
            wet_preds.append(p.item())
    plt.figure()
    plt.hist(dry_preds, bins=20, alpha=0.6, label="Dry")
    plt.hist(wet_preds, bins=20, alpha=0.6, label="Wet")
    plt.xlabel("Predicted Moisture")
    plt.ylabel("Count")
    plt.title("Prediction Distribution: Dry vs. Wet")
    plt.legend()
    plt.show()

    return mae, rmse

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
    
    n_dry = torch.tensor(n_dry)
    p_dry = torch.tensor(p_dry)
    n_wet = torch.tensor(n_wet)
    p_wet = torch.tensor(p_wet)

    # preprocess and split up data
    train_set, test_set = preprocessing_data(n_dry, p_dry, n_wet, p_wet)

    # TRAINING: sample K for net, pass thru with grad descent optim etc
    dl = DataLoader(train_set, batch_size=K, shuffle=True) 
    loss, net = train(dl)
    plot_loss(loss)

    #EVAL: get metrics on test set (mae + rmse, plot residuals, plot histogram wet/dry)
    mae, rmse = evaluate_metrics(test_set, net) # same net we trained
    print("MAE: ", mae)
    print("RMSE: ", rmse)


if __name__ == '__main__':
    main()