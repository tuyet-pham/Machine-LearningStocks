import pandas as pd
from sklearn import tree
from numpy import array
import numpy as np
import matplotlib as plot
# import tqdm
import datetime as dt
import time
import os
import random

pd.options.display.max_rows = 10

# To get the list of Name values from the 'Name' column from the
alst = pd.read_csv("data/all_stocks_5yr.csv")
alst = alst['Name'].unique()

# create and open the listall.csv
file1 = open("data/listall.csv", "wt")

# Column header
file1.writelines("Name\n")

# Write the Name of stocks
for e in alst:
    file1.writelines(e + "\n")
file1.close()


def train_dev_file(dev_set, train_set, stock_name, custom_feat=False):
    # for i in tqdm.tqdm(range(200)):
    #     time.sleep(0.01)

    ts = (r"data/train/" + stock_name + "_train_data.csv")
    ds = (r"data/dev/" + stock_name + "_dev_data.csv")

    dev_set.to_csv(ds, index=custom_feat)
    train_set.to_csv(ts, index=custom_feat)

    time.sleep(.2)  # comment this out if you're going to use the tqdm library
    print("Train and Dev set file creation - Done")


# Getting the a tuple from the file - train and dev set
def train_dev(stockname, dataframe=False):
    """[summary]
        Getting a tuple from the file - train and dev set of a specific stock
    Args:
        stockname (string):
    Returns:
        [tuples]: train_stock, dev_stock
    """
    if dataframe:
        df = stockname
    else:
        print("\nStock good: %s\nUsing 2013-2015 for training and 2016-2018.." % stockname)
        filepath = 'data/individual_stocks_5yr/' + stockname + '_data.csv'
        df = pd.read_csv(filepath)

    # Getting within the time
    if dataframe:
        train_stock = df[(df.index > '2012-12-31') & (df.index < '2016-01-01')]
        dev_stock = df[(df.index > '2015-12-31') & (df.index < '2019-01-01')]
    else:
        train_stock = df[(df['date'] > '2012-12-31') & (df['date'] < '2016-01-01')]
        dev_stock = df[(df['date'] > '2015-12-31') & (df['date'] < '2019-01-01')]

    # Just Printing timer for extra coolness
    # for i in tqdm.tqdm(range(200)):
    #     time.sleep(0.01)

    time.sleep(.2)  # comment this out if you're going to use the tqdm library
    print("Train and Dev set creation - Done")
    return train_stock, dev_stock


# if we want to do this later I guess
def train_all():
    print("Running ALL... might be a minute")


# Open listall
try:
    file1 = pd.read_csv('data/listall.csv')
except FileExistsError:
    print('\nFile doesnt exist.')


# Picking an individual stock
def pick_stock():
    """[summary]
        Make the user pick a correct stock
    Returns:
        [string] - stkname: Good name of file
    """
    found = False
    stkname = ""

    # Get a good name before continuing
    while found == False:
        stkname = input("Enter stock name or QUIT: ").upper()

        if stkname == "QUIT":
            return ""

        for e in file1['Name']:
            if e == stkname:
                return stkname

        print("Can't find %s Try again?" % stkname)

    return ""


def custom_features(stock_input=None):
    # build custom_features dataframe for a single stock

    # file_name = "data/individual_stocks_5yr/AAPL_data.csv"
    if not stock_input:
        file_name = input("enter stock file name (individual stock): ")
    else:
        file_name = "data/individual_stocks_5yr/" + stock_input + "_data.csv"
    print(file_name)

    stock_data = pd.read_csv(file_name, index_col="date")
    feature_names = ["target %", "close-open %", "2 week movement", "Stochastic Oscillator", "Williams %R", "average"]

    for item in stock_data.columns:
        if item != "date":
            feature_names.append(item)
    # feature_names.append("volume")
    custom_features = pd.DataFrame(index=stock_data.index, columns=feature_names)

    for i in stock_data.index:
        row = stock_data.loc[i]
        # print(dt.datetime.strptime(i, "%Y-%m-%d") - dt.datetime.strptime(stock_data.iloc[0].name, "%Y-%m-%d"))
        average = (row['high'] + row['low']) / 2
        custom_features.loc[i]['average'] = average
        today = dt.datetime.strptime(i, "%Y-%m-%d")
        for col in stock_data.columns:
            custom_features.loc[i][col] = row[col]
        custom_features.loc[i]["volume"] = row["volume"]
        if (today - dt.datetime.strptime(stock_data.iloc[0].name, "%Y-%m-%d")).days < 14:
            prev_loc = i
            continue
        close_open = (row['close'] - row['open']) * 100.0 / row['open']
        average = (row['high'] + row['low']) / 2
        fourteen_days_ago = today - dt.timedelta(days=14)
        fourteen_days_ago = fourteen_days_ago.isoformat().split('T')[0]
        try:
            custom_features.loc[fourteen_days_ago]
        except:
            fourteen_days_ago = today - dt.timedelta(days=13)
            fourteen_days_ago = fourteen_days_ago.isoformat().split('T')[0]
            try:
                custom_features.loc[fourteen_days_ago]
            except:
                fourteen_days_ago = today - dt.timedelta(days=12)
                fourteen_days_ago = fourteen_days_ago.isoformat().split('T')[0]
                try:
                    custom_features.loc[fourteen_days_ago]
                except:
                    fourteen_days_ago = today - dt.timedelta(days=11)
                    fourteen_days_ago = fourteen_days_ago.isoformat().split('T')[0]
        two_week_change = average - custom_features.loc[fourteen_days_ago]['average']
        last_two_weeks = stock_data.loc[fourteen_days_ago:i]
        stochastic = 100 * float(
            (row['close'] - min(last_two_weeks['low'])) / (max(last_two_weeks['high']) - min(last_two_weeks['low'])))
        williams = -100 * float(
            (max(last_two_weeks['high']) - row['close']) / (max(last_two_weeks['high']) - min(last_two_weeks['low'])))
        # feature_names = ["close-open", "2 week movement", "Stochastic Oscillator", "Williams %R","average"]
        custom_features.loc[i]["close-open %"] = close_open
        custom_features.loc[i]["2 week movement"] = two_week_change
        custom_features.loc[i]["Stochastic Oscillator"] = stochastic
        custom_features.loc[i]["Williams %R"] = williams
        target = average - custom_features.loc[prev_loc]['average']
        target = 100 * target / custom_features.loc[prev_loc]['average']
        custom_features.loc[prev_loc]["target %"] = target
        prev_loc = i

    custom_features = custom_features.drop(custom_features.index[0:10], axis=0)
    custom_features = custom_features.drop(custom_features.index[len(custom_features.index) - 1], axis=0)
    # custom_features.to_csv(file_name.split('.')[0] + '_custom_features.csv')
    return custom_features


def RandomForest(train_sets):
    forest = []
    for i in range(0, len(train_sets)):
        # for now
        sub_set = train_sets[i]
        # sub_set = Bagging_here(train_set) - Kyle <--- best_features = featureselection(train_set) # Caleb
        sub_target = sub_set['target %']
        sub_set = sub_set.drop(columns=COLUMNS_TO_DROP)
        n_tree = tree.DecisionTreeRegressor(criterion='mse', min_samples_leaf=20)
        n_tree.fit(sub_set, sub_target)
        forest.append(n_tree)
    return forest


def MakePredictions(forest, set):
    set = set.drop(columns=COLUMNS_TO_DROP)
    pred_set = []
    final_preds = []
    for n_tree in forest:
        pred_set.append(n_tree.predict(set))
    pred_set_size = len(pred_set)
    for i in range(0, len(pred_set[0])):
        sum = 0
        for pred in pred_set:
            sum += pred[i]
        final_preds.append(sum / pred_set_size)
    return final_preds


def RandomSubsets(data: pd.DataFrame, n_subsets: int, frac_of_set: float):
    random.seed(dt.datetime.now().microsecond)
    subsets = []
    for i in range(0, n_subsets):
        rand_num = random.randint(0, 99999999)
        subset = data.sample(frac=frac_of_set, random_state=rand_num)
        subsets.append(subset)
    return subsets


def GetMSE(preds, target):
    sum = 0
    n = len(preds)
    if n != len(target):
        return
    for i in range(0, n):
        sum += (preds[i] - target[i]) ** 2
    return float(sum / n)


def GenerateLabels(data):
    labels = []
    high = CUTOFF
    low = -CUTOFF
    for d in data:
        labels.append("sell" if d < low else "buy " if d > high else "hold")
    return labels


def NumericalLabelScore(data):
    results = []
    for i in range(0, len(data)):
        results.append(1 if data[i] == "buy " else 0 if data[i] == "hold" else -1)
    return results


def LabeledMSE(preds, target):
    return GetMSE(NumericalLabelScore(preds), NumericalLabelScore(target))


def Baseline(target):
    baseline = []
    for dat in target:
        if type(dat) == str:
            baseline.append("hold")
        else:
            baseline.append(0)
    if type(baseline[0]) == int:
        return GetMSE(baseline, target)
    return LabeledMSE(baseline, target)


train_set = pd.DataFrame()
dev_set = pd.DataFrame()

# FOR TUNING
# unhelpful columns
COLUMNS_TO_DROP = ['target %', 'Name', 'open', 'high', 'low', 'close']
# cutoff between buy, sell, hold
CUTOFF = 0.5

if __name__ == "__main__":
    # Run to make life easier
    all_or_one = "n"
    # all_or_one = input("Train ALL? *Say no for now* [y/n]: ")

    if all_or_one == "n":
        stock_name = pick_stock()
        if stock_name != "":
            ts = (r"data/train/" + stock_name + "_train_data.csv")
            ds = (r"data/dev/" + stock_name + "_dev_data.csv")
            # noinspection PyBroadException
            try:
                train_set = pd.read_csv(ts, index_col="date")
                dev_set = pd.read_csv(ds, index_col="date")
            except:
                custom = custom_features(stock_name)
                train_set, dev_set = train_dev(custom, dataframe=True)
            #print("\nTraining set\n", train_set, "\n\nDevset\n", dev_set)
            # !Important - This will create .csv file of each sets.
            # if using custom features, the fourth parameter should be true
            train_dev_file(dev_set, train_set, stock_name, True)


            '''train_target = train_set['target %']
            train_set = train_set.drop(columns=['target %', 'Name'])'''
            dev_target = dev_set['target %']

            random_subsets = RandomSubsets(train_set, 20, 0.5)
            forest = RandomForest(random_subsets)
            preds = MakePredictions(forest, dev_set)
            preds_labels = GenerateLabels(preds)
            dev_target_labels = GenerateLabels(dev_target)
            for i in range(0, len(preds)):
                print("pred: {}   actual: {}".format(preds[i], dev_target[i]))
            print("Unlabeled MSE: ", end="")
            print(GetMSE(preds, dev_target))
            print("Unlabeled baseline: ", end="")
            print(Baseline(dev_target))
            print("Labeled MSE: ", end="")
            print(LabeledMSE(preds_labels, dev_target_labels))
            print("Labeled baseline:  ", end="")
            print(Baseline(dev_target_labels))

        # [Caleb] Figure out what features are significant
        # use pearsons correlation (Machine Learning HW2)

        # Create random subset - 50% - 60%
        # [Kyle]  1. Create many (e.g. 100) random sub-samples of our dataset with replacement.
        # [Tuyet] 2. Train a Random Forest model on each sample.
        # [Jon]   3. Given a new dataset, calculate the average prediction from each model.
    # else:
    #     train_all()
