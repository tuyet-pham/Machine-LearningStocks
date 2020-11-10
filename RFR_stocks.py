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
import statistics
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn import linear_model, tree, ensemble

pd.options.display.max_rows=20


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



def rmse(score):
    """[summary]
        Root Mean Square Error (RMSE) is the standard deviation of the residuals (prediction errors). 
        Residuals are a measure of how far from the regression line data points are; 
        RMSE is a measure of how spread out these residuals are. In other words, it tells you how concentrated the data is around the line of best fit. 
        Root mean square error is commonly used in climatology, forecasting, and regression analysis to verify experimental results.
        
        Ref : https://www.statisticshowto.com/probability-and-statistics/regression-analysis/rmse-root-mean-square-error/
    Args:
        score (narray): cross validation score of each split
    """

    rmse = np.sqrt(-score)
    print(f'rmse = {"{:.2f}".format(rmse)}')
    return rmse

def Average(lst): 
    return sum(lst) / len(lst)



def ChoseBest(stock_name):
    try:
        tunedata = pd.read_csv(f'data/tune/{stock_name}_tune_data.csv')
    except FileExistsError:
        print('\nFile doesnt exist.')
        return None
    
    lsme_col = tunedata['labeled MSE']
    min_value = lsme_col.min()
    min_value_index = lsme_col.idxmin()

    print(f'\nIndex: {min_value_index} \
        \nMinimum Labeled MSE: {min_value} \
            \nn_estimator: {tunedata.iloc[min_value_index, 1]} \
                \nmax_depth: {tunedata.iloc[min_value_index, 2]} \
                    \nmax_leaf_node: {tunedata.iloc[min_value_index, 3]} \
                        \naverage kfold score: {tunedata.iloc[min_value_index, 4]} \
                            \nLabeled baseline: {tunedata.iloc[min_value_index, 6]}')
    
    return min_value_index


# Tuning 
def Tune(train_set, dev_set, oldforest, stock_name):
    
    d = train_set.copy()
    y = train_set[['target %']].values.ravel()
    
    d.drop(columns=COLUMNS_TO_DROP, axis=1, inplace=True)                        # Removing target variable from training data
    X = d.copy()
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores_current = []
    treenum = 1

    for model in oldforest:
        score = cross_val_score(model, X, y, cv=kf, scoring="accuracy")
        scores_current.append(score.mean())
        treenum=treenum+1
    
    print(f'\nAverage score for our current the forest for each fold: {"{:.2f}".format(Average(scores_current))}')

    
    max_depth = [4,6,8,9]
    max_leaf_nodes = [5,10, 20,30,40]
    n_estimator = [5, 10, 20, 40, 60, 80]
    all_models = pd.DataFrame(data=None, columns=['n_estimator', 'max_depth', 'max_leaf_node', 'average kfold score', 'labeled MSE', 'Labeled baseline'])
    model_list = []
    
    for val in n_estimator:
        for i in max_depth:
            for j in max_leaf_nodes:
                
                dev_target = dev_set['target %']
                
                random_subsets = RandomSubsets(train_set, val, 0.4)
                forest = RandomForest(random_subsets, mxdepth=i, mx_leaf_nodes=j)

                scores = []
                for tree in forest:
                    score = cross_val_score(tree, X, y, cv=kf, scoring="accuracy")
                    scores.append(score.mean())
                avg = Average(scores)

                lmse, lbse = [], []

                for three in range (0, 3):
                    dev_preds = MakePredictions(forest, dev_set)
                    lmse.append(LabeledMSE(dev_preds, dev_target))
                    lbse.append(Baseline(dev_target))
                lmse = sum(lmse) / 3
                lbse = sum(lbse) / 3

                n_model = {'n_estimator': val, 'max_depth': i, 'max_leaf_node':j, 'average kfold score':avg, 'labeled MSE':lmse, 'Labeled baseline':lbse}
                print(n_model)
                all_models = all_models.append(n_model, ignore_index=True)
                model_list.append(forest)
                
    ts = (f"data/tune/{stock_name}_tune_data.csv")
    all_models.to_csv(ts)
    
    # Returning best forest here.. will need but will mute for now. Check out the csv to look at the best tree result
    # index = ChoseBest(stock_name)
    # return model_list[index]
    



def custom_features(stock_input=None):
    # build custom_features dataframe for a single stock

    # file_name = "data/individual_stocks_5yr/AAPL_data.csv"
    if not stock_input:
        file_name = input("enter stock file name (individual stock): ")
    else:
        file_name = "data/individual_stocks_5yr/" + stock_input + "_data.csv"
    print(file_name)

    stock_data = pd.read_csv(file_name, index_col="date")
    feature_names = ["target %", "close-open %", "2 week movement", "1 week movement", "Stochastic Oscillator", "Williams %R", "average", "volatility 2w", "volatility 1w"]

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
        two_week_change = (average - custom_features.loc[fourteen_days_ago]['average'])
        two_week_change *= (100.0 / custom_features.loc[fourteen_days_ago]['average'])
        last_two_weeks = custom_features.loc[fourteen_days_ago:i]
        one_week_ago = last_two_weeks.index[int(len(last_two_weeks) / 2)]
        last_week = last_two_weeks.loc[one_week_ago:]
        week_change = average - last_two_weeks.loc[one_week_ago]['average']
        week_change *= (100.0 / last_two_weeks.loc[one_week_ago]['average'])

        stochastic = 100 * float(
            (row['close'] - min(last_two_weeks['low'])) / (max(last_two_weeks['high']) - min(last_two_weeks['low'])))
        williams = -100 * float(
            (max(last_two_weeks['high']) - row['close']) / (max(last_two_weeks['high']) - min(last_two_weeks['low'])))
        # feature_names = ["close-open", "2 week movement", "Stochastic Oscillator", "Williams %R","average"]
        custom_features.loc[i]["close-open %"] = close_open
        custom_features.loc[i]["2 week movement"] = two_week_change
        custom_features.loc[i]["1 week movement"] = week_change

        custom_features.loc[i]["Stochastic Oscillator"] = stochastic
        custom_features.loc[i]["Williams %R"] = williams
        custom_features.loc[i]["volatility 2w"] = statistics.stdev(last_two_weeks['average'])
        custom_features.loc[i]["volatility 1w"] = statistics.stdev(last_week['average'])
        target = average - custom_features.loc[prev_loc]['average']
        target = 100 * target / custom_features.loc[prev_loc]['average']
        custom_features.loc[prev_loc]["target %"] = target
        prev_loc = i

    custom_features = custom_features.drop(custom_features.index[0:10], axis=0)
    custom_features = custom_features.drop(custom_features.index[len(custom_features.index) - 1], axis=0)
    # custom_features.to_csv(file_name.split('.')[0] + '_custom_features.csv')
    return custom_features


def RandomForest(train_sets, mxdepth, mx_leaf_nodes):
    forest = []
    random.seed(dt.datetime.now().microsecond)
    for i in range(0, len(train_sets)):
        sub_set = train_sets[i]
        sub_target = sub_set['target %']
        sub_set = sub_set.drop(columns=COLUMNS_TO_DROP)
        rand_num = random.randint(0, 99999999)
        n_tree = tree.DecisionTreeClassifier(random_state=rand_num, min_samples_leaf=3, max_depth=mxdepth, max_leaf_nodes=mx_leaf_nodes, criterion='entropy', splitter='random')
        # print(f'Random state : {rand_num}')
        n_tree.fit(sub_set, sub_target)
        forest.append(n_tree)
    return forest


def MakePredictions(forest, set):
    set = set.drop(columns=COLUMNS_TO_DROP)
    pred_set = []
    final_preds = []
    for n_tree in forest:
        pred_set.append(n_tree.predict(set))
    for i in range(0, len(pred_set[0])):
        votes = {"hold": 0}
        for pred in pred_set:
            if pred[i] not in votes:
                votes[pred[i]] = 1
            else:
                votes[pred[i]] += 1
        mode = ["hold", votes["hold"]]
        for v in votes:
            if votes[v] > mode[1]:
                mode = [v, votes[v]]
        final_preds.append(mode[0])
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
        results.append(1 if data[i] == "buy" else 0 if data[i] == "hold" else -1)
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
COLUMNS_TO_DROP = ['target %', 'Name', 'open', 'low', 'close', 'high', 'average']
# cutoff between buy, sell, hold
CUTOFF = 0.75

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

            # replacing target data with labels
            train_set['target %'] = GenerateLabels(train_set['target %'])
            dev_set['target %'] = GenerateLabels(dev_set['target %'])

            dev_target = dev_set['target %']

            random_subsets = RandomSubsets(train_set, 40, 0.4)
            forest = RandomForest(random_subsets, mxdepth=4, mx_leaf_nodes=20)
            
            dev_preds = MakePredictions(forest, dev_set)
            
            for i in range(0, len(dev_preds)):
                print("pred: {}   actual: {}".format(dev_preds[i], dev_target[i]))
            
            print("Labeled MSE: ", end="")
            print(LabeledMSE(dev_preds, dev_target))
            print("Labeled baseline:  ", end="")
            print(Baseline(dev_target))


            # tunning might take a while since we didnt use the randomforest class from sklearn
            Tune(train_set, dev_set, forest, stock_name)
            ChoseBest(stock_name)


        # [Caleb] Figure out what features are significant
        # use pearsons correlation (Machine Learning HW2)

        # Create random subset - 50% - 60%
        # [Kyle]  1. Create many (e.g. 100) random sub-samples of our dataset with replacement.
        # [Tuyet] 2. Train a Random Forest model on each sample.
        # [Jon]   3. Given a new dataset, calculate the average prediction from each model.
    # else:
    #     train_all()
