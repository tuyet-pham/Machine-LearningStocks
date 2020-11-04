import pandas as pd
from sklearn import tree
from numpy import array
import numpy as np
import matplotlib as plot
# import tqdm
import datetime as dt
import time
import os
pd.options.display.max_rows=10


# To get the list of Name values from the 'Name' column from the 
alst = pd.read_csv("data/all_stocks_5yr.csv")
alst = alst['Name'].unique()

# create and open the listall.csv
file1 = open("data/listall.csv","wt")

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

    dev_set.to_csv(ds,index=custom_feat)
    train_set.to_csv(ts,index=custom_feat)
    
    
    time.sleep(.2) # comment this out if you're going to use the tqdm library
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
        print("\nStock good: %s\nUsing 2013-2015 for training and 2016-2018.." %stockname)
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
    
    
    time.sleep(.2) # comment this out if you're going to use the tqdm library
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
        
        print("Can't find %s Try again?" %stkname)

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
    feature_names = ["target %", "close-open", "2 week movement", "Stochastic Oscillator", "Williams %R", "average"]
    for item in stock_data.columns:
        if item != "date":
            feature_names.append(item)
    custom_features = pd.DataFrame(index=stock_data.index, columns=feature_names)
    for i in stock_data.index:
        row = stock_data.loc[i]
        # print(dt.datetime.strptime(i, "%Y-%m-%d") - dt.datetime.strptime(stock_data.iloc[0].name, "%Y-%m-%d"))
        average = (row['high'] + row['low']) / 2
        custom_features.loc[i]['average'] = average
        today = dt.datetime.strptime(i, "%Y-%m-%d")
        for col in stock_data.columns:
            custom_features.loc[i][col] = row[col]
        if (today - dt.datetime.strptime(stock_data.iloc[0].name, "%Y-%m-%d")).days < 14:
            prev_loc = i
            continue
        close_open = row['close'] - row['open']
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
        custom_features.loc[i]["close-open"] = close_open
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


def RandomForest(train_set, n_subset):
    forest = []
    for i in range(0, n_subset):
        # sub_set = Bagging_here(train_set) - Kyle <--- best_features = featureselection(train_set) # Caleb
        n_tree = tree.DecisionTreeRegressor()
        # n_tree.train( )
        forest.add(n_tree)
    return forest


train_set = pd.DataFrame()
dev_set = pd.DataFrame()



if __name__ == "__main__":
    # Run to make life easier
    all_or_one = ""
    all_or_one = input("Train ALL? *Say no for now* [y/n]: ")

    if(all_or_one == "n"):
        stock_name = pick_stock()
        if(stock_name != ""):
            custom = custom_features(stock_name)
            train_set, dev_set = train_dev(custom, dataframe=True)
            print("\nTraining set\n",train_set,"\n\nDevset\n", dev_set)
            print(os.getcwd())
            # !Important - This will create .csv file of each sets.
            # if using custom features, the fourth parameter should be true
            train_dev_file(dev_set, train_set, stock_name, True)
            tr = tree.DecisionTreeRegressor()
            train_target = train_set['target %']
            train_set = train_set.drop(columns=['target %', 'Name'])
            dev_target = dev_set['target %']
            dev_set = dev_set.drop(columns=['target %', 'Name'])
            tr.fit(train_set, train_target)
            preds = tr.predict(dev_set)
            print("Predictions")
            for i in range(0, len(preds)) :
                print("pred: {}   actual: {}".format(preds[i], dev_target[i]))
            print()
            print("DEV SCORE (1 is perfect): ", end="")
            print(tr.score(dev_set, dev_target))
            x = 1

        # Below does not all need to be done by next Thursday
          # We need is to figure out what our milestones should be.
          # Overall concepts
          # How it ties all together
          # GUI? 

        # [Caleb] Figure out what features are significant
          # use pearsons correlation (Machine Learning HW2)
        
        # Create random subset - 50% - 60%
          # [Kyle]  1. Create many (e.g. 100) random sub-samples of our dataset with replacement.
          # [Tuyet] 2. Train a Random Forest model on each sample.
          # [Jon]   3. Given a new dataset, calculate the average prediction from each model.
    # else:
    #     train_all()