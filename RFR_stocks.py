import pandas as pd
from sklearn import tree
from numpy import array
import numpy as np
import matplotlib as plot
# import tqdm
import time
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


def train_dev_file(dev_set, train_set, stock_name):
    # for i in tqdm.tqdm(range(200)):
    #     time.sleep(0.01)

    ts = (r"data/train/" + stock_name + "_train_data.csv")
    ds = (r"data/dev/" + stock_name + "_dev_data.csv")
    
    dev_set.to_csv(ds,index=False)
    train_set.to_csv(ts,index=False)
    
    
    time.sleep(.2) # comment this out if you're going to use the tqdm library
    print("Train and Dev set file creation - Done")



# Getting the a tuple from the file - train and dev set
def train_dev(stockname):
    
    """[summary]
        Getting a tuple from the file - train and dev set of a specific stock
    Args:
        stockname (string):
    Returns:
        [tuples]: train_stock, dev_stock
    """
    print("\nStock good: %s\nUsing 2013-2015 for training and 2016-2018.." %stockname)
    filepath = 'data/individual_stocks_5yr/' + stockname + '_data.csv'
    df = pd.read_csv(filepath)

    # Getting within the time
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
            train_set, dev_set = train_dev(stock_name)
            print("\nTraining set\n",train_set,"\n\nDevset\n", dev_set)
            
            # !Important - This will create .csv file of each sets.
            train_dev_file(dev_set, train_set, stock_name)
            
            
        
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


