import pandas as pd
from sklearn import tree
import numpy as np
import matplotlib as plot
# import tqdm
# import time
pd.options.display.max_rows=10


# To get the list of names
file1 = open("data/listall.csv","w+") 
alst = pd.read_csv("data/all_stocks_5yr.csv")
alst = alst['Name'].unique()

file1.writelines("Name\n")
for e in alst:
    file1.writelines(e+"\n")
file1.close()
file1 = pd.read_csv('data/listall.csv')


def train_dev(name):
    
    """[summary]
        Getting a tuple from the file - train and dev set of a specific stock
    Args:
        name (string): 
    Returns:
        [tuples]: train_stock, dev_stock
    """
    filepath = 'data/individual_stocks_5yr/' + name + '_data.csv'
    df = pd.read_csv(filepath)
    
    # Getting within the time
    train_stock = df[(df['date'] >= '2012-12-31') & (df['date'] <= '2016-01-01')]
    dev_stock = df[(df['date'] >= '2016-01-02') & (df['date'] <= '2019-01-01')]
    
    # Just Printing timer for extra coolness
    print("\nStock good: %s\nUsing: \n2013-2015 for training... \n2016-2018 for Dev.." %name)
    # for i in tqdm.tqdm(range(200)):
    #     time.sleep(0.01)

    return train_stock, dev_stock



# if we want to do this later I guess
def train_all():
    # for i in tqdm.tqdm(range(200)):
    #     time.sleep(0.01)
    print("Running ALL... might be a minute")



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
        stkname = input("Stock Name? ").upper()              
        
        # finding name
        for e in file1['Name']:                             
            if e == stkname:
                return stkname

        if stkname == "QUIT":
            break
        print("Can't find %s Try again?" %stkname)

    return stkname


train_set = pd.DataFrame()
dev_set = pd.DataFrame()

if __name__ == "__main__":
    all_or_one = ""
    all_or_one = input("Train ALL? *Say no for now* [y/n]: ")

    if(all_or_one == "n"):
        
        stock_name = pick_stock()
        if(stock_name != ""):
            train_set, dev_set = train_dev(stock_name)
        
        print("\n\nTraing set\n",train_set,"\n\nDevset\n", dev_set)

else:
    train_all()


