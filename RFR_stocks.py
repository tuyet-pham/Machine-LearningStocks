
import pandas as pd
import datetime as dt
import time
import random
import statistics
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn import tree
pd.options.display.max_rows = 20


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


# Remove all but tech stocks.
def GettingTechStocks(pathname):
    
    filall = pd.read_csv(pathname)
    filetech = pd.DataFrame()
    
    LISTOFTECH = ['AAPL', 'ACN', 'ADBE', 'ADI', 'ADP', 'ADSK', 'AKAM', 'AMAT', 'AMD',
        'ANET', 'ANSS', 'APH', 'AVGO', 'BR', 'CDNS', 'CDW', 'CRM', 'CSCO', 'CTSH', 'CTXS', 'DXC', 'FFIV',
            'FIS', 'FISV', 'FLIR', 'FLT', 'FTNT', 'GLW', 'GPN', 'HPE', 'HPQ', 'IBM', 'INTC', 'INTU', 'IPGP', 'IT',
                'JKHY', 'JNPR', 'KEYS', 'KLAC', 'LDOS', 'LRCX', 'MA', 'MCHP', 'MSFT', 'MSI', 'MU', 'MXIM', 'NLOK', 'NOW', 'NTAP', 'NVDA', 'ORCL',
                    'PAYC', 'PAYX', 'PYPL', 'QCOM', 'QRVO', 'SNPS', 'STX', 'SWKS', 'TEL', 'TER', 'TXN', 'TYL', 'V', 'VNT', 'VRSN', 'WDC', 'WU', 'XLNX', 'XRX', 'ZBRA']
    
    filetech["Name"] = LISTOFTECH
    filetech.to_csv("data/listTech.csv", index=False)

    # open the listall.csv and compare
    onlytech = filetech['Name'].isin(filall['Name'])
    
    filetech["isin_listall"] = onlytech
    filetech.to_csv("data/listTech.csv", index=False)

    # print(file2)

#Getting the csv for Techstocks only and seeing if it exists in our current listall
GettingTechStocks("data/listall.csv")
    


def train_dev_file(dev_set, train_set, stock_name, custom_feat=False):

    ts = (r"data/train/" + stock_name + "_train_data.csv")
    ds = (r"data/dev/" + stock_name + "_dev_data.csv")

    dev_set.to_csv(ds, index=custom_feat)
    train_set.to_csv(ts, index=custom_feat)

    time.sleep(.2)
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

    time.sleep(.2)
    print("Train and Dev set creation - Done")
    return train_stock, dev_stock



# Picking an individual stock
def pick_stock():
        
    # Open listall
    try:
        file1 = pd.read_csv('data/listall.csv')
    except FileExistsError:
        print('\nFile doesnt exist.')
        return ""

    # Open listall
    try:
        file2 = pd.read_csv('data/listTech.csv')
    except FileExistsError:
        print('\nFile doesnt exist.')
        return ""

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

        if (file2['Name'] == stkname).any() and (file1['Name'] == stkname).any():
            return stkname

        print("Can't find %s Try again?" % stkname)

    return ""


def Average(lst): 
    return sum(lst) / len(lst)


def ChoseBest(stock_name):
    try:
        tunedata = pd.read_csv(f'data/tune/{stock_name}_tune_data.csv')
    except FileExistsError:
        print('\nFile doesnt exist.')
        return None
    
    lsme_col = tunedata['MSE']
    min_value = lsme_col.min()
    min_value_index = lsme_col.idxmin()

    print(f'\nIndex: {min_value_index} \
        \nMinimum Labeled MSE: {min_value} \
            \nn_estimator: {tunedata.iloc[min_value_index, 1]} \
                \nmax_depth: {tunedata.iloc[min_value_index, 2]} \
                    \nmax_leaf_node: {tunedata.iloc[min_value_index, 3]} \
                        \naverage kfold score: {tunedata.iloc[min_value_index, 4]} \
                            \nLabeled baseline: {tunedata.iloc[min_value_index, 6]}')
    
    return min_value_index, min_value


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

    
    max_depth = [4,6,8,10]
    max_leaf_nodes = [5,10, 20,30,40]
    n_estimator = [5, 10, 20, 40, 60, 80]
    all_models = pd.DataFrame(data=None, columns=['n_estimator', 'max_depth', 'max_leaf_node', 'average kfold score', 'MSE', 'baseline MSE',
                                                  'accuracy', "macro avg f1", "macro avg precision", "macro avg recall",
                                                  "baseline accuracy", "baseline f1", "baseline precision", "baseline recall"])
    model_list = []
    train_target = train_set['target %']
    dev_target = dev_set['target %']
    baseline_stats = Baseline(train_set, train_target,  dev_set, dev_target)
    baseline = baseline_stats[0]
    baseline_stats = baseline_stats[1:]
    
    
    for val in n_estimator:
        for i in max_depth:
            for j in max_leaf_nodes:
                
                random_subsets = RandomSubsets(train_set, val, 0.4)
                forest = RandomForest(random_subsets, mxdepth=i, mx_leaf_nodes=j)

                scores = []
                for tree in forest:
                    score = cross_val_score(tree, X, y, cv=kf, scoring="accuracy")
                    scores.append(score.mean())
                avg = Average(scores)

                lmse, lbse, f1_stats = [], [], [0, 0, 0, 0]

                num_loops = 20
                for three in range (0, num_loops):
                    dev_preds = MakePredictions(forest, dev_set)
                    lmse.append(LabeledMSE(dev_preds, dev_target))
                    lbse.append(baseline)
                    temp_f1_stats = ClassificationEvalStats(dev_preds, dev_target)
                    for t_f1 in range(0, len(temp_f1_stats)):
                        f1_stats[t_f1] += temp_f1_stats[t_f1] / num_loops

                lmse = sum(lmse) / num_loops
                lbse = sum(lbse) / num_loops

                n_model = {'n_estimator': val, 'max_depth': i, 'max_leaf_node': j, 'average kfold score': avg,
                           'MSE': lmse, 'baseline MSE': lbse, 'accuracy': f1_stats[0],
                           "macro avg f1": f1_stats[1], "macro avg precision": f1_stats[2], "macro avg recall": f1_stats[3],
                           "baseline accuracy": baseline_stats[0], "baseline f1": baseline_stats[1], "baseline precision": baseline_stats[2],
                           "baseline recall": baseline_stats[3]}
                print(n_model)
                all_models = all_models.append(n_model, ignore_index=True)
                model_list.append(forest)
                
    ts = (f"data/tune/{stock_name}_tune_data.csv")
    all_models.to_csv(ts)
    
    # Returning best forest here.. will need but will mute for now. Check out the csv to look at the best tree result
    index, score = ChoseBest(stock_name)
    return model_list[index], score


# build custom_features dataframe for a single stock
def custom_features(stock_input=None):

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


def ClassificationEvalStats(preds, target):
    if len(preds) != len(target):
        return
    correct = 0
    total = 0

    def blank_stats():
        return {
            "num": 0,
            "TP": 0,
            "FP": 0,
            "FN": 0
        }

    stats = {}
    for p in preds:
        stats[p] = blank_stats()
    for p in target:
        if p not in stats:
            stats[p] = blank_stats()
    for i in range(0, len(preds)):
        stats[target[i]]["num"] += 1
        total += 1
        if preds[i] == target[i]:
            stats[preds[i]]["TP"] += 1
            correct += 1
        else:
            stats[preds[i]]["FP"] += 1
            stats[target[i]]["FN"] += 1

    accuracy = correct / total
    calcs = {}
    for s in stats:
        temp = stats[s]
        if temp["TP"] == 0:
            precision, recall, f1 = 0, 0, 0
        else:
            precision = temp["TP"] / (temp["TP"] + temp["FP"])
            recall = temp["TP"] / (temp["TP"] + temp["FN"])
        calcs[s] = [temp["num"] / total, precision, recall]
    macro_precision, macro_recall = 0, 0
    for s in calcs:
        macro_precision += calcs[s][0] * calcs[s][1]
        macro_recall += calcs[s][0] * calcs[s][2]
    macro_f1 = 2 * (1 / ((1 / macro_precision) + (1 / macro_recall)))
    return accuracy, macro_f1, macro_precision, macro_recall


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


def Baseline(train_set, train_target, dev_set, dev_target):
    base = tree.DecisionTreeClassifier()
    train_set = train_set.drop(columns=COLUMNS_TO_DROP)
    dev_set = dev_set.drop(columns=COLUMNS_TO_DROP)
    base.fit(train_set, train_target)
    dev_preds = base.predict(dev_set)
    '''for dat in target:
        if type(dat) == str:
            baseline.append("hold")
        else:
            baseline.append(0)'''
    stats = ClassificationEvalStats(dev_preds, dev_target)
    return LabeledMSE(dev_preds, dev_target), stats[0], stats[1], stats[2], stats[3]


train_set = pd.DataFrame()
dev_set = pd.DataFrame()

# FOR TUNING
# unhelpful columns
COLUMNS_TO_DROP = ['target %', 'Name', 'open', 'low', 'close', 'high', 'average']
# cutoff between buy, sell, hold
CUTOFF = 0.75

if __name__ == "__main__":
    stock_name = pick_stock()
    if stock_name != "":
        ts = (r"data/train/" + stock_name + "_train_data.csv")
        ds = (r"data/dev/" + stock_name + "_dev_data.csv")
        # noinspection PyBroadException
        try:
            train_set = pd.read_csv(ts, index_col="date")
            dev_set = pd.read_csv(ds, index_col="date")
        except FileNotFoundError:
            custom = custom_features(stock_name)
            train_set, dev_set = train_dev(custom, dataframe=True)
        #print("\nTraining set\n", train_set, "\n\nDevset\n", dev_set)
        
        # !Important - This will create .csv file of each sets.
        # if using custom features, the fourth parameter should be true
        train_dev_file(dev_set, train_set, stock_name, True)

        # replacing target data with labels
        train_set['target %'] = GenerateLabels(train_set['target %'])
        dev_set['target %'] = GenerateLabels(dev_set['target %'])

        train_target = train_set['target %']
        dev_target = dev_set['target %']

        random_subsets = RandomSubsets(train_set, 40, 0.4)
        forest = RandomForest(random_subsets, mxdepth=4, mx_leaf_nodes=20)
        
        dev_preds = MakePredictions(forest, dev_set)

        ClassificationEvalStats(dev_preds, dev_target)

        '''for i in range(0, len(dev_preds)):
            print("pred: {}   actual: {}".format(dev_preds[i], dev_target[i]))
        
        print("Labeled MSE: ", end="")
        print(LabeledMSE(dev_preds, dev_target))
        print("Labeled baseline:  ", end="")
        print(Baseline(train_set, train_target, dev_set, dev_target)[0])'''

        typesector = "listTech"
        # tunning might take a while since we didnt use the randomforest class from sklearn
        df = pd.read_csv(f"data/{typesector}.csv")
        nc = df['Name']   
        cc = df['isin_listall']
        s = len(df)
        avg = []
        forestlist = []
        i = 0
        while i < 10:
            rand_index = random.randint(0, s)
            if cc[rand_index]:
                print(f"\nTuning ========== \nRandom index : {rand_index}\nStock Name: {nc[rand_index]}")
                forest, score = Tune(train_set, dev_set, forest, nc[rand_index])
                avg.append(score)
                forestlist.append(forest)
                i=i+1

        print(f'The Average Tuning score of 10 stocks is {Average(avg)}')
        for i in forestlist:
            print(i[:])
