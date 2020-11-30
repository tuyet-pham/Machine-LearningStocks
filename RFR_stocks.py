import pandas as pd
import datetime as dt
import time
import random
import statistics
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
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
                        \ncriterion: {tunedata.iloc[min_value_index, 4]} \
                            \nmin_samples_leaf: {tunedata.iloc[min_value_index, 5]} \
                                \nsplitter: {tunedata.iloc[min_value_index, 6]}')
    
    return min_value_index, min_value, {tunedata.iloc[min_value_index, 1]}


# Tuning 
def Tune(train_set, dev_set, oldforest, tune_cycle, stock_name):

    # Forest Param 
    n_estimator = [5,10,13,15,20,21,23,25,27,30,40,45,55,60,65,70,80,90,100]

    # https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html    
    # Tree Param
    param_dict_tree = {
        "min_samples_leaf": [2,3,4,5,6,7,8], 
        "max_depth": [2,3,4,5,6,7,8,9,10],
        "max_leaf_nodes": [5,7,8,10,12,14,16,20],
        "criterion": ['entropy', 'gini'],
        "splitter": ['random']
    }
    
    X_train = train_set.copy()
    X_train.drop(columns=COLUMNS_TO_DROP, axis=1, inplace=True)
    y_train = train_set[['target %']].values.ravel()
    
    X_dev = dev_set.copy()
    X_dev.drop(columns=COLUMNS_TO_DROP, axis=1, inplace=True)
    y_dev = dev_set[['target %']].values.ravel()

    # print(f'X_train: \n\n{X_train}\ny_train:\n\n{y_train}')
    # print(f'X_dev: \n\n{X_dev}\ny_dev:\n\n{y_dev}')

    
    # Get best tree model
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_models = pd.DataFrame(data=None, columns=['n_estimator', 'max_depth', 'max_leaf_nodes','criterion','min_samples_leaf','splitter','average kfold score', 'MSE', 'baseline MSE',
                                                  'accuracy', "macro avg f1", "macro avg precision", "macro avg recall",
                                                  "baseline accuracy", "baseline f1", "baseline precision", "baseline recall"])
    
    forest_model_list = []
    train_target = train_set['target %']
    dev_target = dev_set['target %']
    baseline_stats = Baseline(train_set, train_target,  dev_set, dev_target)
    baseline = baseline_stats[0]
    baseline_stats = baseline_stats[1:]
        
    for count in range(0, tune_cycle):
        
        tree_model_list = []
        best_param = None
        for i in range(0, 20):
            decisionTree = DecisionTreeClassifier()
            decisionTree.fit(X_train, y_train)
            grid = GridSearchCV(decisionTree,
                                param_grid=param_dict_tree,
                                cv=kf,
                                scoring='accuracy')
            grid.fit(X_train, y_train)
            best_param = grid.best_params_
            tree_model_list.append(best_param)
            
        df = pd.DataFrame(tree_model_list)

        # update class labels
        df.loc[df["criterion"] == "gini","criterion"] = 1
        df.loc[df[ "criterion"] == "entropy","criterion"] = 0
        df["criterion"] = df["criterion"].astype('int')
        crit = df['criterion'].mode()[0]
        
        if crit == 1:
            best_param = {'criterion': 'gini' , 'max_depth': round(df['max_depth'].mean()), 
                        'max_leaf_nodes':round(df['max_leaf_nodes'].mean()), 'min_samples_leaf':round(df['min_samples_leaf'].mean()), 
                        'splitter': 'random'}
        else:
            best_param = {'criterion': 'entropy' , 'max_depth': round(df['max_depth'].mean()), 
                        'max_leaf_nodes':round(df['max_leaf_nodes'].mean()), 'min_samples_leaf':round(df['min_samples_leaf'].mean()), 
                        'splitter': 'random'}

        print(f'\n\nBest params for tuning cycle ({count+1}) overall : {best_param}')

        for n_size in n_estimator:
            
            random_subsets = RandomSubsets(train_set, n_size, 0.4)
            forest = RandomForest(random_subsets, tree_params=best_param)
            
            scores = []
            for tree in forest:
                score = cross_val_score(tree, X_dev, y_dev, cv=kf, scoring="accuracy")
                scores.append(score.mean())
            avg = Average(scores)

            lmse, lbse, f1_stats = [], [], [0, 0, 0, 0]

            num_loops = 20
            for three in range (0, num_loops):
                dev_preds = MakePredictions(forest, dev_set)
                lmse.append(LabeledMSE(dev_preds, y_dev))
                lbse.append(baseline)
                temp_f1_stats = ClassificationEvalStats(dev_preds, y_dev)
                for t_f1 in range(0, len(temp_f1_stats)):
                    f1_stats[t_f1] += temp_f1_stats[t_f1] / num_loops

            lmse = sum(lmse) / num_loops
            lbse = sum(lbse) / num_loops
            
            n_model = {'n_estimator': n_size, 'max_depth': best_param['max_depth'], 'max_leaf_nodes': best_param['max_leaf_nodes'] , 'criterion':best_param['criterion'], 
                    'min_samples_leaf':best_param['min_samples_leaf'], 'splitter':best_param['splitter'], 'average kfold score': avg,
                    'MSE': lmse, 'baseline MSE': lbse, 'accuracy': f1_stats[0],
                    "macro avg f1": f1_stats[1], "macro avg precision": f1_stats[2], "macro avg recall": f1_stats[3],
                    "baseline accuracy": baseline_stats[0], "baseline f1": baseline_stats[1], "baseline precision": baseline_stats[2],
                    "baseline recall": baseline_stats[3]}
            # print(n_model)
            # print(forest)

            all_models = all_models.append(n_model, ignore_index=True)
            forest_model_list.append(forest)

    ts = (f"data/tune/{stock_name}_tune_data.csv")
    all_models.to_csv(ts)
    
    # # Returning best forest here.. will need but will mute for now. Check out the csv to look at the best tree result
    index, score, n_estim = ChoseBest(stock_name)
    return forest_model_list[index], score, n_estim



def Test(forest, stock_name):
    
    custom = custom_features()
    custom.to_csv(f"data/test/{stock_name}_testcustom_data.csv")
    
    test_set = pd.read_csv(f"data/test/{stock_name}_testcustom_data.csv", index_col="date")
    test_set['target %'] = GenerateLabels(test_set['target %'])
    # print(test_set)
    # test_set.drop(columns=COLUMNS_TO_DROP, axis=1, inplace=True)
    y_test = test_set[['target %']].values.ravel()

    test_preds = MakePredictions(forest, test_set)
    
    print(f"Testing model.. {forest[1].get_params()}\n")
    print(f"Predictions from model..\n")

    for i in range(0, len(test_preds)):
        print("pred: {}   actual: {}".format(test_preds[i], y_test[i]))
    
    print("Labeled MSE: ", end="")
    print(LabeledMSE(test_preds, y_test))

    


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


def RandomForest(train_sets, tree_params):
    forest = []
    random.seed(dt.datetime.now().microsecond)
    if tree_params == None:
        for i in range(0, len(train_sets)):
            sub_set = train_sets[i]
            sub_target = sub_set['target %']
            sub_set = sub_set.drop(columns=COLUMNS_TO_DROP)
            rand_num = random.randint(0, 99999999)
            n_tree = DecisionTreeClassifier()
            n_tree.fit(sub_set, sub_target)
            forest.append(n_tree)
    else:
        for i in range(0, len(train_sets)):
            sub_set = train_sets[i]
            sub_target = sub_set['target %']
            sub_set = sub_set.drop(columns=COLUMNS_TO_DROP)
            rand_num = random.randint(0, 99999999)
            n_tree = DecisionTreeClassifier(random_state=rand_num,**tree_params)
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
    base = DecisionTreeClassifier()
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
        ts = (f"data/train/{stock_name}_train_data.csv")
        ds = (f"data/dev/{stock_name}_dev_data.csv")
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
        forest = RandomForest(random_subsets, tree_params=None)
        
        dev_preds = MakePredictions(forest, dev_set)

        ClassificationEvalStats(dev_preds, dev_target)

        # for i in range(0, len(dev_preds)):
        #     print("pred: {}   actual: {}".format(dev_preds[i], dev_target[i]))
        
        print("Labeled MSE: ", end="")
        print(LabeledMSE(dev_preds, dev_target))
        print("Labeled baseline:  ", end="")
        print(Baseline(train_set, train_target, dev_set, dev_target)[0])

        # print(forest)
        
        # tunecount = input("How many tuning cycle do you want? ")
        # bestforest, score, n_estimator = Tune(train_set, dev_set, forest, int(tunecount), stock_name)
        # print(f'\nBest hyperparameter: {bestforest}')
        # print(f'\nBest score: {score}')
        # print(f'\nBest n_estimator: {n_estimator}')

        best_param = {'criterion': 'entropy', 'max_depth': 6, 'max_leaf_nodes': 10, 'min_samples_leaf': 5, 'splitter': 'random'}
        n_estimator = 40
        
        random_subsets = RandomSubsets(train_set, n_estimator, 0.4)
        forest = RandomForest(random_subsets, best_param)
        for tree in forest:
            X_dev = dev_set.copy()
            X_dev.drop(columns=COLUMNS_TO_DROP, axis=1,inplace=True)
            tree.fit(X_dev, dev_target)

        Test(forest, stock_name)

