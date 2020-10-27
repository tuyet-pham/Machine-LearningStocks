# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import alpaca_trade_api as tradeapi
import os
import datetime as dt
import pandas as pd

os.environ["APCA_API_BASE_URL"] = "https://paper-api.alpaca.markets"
os.environ["APCA_API_KEY_ID"] = "PKFDHI9VWOO7KD6ESV57"
os.environ["APCA_API_SECRET_KEY"] = "HmyA4IeqX5QVRzkes3u5G1I5Zzm14RT5myYGe9ya"

api = tradeapi.REST()

def print_hi(name):
    import time
    symbols = pd.read_csv("russell-3000-index-09-10-2020.csv")
    symbols = symbols.iloc[:, 0]
    barsets = {}
    i = 0
    for symb in symbols:
        try:
            barsets[symb] = api.get_barset(symb, 'day', limit=200)[symb]
            print(symb)
        except:
            print(symb, " not found")
        i += 1
        if i % 5 == 0:
            time.sleep(2)
    data = {
        "date": [],
        "close": [],
        "open": [],
        "high": [],
        "low": [],
        "volume": [],
        "symbol": []
    }

    # See how much AAPL moved in that timeframe.
    time = dt.datetime.now()
    time -= dt.timedelta(days=200)
    for symb in barsets:
        if len(barsets[symb]) != 200:
            continue
        for bar in barsets[symb]:
            data["date"].append(str(bar.t).split(' ')[0])
            # c,h,l,o,v
            data['close'].append(bar.c)
            data['high'].append(bar.h)
            data['low'].append(bar.l)
            data['open'].append(bar.o)
            data['volume'].append(bar.v)
            data['symbol'].append(symb)
            #print(bar)
    frame = pd.DataFrame(data=data.values(), index=data.keys())
    frame = frame.transpose()
    print(frame)
    frame.to_csv("market_data.csv")
    print("Done")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
