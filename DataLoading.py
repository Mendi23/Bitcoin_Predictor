import pandas as pd
import numpy as np
import os
import re
import json
from urllib.request import Request, urlopen
import time
from pytrends.request import TrendReq
from datetime import datetime


data_dir = 'Data'

class DataLoader:
    def __init__(self, train_epoch, test_epoch, fillgaps=True):

        self.date_format = '%Y-%m-%d'
        self.to_dt =lambda x: datetime.strptime(x, self.date_format)
        self.start_train, self.end_train = train_epoch
        self.start_test, self.end_test = test_epoch
        self.from_date, self.to_date = self.start_train, self.end_test
        self.timeout=10.0
        self.fillgaps = fillgaps
        
    def load_if_pickled(name):
        def enter_exit_info(func):
            def wrapper(self, *args, **kw):
                t_path = os.path.join(data_dir, name+'_data.pkl')
                if os.path.exists(t_path):
                    data = pd.read_pickle(t_path)
                else:
                    data = func(self, *args, **kw)
                    data.columns = [re.sub(r"[^a-z_]", "", col.lower()) for col in data.columns]
                    data['date'] = pd.to_datetime(data['date'], format=self.date_format)
                    data = data[(data['date']>=self.from_date) & (data['date']<=self.to_date)].reset_index(drop=True)
                    if self.fillgaps:
                        not_zero = data.ne(0).idxmax().max()
                        data = data[not_zero:].fillna(method='ffill').replace(to_replace=0, method='ffill')
                    data.to_pickle(t_path)
                return data[data['date']<=self.end_train], data[data['date']>=self.start_test].reset_index(drop=True)
            return wrapper
        return enter_exit_info

    @load_if_pickled('bitcoin_trading')
    def get_bitcoin_trading_data(self):
        start, end = self.from_date.replace("-", ""), self.to_date.replace("-", "")
        output = pd.read_html(f"https://coinmarketcap.com/currencies/bitcoin/historical-data/?start={start}&end={end}")
        return output[0].iloc[::-1].rename(columns={'Close**':'Price'})


    def get_bitcoin_tweets_data(self):
        pass

    def _parse_bitinfocharts(self, coin, metric):
        col_name = "_".join([coin, metric])
        parsed_page = Request(f"https://bitinfocharts.com/comparison/{coin}-{metric}.html", headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11'})
        parsed_page = urlopen(parsed_page, timeout=self.timeout).read().decode("utf8")
        start_segment = parsed_page.find("new Dygraph")
        
        if start_segment == -1:
            raise ValueError("Could not find the appropriate text tag in the scraped page")
        
        start_list = parsed_page.find("[[", start_segment)
        end_list = parsed_page.find("]]", start_list)
        parsed_page = parsed_page[start_list: end_list]

        rep = {'new Date(': '', ')': '', 'null': '0', '["': '{"date":"', '",': '","{}":'.format(col_name), '],': '},'}
        for  key, val in rep.items():
            parsed_page = parsed_page.replace(key, val)
        parsed_page += '}]'
        return pd.DataFrame(json.loads(parsed_page))

    @load_if_pickled('cryptos_price')
    def get_coins_price_data(self):
        coins = ['eth', 'xrp']
        res = self._parse_bitinfocharts(coins[0], 'price')
        for coin in coins[1:]:
            res = res.merge(self._parse_bitinfocharts(coin, 'price'), on='date')
        return res


    def _parse_finance(self, market):
        from_date = int(time.mktime(time.strptime(self.from_date, "%Y-%m-%d")))
        # this site works off unix time (86400 seconds = 1 day)
        to_date = int(time.mktime(time.strptime(self.to_date, "%Y-%m-%d"))) + 86400
        
        url = f"https://finance.yahoo.com/quote/{market}/history?period1={from_date}&period2={to_date}&interval=1d&filter=history&frequency=1d"
        parsed_page = urlopen(url, timeout=self.timeout).read().decode("utf8")
        start_segment = parsed_page.find('{\"prices\":')
        
        if start_segment == -1:
            raise ValueError("Could not find the appropriate text tag in the scraped page")
        
        start_list = parsed_page.find("[", start_segment)
        end_list = parsed_page.find("]", start_list)
        output = pd.DataFrame(json.loads(parsed_page[start_list: end_list+1]))
        
        output['date'] = pd.to_datetime(output['date'], unit='s').dt.date
        output['date'] = pd.to_datetime(output['date'])
        return output.iloc[::-1]

    def get_dates_range(self):
        return pd.to_datetime(pd.date_range(self.from_date, self.to_date, freq='D'), format=self.date_format).to_frame().rename(columns={0: 'date'}).reset_index(drop=True)

    @load_if_pickled('stocks_price')
    def get_stocks_price_data(self):
        res = self.get_dates_range()
        market_codes = (('Dow Jones','%5EDJI'), ('Nasdaq','%5EIXIC'), ('S&P 500','%5EGSPC'))
        for stock, market in market_codes:
            t = self._parse_finance(market)
            res = res.merge(t[['date', 'close']], on='date', how='left').rename(columns={'close': stock+'_price'})
        return res

    @load_if_pickled('keywords_search')
    def get_keywords_search_data(self):
        kw_list = ['bitcoin', 'cryptocurrency', 'blockchain']
        window_len = 50
        pytrend=TrendReq()
        dates_intervals = pd.Series((pd.date_range(self.from_date, self.to_date,
                                        freq=f"{window_len}D"))).dt.date
        start = dates_intervals.iloc[0]
        dfs = []
        for end in dates_intervals.iloc[1:]:
            time_epoch = f"{start} {end}"
            pytrend.build_payload(kw_list, timeframe=time_epoch)
            dfs.append(pytrend.interest_over_time().drop(columns=['isPartial']))
            start = end

        norm_factors = dfs[0].iloc[-1]
        normilized = []
        for df in dfs:
            df *= (norm_factors / df.iloc[0])
            norm_factors = df.iloc[-1]
            normilized.append(df.iloc[:-1])

        return pd.concat(normilized).reset_index(level=0)

    @load_if_pickled('commodities_price')
    def get_commodities_price_data(self):
        current_year = datetime.now().year
        from_year = self.to_dt(self.from_date).year
        to_year = self.to_dt(self.to_date).year
        output = []
        for i in range(from_year, to_year+1):
            if i==current_year:
                output.append(pd.read_html("http://www.kitco.com/gold.londonfix.html")[-1])
            else:
                output.append(pd.read_html("http://www.kitco.com/londonfix/gold.londonfix"+
                                       str(i)[-2:]+".html")[-1])
        output = pd.concat(output)
        date_col = output.loc[:,output.iloc[1]=='Date'].apply(pd.to_datetime, errors='coerce')
        output = output.loc[:,output.iloc[1].isin(['PM', '-'])]
        output.columns = [c+'_price' for c in output.iloc[0]]
        output = output.replace("-", np.nan).assign(date=date_col)
        return self.get_dates_range().merge(output, on='date', how='left').apply(lambda x: pd.to_numeric(x) if x.name != 'date' else x)