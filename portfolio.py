import math
import pandas as pd
from db import *
from broker import Borker
from ib_insync import *
from currency_converter import CurrencyConverter
from stock import Stock2, Stock3


class Portfolio:

    def __init__(self):
        self.portfolio = self.getPortfoilio()
        self.summary = self.getAccountSummary()
        self.cash = float(self.summary['value'].iloc[1])
        self.stockMktV = float(self.summary['value'].iloc[4])
        self.totalValue = self.cash + self.stockMktV

    def getPortfoilio(self):
        return Borker().getPortfolio()

    def getAccountSummary(self):
        summary = Borker().getAccountSummary()
        summary = summary[summary['currency'] == 'BASE'].reset_index(drop=True)
        del summary['currency']
        return summary

    def portfolioRatio(self):
        pfolio = self.portfolio[['Ticker', 'Currency', 'mktValue']]
        ratio = pd.DataFrame(columns=['asset', 'value', 'pct'])
        for i in pfolio.values:
            ratio = ratio.append({'asset': i[0], 'value': CurrencyConverter().convert(i[2], i[1], 'EUR'),
                                  'pct': (CurrencyConverter().convert(i[2], i[1], 'EUR') / self.totalValue) * 100},
                                 ignore_index=True)
        ratio = ratio.append({'asset': 'cash', 'value': self.cash, 'pct': (self.cash / self.totalValue) * 100},
                             ignore_index=True)
        return ratio

    def portfolioStockProfile(self, n):
        df = pd.DataFrame(
            columns=['ticker', 'roe', 'roeSTD', 'opm', 'opmSTD', 'gpm', 'gpmSTD', 'gm', 'gmSTD', 'fcf', 'fcfSTD'])

        for i in self.portfolio["Ticker"]:
            print("***************")
            print(i)
            s = Stock2(i, n)
            df = df.append({'ticker': i, 'roe': s.roeProfile()['increasing'], 'roeSTD': s.roeProfile()['std'],
                            'opm': s.opMarginProfile()['increasing'], 'opmSTD': s.opMarginProfile()['std'],
                            'gpm': s.gProfitMarginProfile()['increasing'], 'gpmSTD': s.gProfitMarginProfile()['std'],
                            'gm': s.gMarginProfile()['increasing'], 'gmSTD': s.gMarginProfile()['std'],
                            'fcf': s.fcfProfile()[0]['increasing'], 'fcfSTD': s.fcfProfile()[0]['std']},
                            ignore_index=True)

            pdtodb(df, "portfolio_stock_profile_{n}".format(n=n))



class TestPorfolio:

    def __init__(self):
        self.portfolio = dbtopd("portfolio")
        self.cash = dbtopd("cash").values[0][0]

    def addStock(self, s):
        cash = self.cash - s.cost
        cash = pd.Series(cash)
        seriesToDb(cash, "cash")
        portfolio = self.portfolio.append(s.portfolioDict(), ignore_index=True)
        pdtodb(portfolio,"portfolio")
        print("bought ", s.cost, " worth of ", s.ticker, " stock")

    def updatePortfolio(self, n):
        portfolio = copy(self.portfolio)
        for i in portfolio.index:
            portfolio.at[i,"mktPrice"] = Stock2(portfolio.loc[i]["ticker"],n).price
            portfolio.at[i, "mktValue"] = Stock2(portfolio.loc[i]["ticker"],n).price * portfolio.iloc[i]["shares"]
        print(portfolio)
        pdtodb(portfolio, "portfolio")
        return portfolio

    def deleteStock(self, s,n):
        for i in range(len(self.portfolio)):
            print(i)
        cash = self.cash + Stock2(s.ticker, n).price * s.shares
        cash = pd.Series(cash)
        seriesToDb(cash, "cash")
        portfolio = self.portfolio
        portfolio = portfolio[portfolio["ticker"] != s.ticker]
        pdtodb(portfolio, "portfolio")

    def dividend(self, n):
        count = 0
        for i in self.portfolio.index:
            if math.isnan(Stock2(self.portfolio.loc[i]["ticker"], n).dividendPerShare()):
                count = count
            else:
                count = count + Stock2(self.portfolio.loc[i]["ticker"],n).dividendPerShare() * \
                        self.portfolio.loc[i]["shares"]
        cash = self.cash + count
        cash = pd.Series(cash)
        seriesToDb(cash, "cash")

    def portfolioStockProfile(self, n):
        df = pd.DataFrame(columns=['ticker','roe','roeSTD','opm','opmSTD','gpm','gpmSTD','gm','gmSTD','fcf','fcfSTD'])
        for i in self.portfolio["ticker"]:
            s = Stock2(i, n)
            df = df.append({'ticker':i,'roe':s.roeProfile()['increasing'],'roeSTD':s.roeProfile()['std'],
                            'opm':s.opMarginProfile()['increasing'],'opmSTD':s.opMarginProfile()['std'],
                            'gpm':s.gProfitMarginProfile()['increasing'],'gpmSTD':s.gProfitMarginProfile()['std'],
                            'gm':s.gMarginProfile()['increasing'],'gmSTD':s.gMarginProfile()['std'],
                            'fcf':s.fcfProfile()[0]['increasing'],'fcfSTD':s.fcfProfile()[0]['std']},ignore_index=True)

        pdtodb(df,"portfolio_stock_profile_{n}".format(n=n))

    def accountValue(self,n):
        seriesToDb(pd.Series(self.cash), "cash_{x}".format(x=n))
        seriesToDb(pd.Series(sum(self.portfolio['mktValue'])), "pfValue_{x}".format(x=n))


portfolio = Portfolio().portfolio
cash = Portfolio().cash
print()
