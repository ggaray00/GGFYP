from datetime import date, time
import sche
import schedule

from db import *
from evaluation import reviewTest, porfolio_check, potential_buy, porfolio_checkTest, potential_buyTest, review
from filter import backTestStocks, testfilterr, stocks, filterStocks
from portfolio import TestPorfolio
from stockAPI import Stock
import pandas as pd


def filterTestStocks(n):
    for i in dbtopd("backTestStocks")['tickers'].values:
        testfilterr(i, n)

def filter():
    for i in dbtopd("stocks")['tickers'].values:
        filterStocks(i)

def fsToDb():
    for i in dbtopd("interesting_stocks")['ticker'].values:
        stock = Stock(i, date.today().year)
        try:
            addFStatementsA(stock)
            print(i, "will be considered")
        except OSError:
            print("ERROR")

def backTestStocksDemo():
    df = pd.DataFrame(columns=['tickers'])
    stocks = ["WGO","AAPL"]
    for i in stocks:
        df = df.append({'tickers': i}, ignore_index=True)
    pdtodb(df, "backTestStocks")

def mainTest(n):
    dropTable("viewed_stocks2")
    dropTable("interesting_stocks")
    # backTestStocks()
    backTestStocksDemo()
    filterTestStocks(n)
    try:
        fsToDb()
        reviewTest(n)
    except ValueError:
        print("No interesting stocks")
    TestPorfolio().dividend(n)
    porfolio_checkTest(n)
    TestPorfolio().updatePortfolio(n)
    potential_buyTest(n)
    TestPorfolio().accountValue(n)

pdtodb(pd.DataFrame(columns=["ticker", "cost", "shares", "mktPrice", "mktValue","year"]), "portfolio")
seriesToDb(pd.Series(1000), "cash")
for i in range(2002, 2010):
    print("*******************************")
    print("year: ", i)
    print("cash available:")
    print(TestPorfolio().cash)
    mainTest(i)
    print()

def main():
    dropTable("viewed_stocks2")
    dropTable("interesting_stocks")
    stocks()
    filter()
    try:
        fsToDb()
        reviewTest(date.today().year)
    except ValueError:
        print("No interesting stocks")
    porfolio_check(date.today().year)
    potential_buy(date.today().year)

schedule.every(365).days.do(main)


while True:
    schedule.run_pending()
    time.sleep(1)
