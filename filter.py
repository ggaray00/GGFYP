from datetime import date
from broker import Borker
from db import*
from stockAPI import Stock


def backTestStocks():
    companies = fa.available_companies(api_key)
    companies = companies[(companies['type'] == "stock")]
    # companies = companies[(companies['exchange'] == "NYSE")]
    # companies = companies[(companies['exchange'] == "NASDAQ")]
    companies.reset_index(inplace=True)
    df = pd.DataFrame(columns=['tickers'])
    # count = 20
    for i in companies['symbol']:
        try:
            if Stock(i, date.today().year - 1).age >= 30:
                df = df.append({'tickers': i}, ignore_index=True)
                # count = count - 1
        except AttributeError:
            print("AttributeError")
        # if count == 0:
        #     break
    pdtodb(df, "backTestStocks")


def stocks():
    companies = fa.available_companies(api_key)
    companies = companies[(companies['type'] == "stock")]
    companies.reset_index(inplace=True)
    df = pd.DataFrame(columns=['tickers'])
    for i in companies['symbol']:
        try:
            if Stock(i, date.today().year - 1).age >= 10:
                df = df.append({'tickers': i}, ignore_index=True)
        except AttributeError:
            print("AttributeError")
    pdtodb(df, "stocks")


def checkPbR(s):
    a = False
    try:
        a = s.ratios["priceToBookRatio"][0] < 15
    except KeyError:
        print("error")
    return a

def checkPsR(s):
    return s.profile["mktCap"][0] / s.cashFlow["netIncome"][0] < 18
def checkBroker(s,b):
    return b.whatIfOrder2(s.stTicker(),s.currency) != []
def checkIv(s):
    try:
        return max(s.iv(), s.dCF()) > int(s.mktCap)
    except KeyError:
        return False
def checkROE(s):
    return s.roeProfile()

def filterStocks(i):
    stock = Stock(i,date.today().year-1)
    try:
        if checkIv(stock):
            if checkROE(stock):
                if checkPbR(stock) or checkPsR(stock):
                    if checkBroker(stock, Borker()):
                        addInterestingStock(stock)
    except ValueError:
        print("error1")


def testfilterr(i,n):
    print("testing",i)
    stock = Stock(i,n)
    try:
        if checkIv(stock):
            if checkROE(stock):
                if checkPbR(stock) or checkPsR(stock):
                    addInterestingStock(stock)
                else:
                    print(i, "was filtered out")
            else:
                print(i, "was filtered out")
        else:
            print(i, "was filtered out")
    except ValueError:
        print("error3")

















