from copy import copy
import requests
import pandas as pd
from sqlalchemy import create_engine
import pymysql
from user import *
import fundamentalanalysis as fa




database = "demo"
api_key = api
def floatN(x):
    try:
        return float(x)
    except TypeError:
        return x
def format3(x):
    df = copy(x)
    #df.drop(["reportedCurrency"], axis=1, inplace=True)
    df.reset_index(inplace=True,drop= True)
    for i in df.columns:
        try:
            df[i] = df[i].apply(floatN)
        except ValueError:
            continue
    return df

conn = pymysql.connect(host="localhost", user="root", passwd=password)
conn.cursor().execute("CREATE DATABASE IF NOT EXISTS {0} ".format(database))
engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}".format(user="root", pw=password, db=database))

def addFStatementsA(s):
    addtodb(s.incomeStatement,'{i}_incomeStatment'.format(i=s.sttiker))
    addtodb(s.balanceSheet, '{i}_balanceSheet'.format(i=s.sttiker))
    addtodb(s.cashFlow, '{i}_cashFlow'.format(i=s.sttiker))
    addtodb(s.entreprise_value, '{i}_entreprise_value'.format(i=s.sttiker))
    addtodb(s.ratios, '{i}_ratios'.format(i=s.sttiker))
    addtodb(s.growth_annually, '{i}_growth_annually'.format(i=s.sttiker))
    addtodb(s.key_metrics_annually, '{i}_key_metrics_annually'.format(i=s.sttiker))
    addtodb(s.profile, '{i}_profile'.format(i=s.sttiker))



def addFStatementsQ(s):
    addtodb(format3(fa.balance_sheet_statement(s.ticker, api_key, period="quarter").transpose()),
                '{i}_balanceSheetQ'.format(i=s.sttiker))
    addtodb(format3(fa.income_statement(s.ticker, api_key, period="quarter").transpose()),
                '{i}_incomeStatmentQ'.format(i=s.sttiker))
    addtodb(format3(fa.cash_flow_statement(s.ticker, api_key, period="quarter").transpose()),
                '{i}_cashFlowQ'.format(i=s.sttiker))
    addtodb(format3(fa.financial_ratios(s.ticker, api_key, period="quarter").transpose()),
                '{i}_ratiosQ'.format(i=s.sttiker))
    addtodb(format3(fa.key_metrics(s.ticker, api_key, period="quarter").transpose()),
                '{i}_key_metricsQ'.format(i=s.sttiker))

def addtodb(df,name):
    df.to_sql(con=engine, name=name,if_exists='replace',index=False)

def allStockstoDB(df):
    df = df.copy()
    df = df[df["type"] == "stock"]
    df["ticker"] = df.index
    pdtodb(df,"allStokss")

def createTable(x):
    conn = pymysql.connect(host="localhost", user="root", passwd=password,database=database)
    conn.cursor().execute(('''CREATE TABLE IF NOT EXISTS {TABLE_NAME}(ticker CHAR(20) PRIMARY KEY,currency CHAR(20),exchange CHAR(20))''').
                          format(TABLE_NAME =x))
    conn.commit()
    conn.close()

def addViwedStock(s,year):
    conn = pymysql.connect(host="localhost", user="root", passwd=password, database=database)
    conn.cursor().execute(('''CREATE TABLE IF NOT EXISTS {TABLE_NAME}(ticker CHAR(20) PRIMARY KEY)''').
                          format(TABLE_NAME='viewed_stocks'.format(year=year)))
    conn.cursor().execute("INSERT INTO  viewed_stocks (`ticker`) VALUES (%s)",
                          (s.ticker))
    conn.commit()
    conn.close()

def addViwedStock2(s):
    conn = pymysql.connect(host="localhost", user="root", passwd=password, database=database)
    conn.cursor().execute(('''CREATE TABLE IF NOT EXISTS {TABLE_NAME}(ticker CHAR(20) PRIMARY KEY)''').
                          format(TABLE_NAME='viewed_stocks2'))
    conn.cursor().execute("INSERT INTO  viewed_stocks2 (`ticker`) VALUES (%s)",
                          (s.ticker))
    conn.commit()
    conn.close()

def addInterestingStock(s):
    conn = pymysql.connect(host="localhost", user="root", passwd=password, database=database)
    conn.cursor().execute(('''CREATE TABLE IF NOT EXISTS {TABLE_NAME}(ticker CHAR(20) PRIMARY KEY,currency CHAR(20),exchange CHAR(20), iv CHAR(30))''').
                          format(TABLE_NAME='interesting_stocks'))
    conn.cursor().execute("INSERT INTO interesting_stocks (`ticker`, `currency`, `exchange`, `iv`) VALUES (%s, %s, %s, %s)",
                          (s.ticker, s.currency, s.exchange, s.intrinsic()))
    conn.commit()
    conn.close()

def addMkt(s):
    conn = pymysql.connect(host="localhost", user="root", passwd=password, database=database)
    conn.cursor().execute(('''CREATE TABLE IF NOT EXISTS {TABLE_NAME}(ticker CHAR(20) PRIMARY KEY,currency CHAR(20),exchange CHAR(20), iv CHAR(30))''').
                          format(TABLE_NAME='mkt2'))
    conn.cursor().execute("INSERT INTO mkt2 (`ticker`) VALUES (%s)",(s))
    conn.commit()
    conn.close()

def addInterestingStock2(s):
    conn = pymysql.connect(host="localhost", user="root", passwd=password, database=database)
    conn.cursor().execute(('''CREATE TABLE IF NOT EXISTS {TABLE_NAME}(ticker CHAR(20) PRIMARY KEY)''').
                          format(TABLE_NAME='interesting_stocks2'))
    conn.cursor().execute("INSERT INTO interesting_stocks2 (`ticker`, `currency`, `exchange`, `iv`) VALUES (%s, %s, %s, %s)",
                          (s.ticker, s.currency, s.exchange, s.intrinsic()))
    conn.commit()
    conn.close()

def addPotentialBuy(s,year):
    conn = pymysql.connect(host="localhost", user="root", passwd=password, database=database)
    conn.cursor().execute(('''CREATE TABLE IF NOT EXISTS {TABLE_NAME}(ticker CHAR(20) PRIMARY KEY, roe CHAR(20),roeSTD CHAR(30), opm CHAR(20),opmSTD CHAR(30), gpm CHAR(20),gpmSTD CHAR(30), gm CHAR(20),gmSTD CHAR(30),fcf CHAR(20),fcfSTD CHAR(30))''').
                          format(TABLE_NAME='potential_buy_{year}'.format(year=year)))
    conn.cursor().execute("INSERT INTO potential_buy (`ticker`, `roe`, `roeSTD`, `opm`,`opmSTD`,`gpm`,`gpmSTD`,`gm`,`gmSTD`,`fcf`,`fcfSTD`) VALUES (%s, %s, %s, %s,%s, %s, %s, %s,%s, %s, %s)",
                          (s.ticker, s.roeProfile()['increasing'], s.roeProfile()['std'], s.opMarginProfile()['increasing'],s.opMarginProfile()['std'],s.gProfitMarginProfile()['increasing'],s.gProfitMarginProfile()['std'],s.gMarginProfile()['increasing'],s.gMarginProfile()['std'],s.fcfProfile()[0]['increasing'],s.fcfProfile()[0]['std']))
    conn.commit()
    conn.close()

def sotckProtfolioProfile(s,year):
    conn = pymysql.connect(host="localhost", user="root", passwd=password, database=database)
    conn.cursor().execute(('''CREATE TABLE IF NOT EXISTS {TABLE_NAME}(ticker CHAR(20) PRIMARY KEY, roe CHAR(20),roeSTD CHAR(30), opm CHAR(20),opmSTD CHAR(30), gpm CHAR(20),gpmSTD CHAR(30), gm CHAR(20),gmSTD CHAR(30),fcf CHAR(20),fcfSTD CHAR(30))''').
                          format(TABLE_NAME='stock_portfolio_profile_{year}'.format(year=year)))
    conn.cursor().execute("INSERT INTO 'stock_portfolio_profile_{year}'.format(year=year)) (`ticker`, `roe`, `roeSTD`, `opm`,`opmSTD`,`gpm`,`gpmSTD`,`gm`,`gmSTD`,`fcf`,`fcfSTD`) VALUES (%s, %s, %s, %s,%s, %s, %s, %s,%s, %s, %s)",
                          (s.ticker, s.roeProfile()[0]['increasing'], s.roeProfile()[0]['std'], s.opMarginProfile()['increasing'],s.opMarginProfile()['std'],s.gProfitMarginProfile()['increasing'],s.gProfitMarginProfile()['std'],s.gMarginProfile()['increasing'],s.gMarginProfile()['std'],s.fcfProfile()['increasing'],s.fcfProfile()['std']))
    conn.commit()
    conn.close()

def addNotViwedStock(s):
    conn = pymysql.connect(host="localhost", user="root", passwd=password, database=database)
    conn.cursor().execute("INSERT INTO stocks_not_viewed (`ticker`, `currency`, `exchange`) VALUES (%s, %s, %s)",
                          (s.ticker, s.currency, s.exchange))
    conn.commit()
    conn.close()

def addTestStock(s):
    conn = pymysql.connect(host="localhost", user="root", passwd=password, database=database)
    conn.cursor().execute("INSERT INTO test_stock (`ticker`) VALUES (%s)",
                          (s.ticker))
    conn.commit()
    conn.close()

def delteRow(s):
    conn = pymysql.connect(host="localhost", user="root", passwd=password, database=database)
    conn.cursor().execute(
        """DELETE FROM stocks_not_viewed WHERE ticker = %s""",s.ticker)
    conn.commit()
    conn.close()

def delteRow(s,name):
    conn = pymysql.connect(host="localhost", user="root", passwd=password, database=database)
    conn.cursor().execute("""DELETE FROM n100_stocks WHERE ticker = %s""",s.ticker)
    conn.commit()
    conn.close()

def delteInterestingStock(s):
    conn = pymysql.connect(host="localhost", user="root", passwd=password, database=database)
    conn.cursor().execute(
        """DELETE FROM interesting_stocks WHERE ticker = %s""",s.ticker)
    conn.commit()
    conn.close()

def dbtopd(table):
    return pd.read_sql_table("{table}".format(table=table), engine)

def pdtodb(df,name):
    df.to_sql(con=engine, name='{i}'.format(i=name),if_exists='replace', index=False)

def dropFinfo(s):
    conn = pymysql.connect(host="localhost", user="root", passwd=password, database=database)
    conn.cursor().execute("DROP TABLE IF EXISTS {ticker}_incomeStatment".format(ticker=s.stTicker()))
    conn.cursor().execute("DROP TABLE IF EXISTS {ticker}_balanceSheet".format(ticker=s.stTicker()))
    conn.cursor().execute("DROP TABLE IF EXISTS {ticker}_cashFlow".format(ticker=s.stTicker()))
    conn.cursor().execute("DROP TABLE IF EXISTS {ticker}_ratios".format(ticker=s.stTicker()))
    conn.cursor().execute("DROP TABLE IF EXISTS {ticker}_key_metrics_annually".format(ticker=s.stTicker()))
    conn.cursor().execute("DROP TABLE IF EXISTS {ticker}_entreprise_value".format(ticker=s.stTicker()))
    conn.cursor().execute("DROP TABLE IF EXISTS {ticker}_growth_annually".format(ticker=s.stTicker()))
    conn.cursor().execute("DROP TABLE IF EXISTS {ticker}_profile".format(ticker=s.stTicker()))
    conn.commit()
    conn.close()

def dropTable(name):
    conn = pymysql.connect(host="localhost", user="root", passwd=password, database=database)
    conn.cursor().execute("DROP TABLE IF EXISTS {ticker}".format(ticker=name))


def createTable2(x):
    conn = pymysql.connect(host="localhost", user="root", passwd=password,database=database)
    conn.cursor().execute(('''CREATE TABLE IF NOT EXISTS {TABLE_NAME}(ticker CHAR(20) PRIMARY KEY)''').
                          format(TABLE_NAME =x))
    conn.commit()
    conn.close()

def addPortfolio():
    conn = pymysql.connect(host="localhost", user="root", passwd=password, database=database)
    conn.cursor().execute(('''CREATE TABLE IF NOT EXISTS {TABLE_NAME}(ticker CHAR(20) PRIMARY KEY, shares CHAR(20),mktPrice CHAR(30), mktValue CHAR(20),avCost CHAR(30))''').
                          format(TABLE_NAME='portfolio'))
    conn.commit()
    conn.close()

def seriesToDb(series,name):
    series.to_sql(con=engine, name='{i}'.format(i=name), if_exists='replace', index=False)
