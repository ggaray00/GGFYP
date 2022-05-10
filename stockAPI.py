import math
from pandas.core.indexing import IndexingError
import warnings
from sklearn.metrics import mean_squared_error
from db import*
import pandas as pd
import fundamentalanalysis as fa
from ml import bestFitModel
from user import *
import requests

warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
api_key = api

def intN(x):
    try:
        return float(x)
    except TypeError:
        return x

def floatN(x):
    try:
        return float(x)
    except TypeError:
        return x

def formatt(x):
    df = copy(x)
    df.reset_index(inplace=True)
    df.rename(columns={"index": "year"}, inplace=True)
    for i in df.columns:
        try:
            df[i] = df[i].apply(intN)
        except ValueError:
            continue

    return df

def format2(x):
    df = copy(x)
    df.reset_index(inplace=True)
    df.rename(columns={"index": "year"}, inplace=True)
    df['year'] = df['year'].apply(intN)
    for i in df.columns:
        try:
            df[i] = df[i].apply(floatN)
        except ValueError:
            continue
    return df

def msq(df):
    return mean_squared_error(df.iloc[:, 0].values.reshape(-1,1), df.iloc[:, 1].values)

class Stock:

    def __init__(self, ticker, n):
        try:
            self.ticker = ticker
            self.incomeStatement = self.getIncomeStatemntA()[self.getIncomeStatemntA()['year'] <= n]
            self.balanceSheet = self.getBalanceSheetA()[self.getIncomeStatemntA()['year'] <= n]
            self.cashFlow = self.getCashFlowA()[self.getIncomeStatemntA()['year'] <= n]
            self.ratios = self.getRatiosA()[self.getIncomeStatemntA()['year'] <= n]
            self.key_metrics_annually = self.getKeyMetricsA()[self.getIncomeStatemntA()['year'] <= n]
            self.profile = fa.profile(ticker, api_key).transpose()
            self.entreprise_value = self.getEnterpriseValue()[self.getIncomeStatemntA()['year'] <= n]
            self.growth_annually = self.getGrowthAnnually()[self.getIncomeStatemntA()['year'] <= n]

            self.incomeStatement.reset_index(inplace=True)
            self.balanceSheet.reset_index(inplace=True)
            self.cashFlow.reset_index(inplace=True)
            self.ratios.reset_index(inplace=True)
            self.key_metrics_annually.reset_index(inplace=True)
            self.entreprise_value.reset_index(inplace=True)
            try:
                self.currency = self.profile["currency"][0]
                self.mktCap = self.entreprise_value["marketCapitalization"][0]
                self.exchange = self.profile["exchangeShortName"][0]
                self.age = self.dflength()
                self.year = int(self.balanceSheet["year"][0])
                self.sttiker = self.stTicker()
                self.price = self.profile["price"][0]
            except KeyError:
                print("KeyError")

        except IndexingError:
            print("error3")

    def getBalanceSheetA(self):
        return formatt(fa.balance_sheet_statement(self.ticker, api_key, period="annual").transpose())

    def getIncomeStatemntA(self):
        return formatt(fa.income_statement(self.ticker, api_key, period="annual").transpose())

    def getCashFlowA(self):
        return formatt(fa.cash_flow_statement(self.ticker, api_key, period="annual").transpose())

    def getRatiosA(self):
        return format2(fa.financial_ratios(self.ticker, api_key, period="annual").transpose())

    def getKeyMetricsA(self):
        return format2(fa.key_metrics(self.ticker, api_key, period="annual").transpose())

    def getGrowthAnnually(self):
        return format2(fa.financial_statement_growth(self.ticker, api_key, period="annual").transpose())

    def getEnterpriseValue(self):
        return copy(formatt(fa.enterprise(self.ticker, api_key).transpose()))

    def stTicker(self):
        s = self.ticker
        if "." in s:
            s = self.ticker.split(".")[0]
        return s

    def dflength(self):
        return min(len(self.incomeStatement), len(self.balanceSheet), len(self.cashFlow))

    def data(self, param):
        df = pd.DataFrame()
        df['year'] = self.incomeStatement['calendarYear']
        try:
            df[param] = self.incomeStatement[param]
        except KeyError:
            try:
                df[param] = self.balanceSheet[param]
            except KeyError:
                try:
                    df[param] = self.cashFlow[param]
                except KeyError:
                    try:
                        df[param] = self.ratios[param]
                    except KeyError:
                        df[param] = self.key_metrics_annually[param]

        df = df[:self.dflength()]

        model = bestFitModel(df, param)
        tt_pred = model[1][1]  # test model result
        f_pred = model[0][2][['year', 'y_pred']]  # predictions 4 years into the future
        data = model[0][0]  # df containing given data + regression model
        tt_data = model[1][0]
        return tt_pred, f_pred, data, tt_data

    def roe(self):
        df = pd.DataFrame()
        df['year'] = self.incomeStatement['calendarYear']
        df['roe'] = self.incomeStatement['netIncome'] / self.balanceSheet['totalStockholdersEquity']
        df = df[:self.dflength()]

        model = bestFitModel(df, 'roe')
        test_pred = model[1][1]  # test model result
        complex_pred = self.data('netIncome')[0]
        complex_pred.rename(columns={'y_pred': 'income'}, inplace=True)
        complex_pred['shEquity'] = self.data('totalStockholdersEquity')[0]['y_pred']
        complex_pred['revenue'] = self.data('revenue')[0]['y_pred']
        complex_pred['assets'] = self.data('totalAssets')[0]['y_pred']
        complex_pred['roe_pred'] = (complex_pred['income'] / complex_pred['revenue']) * (
                    complex_pred['revenue'] / complex_pred['assets']) * (
                                           complex_pred['assets'] / complex_pred['shEquity'])

        # plt.scatter(test_pred['year'], test_pred['y_pred'], color='green')
        # plt.scatter(test_pred['year'], test_pred['y_test'], color='purple')
        # plt.plot(train['year'], train['y_pred'], color='blue')
        # plt.scatter(train['year'], train['y_train'], color='orange')
        # plt.scatter(complex_pred['year'], complex_pred['roe_pred'], color='blue')
        #
        # plt.title("roe" + self.ticker)
        # plt.suptitle(self.year)
        # plt.show()

        st_pred = test_pred[['y_pred', 'y_test']]
        complex_pred['y_test'] = test_pred['y_test']
        return st_pred, complex_pred[['roe_pred', 'y_test']], model[0][0]

    def roePred(self):
        f_pred = 0
        roe = self.roe()
        if msq(roe[0]) <= msq(roe[1]):
            f_pred = self.data('netIncome')[1]
            f_pred['y_pred2'] = self.data('totalStockholdersEquity')[1]['y_pred']
            f_pred['roe_pred'] = f_pred['y_pred']/f_pred['y_pred2']
        if msq(roe[0]) > msq(roe[1]):
            f_pred = self.data('netIncome')[1]
            f_pred.rename(columns={'y_pred': 'income'}, inplace=True)
            f_pred['shEquity'] = self.data('totalStockholdersEquity')[1]['y_pred']
            f_pred['revenue'] = self.data('revenue')[1]['y_pred']
            f_pred['assets'] = self.data('totalAssets')[1]['y_pred']
            f_pred['roe_pred'] = (f_pred['income'] / f_pred['revenue']) * (f_pred['revenue'] / f_pred['assets']) * (
                                           f_pred['assets'] / f_pred['shEquity'])

        return f_pred[['year','roe_pred']]

    def roeProfile(self):
        data = self.roe()[2]
        f_data = self.roePred()
        if f_data['roe_pred'].values[-1] > data['y_train'].values[-3] > data['y_train'].values[-6]:
            return True
        else:
            return False

    def dCF(self):
        df = pd.DataFrame()
        df["year"] = self.balanceSheet["year"]
        if "marketCap" in self.cashFlow:
            df["marketCap"] = self.ratios["marketCap"]
        else:
            df["marketCap"] = self.entreprise_value["marketCapitalization"]
        df["freeCashFlow"] = self.cashFlow["freeCashFlow"]
        df["MCap/FCash"] = df["marketCap"] / df["freeCashFlow"]
        fcfGrowth = []
        avMcapToFcf = []
        for i, v in enumerate(df["freeCashFlow"].dropna()):
            count1 = 0
            count2 = 0
            if len(df["freeCashFlow"].dropna()) > i + 10:
                for j in range(10):
                    count1 = count1 + (df["freeCashFlow"][i + j] / df["freeCashFlow"][i + j + 1])
                    count2 = count2 + (df["marketCap"][i + j] / df["freeCashFlow"][i + j])
            fcfGrowth.append(count1 / 10)
            avMcapToFcf.append(count2 / 10)
        df["av10FcfGrowth"] = pd.Series(fcfGrowth)
        df["McFcfav10"] = pd.Series(avMcapToFcf)
        dfcf = []
        for i, v in enumerate(df["freeCashFlow"].dropna()):
            free_CF = []
            discounted_free_CF = []
            for j in range(1, 11):
                free_CF.append(df["freeCashFlow"][i] * df["av10FcfGrowth"][i] ** j)
            for j, v in enumerate(free_CF):
                discounted_free_CF.append(v / (1.15 ** (j + 1)))
            dfcf.append((free_CF[-1] * df["McFcfav10"][i]) / df["av10FcfGrowth"][i] ** 10 + sum(discounted_free_CF))
        df["discountedFreeCf"] = pd.Series(dfcf)

        dcf = df["discountedFreeCf"][0]
        if math.isnan(dcf):
            dcf = 0

        return dcf

    def iv(self):
        g1 = 0.1
        g2 = 0.05
        d1 = 0.12
        d2 = 0.15
        be = self.incomeStatement["epsdiluted"][0] * self.entreprise_value["numberOfShares"][0]
        n = 10
        discounted_10y_earnings = []
        for i in range(1, 11):
            discounted_10y_earnings.append((be * ((1 + g1) ** i)) / ((1 + d1) ** i))
        a = sum(discounted_10y_earnings)
        continueV10years = ((((be * (1 + g1) ** (n + 1))) / (d2 - g2)) / (1 + d1) ** n)
        future = (a + continueV10years) - (self.balanceSheet["longTermDebt"][0])
        return future

    def intrinsic(self):
        return self.mktCap/max(self.dCF(),self.iv())







