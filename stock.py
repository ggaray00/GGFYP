import warnings

import matplotlib.pyplot as plt

from ml import *
from db import *
import pandas as pd
import fundamentalanalysis as fa
from user import *
import math


warnings.simplefilter(action='ignore', category=RuntimeWarning)

# database = "results"
# conn = pymysql.connect(host="localhost", user="root", passwd=password)
# conn.cursor().execute("CREATE DATABASE IF NOT EXISTS {0} ".format(database))
# engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}".format(user="root", pw=password, db=database))


def intN(x):
    try:
        return float(x)
    except TypeError:
        return x

def msq(df):
    return mean_squared_error(df.iloc[:, 0].values.reshape(-1,1), df.iloc[:, 1].values)

def std(df):
    X = df.iloc[:, 1:].values
    X = StandardScaler().fit_transform(X)
    c = 0
    for i in X:
        c = c + (i[1] - i[0]) ** 2
    std = math.sqrt(c / len(X))
    return std

def floatN(x):
    try:
        return float(x)
    except TypeError:
        return x

def format2(x):
    df = copy(x)
    # df.drop(["period"], axis=1, inplace=True)
    df.reset_index(inplace=True)
    df.rename(columns={"index": "year"}, inplace=True)
    for i in df.columns:
        try:
            df[i] = df[i].apply(floatN)
        except ValueError:
            continue
    return df

def format3(x):
    df = copy(x)
    df.reset_index(inplace=True, drop=True)
    for i in df.columns:
        try:
            df[i] = df[i].apply(floatN)
        except ValueError:
            continue
    return df

class Stock2:

    def __init__(self, ticker,n):
        self.ticker = ticker
        self.sttiker = self.stTicker()
        self.incomeStatement = dbtopd("{ticker}_incomeStatment".format(ticker=self.stTicker()))
        self.balanceSheet = dbtopd("{ticker}_balanceSheet".format(ticker=self.stTicker()))
        self.cashFlow = dbtopd("{ticker}_cashFlow".format(ticker=self.stTicker()))
        self.ratios = dbtopd("{ticker}_ratios".format(ticker=self.stTicker()))
        self.key_metrics_annually = dbtopd("{ticker}_key_metrics_annually".format(ticker=self.stTicker()))
        self.profile = dbtopd("{ticker}_profile".format(ticker=self.stTicker()))
        self.entreprise_value = dbtopd("{ticker}_entreprise_value".format(ticker=self.stTicker()))
        self.growth_annually = dbtopd("{ticker}_growth_annually".format(ticker=self.stTicker()))


        self.incomeStatement = self.incomeStatement[self.incomeStatement['year'] <= n]
        self.balanceSheet = self.balanceSheet[self.balanceSheet['year'] <= n]
        self.cashFlow = self.cashFlow[self.cashFlow['year'] <= n]
        self.ratios = self.ratios[self.ratios['year'] <= n]
        self.entreprise_value = self.entreprise_value[self.entreprise_value['year'] <= n]
        self.growth_annually = self.growth_annually[self.growth_annually['year'] <= n]
        self.key_metrics_annually = self.key_metrics_annually[self.key_metrics_annually['year'] <= n]

        self.incomeStatement.reset_index(inplace=True,drop=True)
        self.balanceSheet.reset_index(inplace=True,drop=True)
        self.cashFlow.reset_index(inplace=True,drop=True)
        self.ratios.reset_index(inplace=True,drop=True)
        self.entreprise_value.reset_index(inplace=True,drop=True)
        self.growth_annually.reset_index(inplace=True,drop=True)
        self.key_metrics_annually.reset_index(inplace=True, drop=True)

        self.currency = self.profile["currency"][0]
        # self.incomeStatement.reset_index(inplace=True)
        # self.balanceSheet.reset_index(inplace=True)
        # self.cashFlow.reset_index(inplace=True)
        # self.ratios.reset_index(inplace=True)
        # self.key_metrics_annually.reset_index(inplace=True)
        # self.entreprise_value.reset_index(inplace=True)
        self.year = self.balanceSheet["year"][0]
        self.age = self.ageOfStock()
        self.price = self.entreprise_value['stockPrice'][0]
        self.mktCap = self.entreprise_value['marketCapitalization'][0]

    def dflength(self):
        return min(len(self.incomeStatement), len(self.balanceSheet), len(self.cashFlow))

    def getGrowthAnnually(self):
        return format2(pd.read_sql_table("{ticker}_growth_annually".format(ticker=self.stTicker()), engine))

    def ageOfStock(self):
        return len(self.balanceSheet["year"])

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
        return df["discountedFreeCf"][0]

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

    def stTicker(self):
        s = self.ticker
        if "." in s:
            s = self.ticker.split(".")[0]
        return s

    def qDataToDb(self):
        addFStatementsQ(self)

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
        train = model[1][0]
        train = train[:self.dflength()]

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

        # f_pred = model[0][2][['year', 'y_pred']]  # predictions 4 years into the future
        # data = model[0][0]  # df containing given data + regression model
        return st_pred, complex_pred[['roe_pred', 'y_test']], model[0][0]

    def gorssProfitMargin(self):
        df = pd.DataFrame()
        df['year'] = self.incomeStatement['calendarYear']
        df['cogs'] = self.incomeStatement['revenue'] - self.incomeStatement['grossProfit']
        df = df[:self.dflength()]
        cogs_model = bestFitModel(df, 'cogs')
        # cogs_test_pred = cogs_model[1][1]['y_pred']  # test model result
        futrure_cogs = cogs_model[0][2][['year', 'y_pred']]
        # train = cogs_model[1][1][:self.dflength()]

        complex_test_pred_df = self.data('revenue')[0].rename(columns={'y_pred':'revenue'})
        complex_test_pred_df['cogs'] = cogs_model[1][1]['y_pred']
        complex_test_pred_df['grossProfit'] = complex_test_pred_df['revenue'] - complex_test_pred_df['cogs']


        grossProfit = self.data('grossProfit')

        gpData_tt = grossProfit[3]
        gp_simple_tt = grossProfit[0]
        gp_complex_tt = pd.DataFrame()

        gp_complex_tt['year'] = gp_simple_tt['year']
        gp_complex_tt['y_pred'] = complex_test_pred_df['grossProfit']
        gp_complex_tt['y_test'] = gp_simple_tt['y_test']

        # plt.plot(gpData_tt['year'], gpData_tt['y_pred'], color='blue')
        # plt.scatter(gpData_tt['year'], gpData_tt['y_train'], color='green')
        # plt.scatter(gp_simple_tt['year'], gp_simple_tt['y_pred'], color='red')
        # plt.scatter(gp_simple_tt['year'], gp_simple_tt['y_test'], color='purple')
        # plt.scatter(complex_test_pred_df['year'], complex_test_pred_df['grossProfit'], color='orange')
        # plt.title("gorssProfitMargin" + self.ticker)
        # plt.suptitle(self.year)
        # plt.show()

        return gp_simple_tt, gp_complex_tt, futrure_cogs, grossProfit[2], cogs_model, df

    def grossMargin(self):
        df = pd.DataFrame()
        df['year'] = self.incomeStatement['calendarYear']

        df['grossMargin'] = (self.incomeStatement['revenue'] - self.gorssProfitMargin()[5]['cogs'])\
                            /self.incomeStatement['revenue']

        df = df[:self.dflength()]
        grossMargin_model = bestFitModel(df, 'grossMargin')

        complex_test_pred_df = self.data('revenue')[0].rename(columns={'y_pred': 'revenue'})
        complex_test_pred_df['cogs'] = self.gorssProfitMargin()[4][1][1]['y_pred']
        complex_test_pred_df['grossMargin'] = (complex_test_pred_df['revenue'] - complex_test_pred_df['cogs'])\
                                              /complex_test_pred_df['revenue']

        gMData_tt = grossMargin_model[0][0]
        gM_simple_tt = grossMargin_model[1][1]
        gM_complex_tt = pd.DataFrame()

        gM_complex_tt['year'] = gM_simple_tt['year']
        gM_complex_tt['y_pred'] = complex_test_pred_df['grossMargin']
        gM_complex_tt['y_test'] = gM_simple_tt['y_test']

        # plt.plot(gMData_tt['year'], gMData_tt['y_pred'], color='blue')
        # plt.scatter(gMData_tt['year'], gMData_tt['y_train'], color='green')
        # plt.scatter(gM_simple_tt['year'], gM_simple_tt['y_pred'], color='red')
        # plt.scatter(gM_simple_tt['year'], gM_simple_tt['y_test'], color='purple')
        # plt.scatter(complex_test_pred_df['year'], complex_test_pred_df['grossMargin'], color='orange')
        # plt.title("gorssMargin" + self.ticker)
        # plt.suptitle(self.year)
        # plt.show()

        return gM_simple_tt, gM_complex_tt, grossMargin_model[0][2][['year', 'y_pred']],grossMargin_model[0][0]

    def opMargin(self):
        df = pd.DataFrame()
        df['year'] = self.incomeStatement['calendarYear']
        df['opMargin'] = self.incomeStatement['ebitda']/self.incomeStatement['revenue']
        df = df[:self.dflength()]
        opMargin_model = bestFitModel(df, 'opMargin')
        simple_future_pred = opMargin_model[0][2][['year', 'y_pred']]
        opMargin_simple_test = opMargin_model[1][1][:self.dflength()]
        complex_test_pred_df = self.data('revenue')[0].rename(columns={'y_pred': 'revenue'})
        complex_test_pred_df['ebitda'] = self.data('ebitda')[0]['y_pred']
        complex_test_pred_df['opMargin'] = complex_test_pred_df['ebitda'] / complex_test_pred_df['revenue']

        # plt.scatter(complex_test_pred_df['year'], complex_test_pred_df['opMargin'], color='black')
        # plt.plot(opMargin_model_train['year'], opMargin_model_train['y_pred'], color='blue')
        # plt.scatter(opMargin_model_train['year'], opMargin_model_train['y_train'], color='green')
        # plt.scatter(opMargin_simple_test['year'], opMargin_simple_test['y_pred'], color='orange')
        # plt.scatter(opMargin_simple_test['year'], opMargin_simple_test['y_test'], color='purple')
        # plt.title("opMargin")
        # plt.show()

        opMargin_complex_test = pd.DataFrame()
        opMargin_complex_test['year'] = opMargin_simple_test['year']
        opMargin_complex_test['y_pred'] = complex_test_pred_df['opMargin']
        opMargin_complex_test['y_test'] = opMargin_simple_test['y_test']

        # plt.plot(opMargin_model[1][0]['year'], opMargin_model[1][0]['y_pred'], color='blue')
        # plt.scatter(opMargin_model[1][0]['year'], opMargin_model[1][0]['y_train'], color='green')
        # plt.scatter(opMargin_simple_test['year'], opMargin_simple_test['y_pred'], color='red')
        # plt.scatter(opMargin_simple_test['year'], opMargin_simple_test['y_test'], color='purple')
        # plt.scatter(complex_test_pred_df['year'], complex_test_pred_df['opMargin'], color='orange')
        # plt.title("opMargin " + self.ticker)
        # plt.suptitle(self.year)
        # plt.show()

        return opMargin_simple_test, opMargin_complex_test,simple_future_pred, opMargin_model[0][0]

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

    def gMarginPred(self):
        f_pred = 0
        gm = self.grossMargin()
        if msq(gm[0]) <= msq(gm[1]):
            f_pred = gm[2]
        if msq(gm[0]) > msq(gm[1]):
            f_pred = self.data('revenue')[1]
            f_pred.rename(columns={'y_pred': 'revenue'}, inplace=True)
            f_pred['cogs'] = self.gorssProfitMargin()[2]['y_pred']
            f_pred['y_pred'] = (f_pred['revenue'] - f_pred['cogs'])/ f_pred['revenue']

        return f_pred.iloc[:, [0, -1]]

    def opMarginPred(self):
        f_pred = 0
        opMargin = self.opMargin()
        # for i in opMargin:
        #     print(i)
        if msq(opMargin[0]) > msq(opMargin[1]):
            f_pred = self.data('ebitda')[1]
            f_pred.rename(columns={'y_pred': 'opMargin'}, inplace=True)
            f_pred['y_pred2'] = self.data('revenue')[1]['y_pred']
            f_pred['y_pred'] = f_pred['opMargin']/f_pred['y_pred2']
        if msq(opMargin[0]) <= msq(opMargin[1]):
            f_pred = opMargin[2]
        return f_pred.iloc[:,[0,-1]]

    def grossProfitMarginPred(self):
        f_pred = 0
        gorssMargin = self.gorssProfitMargin()
        if msq(gorssMargin[0]) > msq(gorssMargin[1]):
            f_pred = gorssMargin[2]
            f_pred['y_pred2'] = self.data('revenue')[1]['y_pred']
            f_pred['grossMarginPred'] = f_pred['y_pred2'] - f_pred['y_pred']
        if msq(gorssMargin[0]) <= msq(gorssMargin[1]):
            f_pred = self.data('grossProfit')[1]
            f_pred.rename(columns={'y_pred': 'grossMarginPred'}, inplace=True)


        # plt.scatter(hola['year'], hola['y_train'], color= 'red')
        # plt.plot(hola['year'], hola['y_pred'], color='blue')
        # print(f_pred)
        # plt.scatter(f_pred['year'], f_pred['grossMarginPred'], color='blue')
        # plt.scatter(f_pred['year'], f_pred['gorossMarginPred'], color='purple')
        # plt.show()

        return f_pred.iloc[:, [0, -1]]

    def roeProfile(self):
        data = self.roe()[2]
        f_data = self.roePred()
        roe = {}
        if f_data['roe_pred'].values[-1] > \
                f_data['roe_pred'].values[0] > data['y_train'].values[-3] > data['y_train'].values[-9]:
            roe['increasing'] = True
        else:
            roe['increasing'] = False
        roe['std'] = std(data)
        return roe

    def opMarginProfile(self):
        data = self.opMargin()[3]
        f_data = self.opMarginPred()
        opMargin = {}
        if f_data['y_pred'].values[-1] > \
                f_data['y_pred'].values[0] > data['y_train'].values[-3] > data['y_train'].values[-9]:
            opMargin['increasing'] = True
        else:
            opMargin['increasing'] = False
        opMargin['std'] = std(data)
        return opMargin

    def gMarginProfile(self):
        data = self.grossMargin()[3]
        f_data = self.gMarginPred()
        gMargin = {}
        if f_data['y_pred'].values[-1] > data['y_train'].values[-5]:
                # f_data['y_pred'].values[0] > data['y_train'].values[-3]: > data['y_train'].values[-9]:
            gMargin['increasing'] = True
        else:
            gMargin['increasing'] = False
        gMargin['std'] = std(data)
        return gMargin

    def gProfitMarginProfile(self):
        data = self.gorssProfitMargin()[3]
        f_data = self.grossProfitMarginPred()
        gMargin = {}
        if f_data['grossMarginPred'].values[-1] > \
                f_data['grossMarginPred'].values[0] > data['y_train'].values[-3] > data['y_train'].values[-9]:
            gMargin['increasing'] = True
        else:
            gMargin['increasing'] = False
        gMargin['std'] = std(data)
        return gMargin

    def revenueProfile(self):
        data = self.data("revenue")
        rev = {}
        if data[1]['y_pred'].values[-1] > \
                data[1]['y_pred'].values[0] > data[2]['y_train'].values[-3] > data[2]['y_train'].values[-9]:
            rev['increasing'] = True
        else:
            rev['increasing'] = False
        rev['std'] = std(data[2])
        return rev, (data[1]['y_pred'].values[-1])*2 < data[2]['y_train'].values[-3]

    def retTangibeAssetsProfile(self):
        model = self.data('returnOnTangibleAssets')
        f_data = model[1]
        data = model[2]
        rta = {}
        if f_data['y_pred'].values[-1] > \
                f_data['y_pred'].values[0] > data['y_train'].values[-3] > data['y_train'].values[-9]:
            rta['increasing'] = True
        else:
            rta['increasing'] = False
        rta['std'] = std(data)
        return rta

    def fcfProfile(self):
        model = self.data('freeCashFlow')
        f_data = model[1]
        data = model[2]
        fcf = {}
        # plt.scatter(data["year"],data['y_train'],color="blue")
        # plt.scatter(f_data["year"], f_data['y_pred'], color="green")
        # plt.show()
        if f_data['y_pred'].values[-1] > data['y_pred'].values[-1] > data['y_pred'].values[-9]:
            fcf['increasing'] = True
        else:
            fcf['increasing'] = False
        fcf['std'] = std(data)
        return fcf,f_data['y_pred'].values[-1]

    def debtProfile(self):
        deR = self.ratios['debtEquityRatio'].values
        if min(deR) >= 0 and max(deR) <= 2:
            return True
        else:
            return False

    def dividend(self):
        if self.ratios['dividendYield'][0] is None:
            return False
        else:
            return True

    def dividendPerShare(self):
        if self.dividend():
            return self.ratios['dividendYield'][0] * self.price
        else:
            return 0

    def profiles(self):
        print(self.ticker, "  ", self.year)
        print("price: ", self.price)
        print("intrinsic:            ",self.intrinsic())
        print("return on equity:     ",self.roeProfile())
        print("operating margins:    ",self.opMarginProfile())
        print("revenue:              ",self.revenueProfile()[0])
        try:
            print("gross profit margins: ",self.gProfitMarginProfile())
        except ValueError:
            print("error")
        try:
            print("gross margins:        ",self.gMarginProfile())
        except ValueError:
            print("error")
        print("free cash flow:       ",self.fcfProfile()[0])

    def intrinsic(self):
        #a = float(self.iv(0.1, 0.05,0.12, 0.15))
        a = float(self.iv())
        b = float(self.dCF())
        if a < 0 or math.isnan(a):
            a = 1
        if b <= 0 or math.isnan(b):
            b = 1
        return {"intrinsic value": self.mktCap/max(a,b)}



class Stock3:
    def __init__(self, ticker, cost, year,n):
        self.ticker = ticker
        self.cost = cost
        self.year = year
        self.shares = cost/Stock2(ticker,year).entreprise_value["stockPrice"].values[0]
        self.mktPrice = Stock2(ticker,n).price
        self.mktValue = self.shares * self.mktPrice

    def portfolioDict(self):
        return {"ticker":self.ticker,"cost": self.cost, "shares": self.shares,
                "mktPrice":self.mktPrice, "mktValue": self.mktValue,"year": self.year}





