
from filter import *
from portfolio import *

def reviewTest(n):
    df = pd.DataFrame(
        columns=['ticker', 'roe', 'roeSTD', 'opm', 'opmSTD', 'gpm', 'gpmSTD', 'gm', 'gmSTD', 'fcf', 'fcfSTD'])
    if dbtopd("interesting_stocks")['ticker'].empty:
        pass
    else:
        for i in dbtopd("interesting_stocks")['ticker'].values:
            try:
                s = Stock2(i, n)
                print(s.profiles())
                if s.roeProfile()['increasing'] and s.opMarginProfile()['increasing'] \
                        and s.gProfitMarginProfile()['increasing'] and s.gMarginProfile()['increasing'] and \
                        s.fcfProfile()[0]['increasing']:
                    df = df.append({'ticker': i, 'roe': s.roeProfile()['increasing'], 'roeSTD': s.roeProfile()['std'],
                                    'opm': s.opMarginProfile()['increasing'], 'opmSTD': s.opMarginProfile()['std'],
                                    'gpm': s.gProfitMarginProfile()['increasing'],
                                    'gpmSTD': s.gProfitMarginProfile()['std'],
                                    'gm': s.gMarginProfile()['increasing'], 'gmSTD': s.gMarginProfile()['std'],
                                    'fcf': s.fcfProfile()[0]['increasing'], 'fcfSTD': s.fcfProfile()[0]['std']},
                                   ignore_index=True)
                    pdtodb(df, "potential_buy_{n}".format(n=n))
            except ValueError:
                print("error1")

def potential_buyTest(n):
    fiveStocksDict = {}
    try:
        potential_buy = dbtopd("potential_buy_{n}".format(n=n))
        if potential_buy.empty:
            pass
        else:
            if len(potential_buy) <= 5:
                invest = TestPorfolio().cash * 0.1
                for i in potential_buy["ticker"]:
                    TestPorfolio().addStock(Stock3(i, invest, n,n))
            else:
                for i in potential_buy["tickers"]:
                    fiveStocksDict[i] = Stock2(i, n).intrinsic()
                    if Stock2(i, n).debtProfile() != True:
                        potential_buy = potential_buy[potential_buy['tickers'] != i].reset_index(drop=True)
                        if len(potential_buy) > 5:
                            if Stock2(i, n).retTangibeAssetsProfile()["increasing"] == 0:
                                potential_buy = potential_buy[potential_buy['tickers'] != i].reset_index(drop=True)

                if len(potential_buy) <= 5:
                    invest = TestPorfolio().cash * 0.1
                    for i in potential_buy["ticker"]:
                        TestPorfolio().addStock(Stock3(i, invest, n,n))
                else:
                    fiveStocks = pd.DataFrame(list(fiveStocksDict.items()))
                    fiveStocks = fiveStocks.nsmallest(5, 1)
                    for i in fiveStocks[0]:
                        invest = TestPorfolio().cash * 0.1
                        TestPorfolio().addStock(Stock3(i, invest, n,n))
    except ValueError:
        print("No undervalued stocks last year")

def porfolio_checkTest(n):
    TestPorfolio().portfolioStockProfile(n)
    pp = dbtopd("portfolio_stock_profile_{n}".format(n=n))
    portfolio = dbtopd("portfolio")
    if not portfolio.empty:
        for i in range(len(pp)):
            s = Stock2(pp.iloc[i]["ticker"], n)
            b = Stock3(portfolio.iloc[i]["ticker"], portfolio.iloc[i]["cost"],portfolio.iloc[i]["year"], n)
            if pp.iloc[i]["roe"] == 0 and pp.iloc[i]["opm"] == 0 \
                    and pp.iloc[i]["gpm"] == 0 and pp.iloc[i]["gm"] == 0 and pp.iloc[i]["fcf"] == 0 and \
                   s.revenueProfile()[0]["increasing"] == False:
                TestPorfolio().deleteStock(b,n)
                print("sold",portfolio.iloc[i]["ticker"], "p/l:", (b.mktValue/b.cost)*100 - 100,"%")
                continue
            if pp.iloc[i]["opm"] == 0:

                if (s.opMarginPred()["y_pred"].values[-1] * 3) <= (s.incomeStatement['ebitda'].values[1]
                                                                   / s.incomeStatement['revenue'].values[1]):
                    print("sold",portfolio.iloc[i]["ticker"], "p/l:", (b.mktValue/b.cost)*100 - 100,"%")
                    TestPorfolio().deleteStock(b, n)
                continue
            if pp.iloc[i]["fcf"] == 0:
                if (s.fcfProfile()[1] * 2) <= s.cashFlow['freeCashFlow'].values[3]:
                    TestPorfolio().deleteStock(b,n)
                continue
            if s.revenueProfile()[0]["increasing"] == False:
                if s.revenueProfile()[1]:
                    print("sold", portfolio.iloc[i]["ticker"], "p/l:", (b.mktValue / b.cost) * 100 - 100,"%")
                    TestPorfolio().deleteStock(b, n)
                continue

def review(n):
    df = pd.DataFrame(
        columns=['ticker', 'roe', 'roeSTD', 'opm', 'opmSTD', 'gpm', 'gpmSTD', 'gm', 'gmSTD', 'fcf', 'fcfSTD'])
    for i in dbtopd("interesting_stocks")['ticker'].values:
        try:
            s = Stock2(i, n)
            print(s.profiles())
            if s.roeProfile()['increasing'] and s.opMarginProfile()['increasing'] \
                    and s.gProfitMarginProfile()['increasing'] and s.gMarginProfile()['increasing'] and \
                    s.fcfProfile()[0]['increasing']:
                df = df.append({'ticker': i, 'roe': s.roeProfile()['increasing'], 'roeSTD': s.roeProfile()['std'],
                                'opm': s.opMarginProfile()['increasing'], 'opmSTD': s.opMarginProfile()['std'],
                                'gpm': s.gProfitMarginProfile()['increasing'],
                                'gpmSTD': s.gProfitMarginProfile()['std'],
                                'gm': s.gMarginProfile()['increasing'], 'gmSTD': s.gMarginProfile()['std'],
                                'fcf': s.fcfProfile()[0]['increasing'], 'fcfSTD': s.fcfProfile()[0]['std']},
                               ignore_index=True)
                pdtodb(df, "potential_buy_{n}".format(n=n))

        except ValueError:
            print("error")

def potential_buy(n):
    fiveStocksDict = {}
    try:
        potential_buy = dbtopd("potential_buy_{n}".format(n=n))
        if potential_buy.empty:
            pass
        else:
            if len(potential_buy) <= 5:
                for i in potential_buy["ticker"]:
                    stock = Stock2(i, n)
                    shares = Portfolio().cash * 0.1 // Stock(i, n).price
                    Borker().mktOrder(i,shares ,"BUY", stock.currency, "SMART")
            else:
                for i in potential_buy["tickers"]:
                    fiveStocksDict[i] = Stock2(i, n).intrinsic()
                    if Stock2(i, n).debtProfile() != True:
                        potential_buy = potential_buy[potential_buy['tickers'] != i].reset_index(drop=True)
                        if len(potential_buy) > 5:
                            if Stock2(i, n).retTangibeAssetsProfile()["increasing"] == 0:
                                potential_buy = potential_buy[potential_buy['tickers'] != i].reset_index(drop=True)

                if len(potential_buy) <= 5:

                    for i in potential_buy["ticker"]:
                        shares = Portfolio().cash * 0.1 // Stock(i, n).price
                        stock = Stock2(i, n)
                        Borker().mktOrder(i, shares, "BUY", stock.currency, "SMART")
                else:
                    fiveStocks = pd.DataFrame(list(fiveStocksDict.items()))
                    fiveStocks = fiveStocks.nsmallest(5, 1)
                    for i in fiveStocks[0]:
                        shares = Portfolio().cash * 0.1 // Stock(i, n).price
                        stock = Stock2(i, n)
                        Borker().mktOrder(i, shares, "BUY", stock.currency, "SMART")
    except ValueError:
        print("No undervalued stocks last year")

def porfolio_check(n):
    Portfolio().portfolioStockProfile(n)
    try:
        pp = dbtopd("portfolio_stock_profile_{n}".format(n=n))
    except ValueError:
        print("No stocks in the portfolio")
    print(pp)
    portfolio = Portfolio().portfolio
    if not portfolio.empty:
        for i in range(len(pp)):
            s = Stock2(pp.iloc[i]["ticker"], n)
            b = Stock3(portfolio.iloc[i]["ticker"], portfolio.iloc[i]["cost"],portfolio.iloc[i]["year"], n)
            if pp.iloc[i]["roe"] == 0 and pp.iloc[i]["opm"] == 0 \
                    and pp.iloc[i]["gpm"] == 0 and pp.iloc[i]["gm"] == 0 and pp.iloc[i]["fcf"] == 0 and \
                   s.revenueProfile()[0]["increasing"] == False: #and s.intrinsic() > 1:
                Borker().mktOrder(i, portfolio.iloc[i]["Shares"], "SELL", s.currency, "SMART")
                print("sold",portfolio.iloc[i]["ticker"], "p/l:", (b.mktValue/b.cost)*100 - 100 )
                continue
            if pp.iloc[i]["opm"] == 0:

                if (s.opMarginPred()["y_pred"].values[-1] * 3) <= (s.incomeStatement['ebitda'].values[1]
                                                                   / s.incomeStatement['revenue'].values[1]):
                    print("sold",portfolio.iloc[i]["ticker"], "p/l:", (b.mktValue/b.cost)*100 - 100 )
                    Borker().mktOrder(i, portfolio.iloc[i]["Shares"], "SELL", s.currency, "SMART")
                continue
            if pp.iloc[i]["fcf"] == 0:
                if (s.fcfProfile()[3] * 2) <= s.cashFlow['freeCashFlow'].values[3]:
                    Borker().mktOrder(i, portfolio.iloc[i]["Shares"], "SELL", s.currency, "SMART")
                continue
            if s.revenueProfile()[0]["increasing"] == False:
                if s.revenueProfile()[1]:
                    print("sold", portfolio.iloc[i]["ticker"], "p/l:", (b.mktValue / b.cost) * 100 - 100)
                    Borker().mktOrder(i, portfolio.iloc[i]["Shares"], "SELL", s.currency, "SMART")
                continue
