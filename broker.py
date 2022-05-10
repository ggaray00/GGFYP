from ib_insync import *
import pandas as pd



class Borker:


    def mktOrder(self, ticker, amount, action, currency, exchange):
        ib = IB()
        ib.connect('127.0.0.1', 4002, clientId=1)
        order = ib.placeOrder(Stock(ticker, exchange, currency), MarketOrder(action, amount))
        ib.disconnect()
        return order

    def getPortfolio(self):
        ib = IB().connect('127.0.0.1', 4002, clientId=1)
        portfolio = pd.DataFrame(columns=['Ticker', 'Exchange', 'Currency', 'Shares', 'mktPrice',
                                          'mktValue', 'avCost', 'urPnl', 'rPnl'])
        for i in ib.portfolio():
            portfolio = portfolio.append({'Ticker': i[0].symbol, 'Exchange': i[0].primaryExchange,
                                          'Currency': i[0].currency, 'Shares': i.position, 'mktPrice': i.marketPrice,
                                          'mktValue': i.marketValue, 'avCost': i.averageCost,
                                          'urPnl': i.unrealizedPNL, 'rPnl': i.realizedPNL}, ignore_index=True)
        ib.disconnect()
        return portfolio

    def getAccountSummary(self):
        ib = IB().connect('127.0.0.1', 4002, clientId=1)
        accountSummary = pd.DataFrame(columns=['tag', 'value', 'currency'])
        for i in ib.accountSummary():
            accountSummary = accountSummary.append({'tag': i.tag, 'value': i.value, 'currency': i.currency},
                                                   ignore_index=True)
        ib.disconnect()
        return accountSummary

    def whatIfOrder(self, ticker, amount, action,currency):
        ib = IB()
        ib.connect('127.0.0.1', 4002, clientId=1)
        order = ib.whatIfOrder(Stock(ticker, 'SMART', currency), MarketOrder(action, amount))
        ib.disconnect()
        return order

    def order(self):
        ib = IB()
        ib.connect('127.0.0.1', 4002, clientId=1)
        order = ib.orders()
        ib.disconnect()
        return order






# Borker().mktOrder("APPLE",10,"SELL","USD","SMART")

