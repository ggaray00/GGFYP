from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import math


def slr(df):
    df = df[::-1]
    df = df.dropna()
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0, shuffle=False)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train.ravel())
    y_pred = regressor.predict(X_test)

    # plt.scatter(X_train, y_train, color='red')# Visualising the Test set results
    # plt.scatter(X_test, regressor.predict(X_test), color='green')
    # plt.scatter(X_test, y_test, color='orange')
    # plt.plot(X_train, regressor.predict(X_train), color='blue')
    # plt.title("Simple linear regression")
    # plt.show()
    # print("sml")
    # r2 = r2_score(y_test, y_pred)
    # print(r2)
    # r2 = r2_score(y_test, y_pred)
    # r22 = r2_score(y_pred, y_test)
    msq = mean_squared_error(y_test, y_pred)
    bic = calculate_bic(len(y_test), msq * len(y_test), len(df.columns))
    df1 = pd.DataFrame({'year': X_train.reshape(-1), 'y_pred': regressor.predict(X_train), 'y_train': y_train})
    df2 = pd.DataFrame({'year': X_test.reshape(-1), 'y_pred': y_pred, 'y_test': y_test})
    return df1, df2, bic


def slr1(df):
    df = df[::-1]
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    regressor = LinearRegression()
    regressor.fit(X, y.ravel())
    y_pred = regressor.predict(X)
    r2 = r2_score(y, y_pred)
    pred4years = pd.DataFrame(data={'year': [df['year'][0] + 1, df['year'][0] + 2,
                                             df['year'][0] + 3, df['year'][0] + 4],
                                    'y_pred': [regressor.predict([[df['year'][0] + 1]])[0],
                                               regressor.predict([[df['year'][0] + 2]])[0],
                                               regressor.predict([[df['year'][0] + 3]])[0],
                                               regressor.predict([[df['year'][0] + 4]])[0]]})
    df1 = pd.DataFrame({'year': X.reshape(-1), 'y_pred': regressor.predict(X), 'y_train': y})
    return df1, r2, pred4years


def pr(df):
    df = df[::-1]
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0, shuffle=False)
    poly_reg = PolynomialFeatures(degree=4)
    X_poly = poly_reg.fit_transform(X_train)
    lin_reg_2 = LinearRegression()
    lin_reg_2.fit(X_poly, y_train.ravel())
    y_pred = lin_reg_2.predict(poly_reg.fit_transform(X_test))
    # plt.scatter(X_train, y_train, color='red')
    # plt.scatter(X_test, y_pred, color='orange')
    # plt.scatter(X_test, y_test, color='green')
    # plt.plot(X_train, lin_reg_2.predict(poly_reg.fit_transform(X_train)), color='blue')
    # plt.title("Polynomial regression")
    # plt.show()

    msq = mean_squared_error(y_test, y_pred)
    bic = calculate_bic(len(y_test), msq * len(y_test), len(df.columns))
    df1 = pd.DataFrame({'year': X_train.reshape(-1), 'y_pred': lin_reg_2.predict(poly_reg.fit_transform(X_train)),
                        'y_train': y_train})
    df2 = pd.DataFrame({'year': X_test.reshape(-1), 'y_pred': y_pred, 'y_test': y_test})
    return df1, df2, bic


def pr1(df):
    df = df[::-1]
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    poly_reg = PolynomialFeatures(degree=4)
    X_poly = poly_reg.fit_transform(X)
    lin_reg_2 = LinearRegression()
    lin_reg_2.fit(X_poly, y.ravel())
    y_pred = lin_reg_2.predict(poly_reg.fit_transform(X))
    # plt.scatter(X, y, color='red')
    # plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color='blue')
    # plt.title("Polynomial regression")
    # plt.show()

    r2 = r2_score(y_pred, y)

    pred4years = pd.DataFrame(data={'year': [df['year'][0] + 1, df['year'][0] + 2,
                                             df['year'][0] + 3, df['year'][0] + 4],
                                    'y_pred': [lin_reg_2.predict(poly_reg.fit_transform([[df['year'][0] + 1]]))[0],
                                               lin_reg_2.predict(poly_reg.fit_transform([[df['year'][0] + 2]]))[0],
                                               lin_reg_2.predict(poly_reg.fit_transform([[df['year'][0] + 3]]))[0],
                                               lin_reg_2.predict(poly_reg.fit_transform([[df['year'][0] + 4]]))[0]]})

    df1 = pd.DataFrame({'year': X.reshape(-1), 'y_pred': lin_reg_2.predict(poly_reg.fit_transform(X)),
                        'y_train': y})

    return df1, r2, pred4years


def svr(df, kernel):
    df = df[::-1]
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    y = y.reshape(len(y), 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0, shuffle=False)
    y_trainn = y_train
    y_testt = y_test
    X_testt = X_test
    sc_X = StandardScaler()
    sc_y = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    y_train = sc_y.fit_transform(y_train)
    regressor = SVR(kernel=kernel)
    regressor.fit(X_train, y_train.ravel())
    y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(X_test)).reshape(1, -1))

    # plt.scatter(sc_X.inverse_transform(X_train).reshape(-1), sc_y.inverse_transform(y_train).reshape(-1), color='red')
    # plt.scatter(X_test, y_test, color='orange')
    # plt.scatter(X_test,y_pred,color ='green')
    # plt.title("Support Vector regression  " + kernel)
    # plt.plot(sc_X.inverse_transform(X_train).reshape(-1),
    #          sc_y.inverse_transform(regressor.predict(X_train).reshape(-1, 1)).reshape(-1), color='blue')
    # plt.show()
    msq = mean_squared_error(y_test, y_pred.reshape(-1))
    bic = calculate_bic(len(y_test), msq * len(y_test), len(df.columns))
    df1 = pd.DataFrame({'year': sc_X.inverse_transform(X_train).reshape(-1),
                        'y_pred': sc_y.inverse_transform(regressor.predict(X_train).reshape(1, -1)).reshape(-1),
                        'y_train': y_trainn.reshape(-1)})
    df2 = pd.DataFrame({'year': X_testt.reshape(-1), 'y_pred': y_pred.reshape(-1), 'y_test': y_testt.reshape(-1)})

    return df1, df2, bic


def svr1(df, kernel):
    df = df[::-1]
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    y = y.reshape(len(y), 1)
    yy = y
    sc_X = StandardScaler()
    sc_y = StandardScaler()
    X = sc_X.fit_transform(X)
    y = sc_y.fit_transform(y)
    regressor = SVR(kernel=kernel)
    regressor.fit(X, y.ravel())
    y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(X)).reshape(1, -1))

    # plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color='red')
    # plt.title("Support Vector regression 1  " +  kernel)
    # plt.plot(sc_X.inverse_transform(X).reshape(-1),
    #          sc_y.inverse_transform(regressor.predict(X).reshape(-1, 1)).reshape(-1), color='blue')
    # plt.show()

    r2 = r2_score(yy.reshape(-1), y_pred.reshape(-1))

    df1 = pd.DataFrame({'year': sc_X.inverse_transform(X).reshape(-1),
                        'y_pred': sc_y.inverse_transform(regressor.predict(X).reshape(1, -1)).reshape(-1),
                        'y_train': yy.reshape(-1)})

    pred4years = pd.DataFrame(data={'year': [df['year'][0] + 1, df['year'][0] + 2, df['year'][0] + 3,
                                             df['year'][0] + 4],
    'y_pred': [sc_y.inverse_transform(regressor.predict(sc_X.transform([[df['year'][0] + 1]])).reshape(1, -1))[0][0],
               sc_y.inverse_transform(regressor.predict(sc_X.transform([[df['year'][0] + 2]])).reshape(1,-1))[0][0],
               sc_y.inverse_transform(regressor.predict(sc_X.transform([[df['year'][0] + 3]])).reshape(1,-1))[0][0],
               sc_y.inverse_transform(regressor.predict(sc_X.transform([[df['year'][0] + 4]])).reshape(1,-1))[0][0]]})
    return df1, r2, pred4years


def bestFitModel(df, title):
    df = df.dropna()
    simplesplit = slr(df)
    simple = slr1(df)
    polysplit = pr(df)
    poly = pr1(df)
    svrbfsplit = svr(df, 'rbf')
    svrbf = svr1(df, 'rbf')
    svpolysplit = svr(df, 'poly')
    svpoly = svr1(df, 'poly')
    svsigsplit = svr(df, 'sigmoid')
    svsig = svr1(df, 'sigmoid')
    model = simplesplit
    model1 = simple
    a = "simple"
    if model[2] > polysplit[2]:
        model = polysplit
        model1 = poly
        a = "poly"
    if model[2] > svrbfsplit[2]:
        model = svrbfsplit
        model1 = svrbf
        a = "rbf"
    if model[2] > svpolysplit[2]:
        model = svpolysplit
        model1 = svpoly
        a = "svpoly"
    if model[2] > svpolysplit[2]:
        model = svsigsplit
        model1 = svsig
        a = "svsig"

    # plt.scatter(model[0]['year'], model[0]['y_train'], color='red')
    # plt.scatter(model[1]['year'], model[1]['y_pred'], color='green')
    # plt.scatter(model[1]['year'], model[1]['y_test'], color='orange')
    # plt.plot(model[0]['year'], model[0]['y_pred'], color='blue')
    # plt.title("Train/Test model for  " + title + a)
    # plt.show()
    # model1[2].plot.scatter(x='year', y='y_pred', color='green')
    # plt.plot(model1[0]['year'], model1[0]['y_pred'], color='blue')
    # plt.scatter(model1[0]['year'], model1[0]['y_train'], color='red')
    # plt.title("future " + title + " predictions" + a)
    # plt.show()


    return model1, model


def std(df):
    X = df.iloc[:, 1:].values
    X = StandardScaler().fit_transform(X)
    c = 0
    for i in X:
        c = c + (i[1] - i[0]) ** 2
    return math.sqrt(c / len(X))


def calculate_bic(n, ss, p):
    return n * math.log(ss) - n * math.log(n) + math.log(n) * p
