import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pymysql
from config import *


from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import scipy
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
import statsmodels.api as sm


conn = None
cur = None
idCounter = 0

def loadDB():
    global conn
    global cur
    global idCounter
    idCounter = 0
    conn = pymysql.connect(host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER, passwd=MYSQL_PASSWORD, db='mysql', charset='utf8')

    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute("USE " + MYSQL_DATABASE)

def closeDB():
    conn.close()



def getPointSummaryWithAcceleration():
   global cur
   global conn
   cur.execute("SELECT AVG(gearratio) as gearratio, AVG(fuelrate) as fuelrate, AVG(vehiclespeed) as vehiclespeed, AVG(enginespeed) as enginespeed, AVG(acceleration) as acceleration FROM (SELECT (vehiclespeed/enginespeed) as gearratio, fuelrate, vehiclespeed, enginespeed, acceleration FROM point_summary WHERE acceleration IS NOT NULL AND vehiclespeed < 70) as a WHERE gearratio < 0.06 GROUP BY TRUNCATE(acceleration, 2), TRUNCATE(gearratio,3)")
   return cur.fetchall()

def getMpgForAcceleration(acc):
   global cur
   global conn
   minacc = acc - 0.3
   maxacc = acc + 0.3
   cur.execute("SELECT (vehiclespeed/enginespeed) as gearratio, fuelrate, vehiclespeed, enginespeed, acceleration from point_summary WHERE (vehiclespeed/enginespeed) < 0.06 AND acceleration > " + str(minacc) + " AND acceleration < " + str(maxacc))
   return cur.fetchall()


def scatterGearRatios():
    loadDB()
    points = getPointSummaryWithAcceleration()

    gearratios = []
    fuelrates = []
    vehiclespeeds = []
    accelerations = []
    enginespeeds = []
    mpgs = []
    X = []

    for point in points:
        val = float((point['vehiclespeed'])/(float(point['fuelrate']))*3.7854)
        if val < 10.0:
            fuelrates.append(point['fuelrate'])
            accelerations.append(point['acceleration'])
            gearratios.append(point['gearratio'])
            vehiclespeeds.append(point['vehiclespeed'])
            enginespeeds.append(point['enginespeed'])
            mpgs.append(float((point['vehiclespeed'])/(float(point['fuelrate']))*3.7854))
            val = float((point['vehiclespeed'])/(float(point['fuelrate']))*3.7854)
            mpgs.append(val)
            X.append([point['acceleration'], val])

    # print(mpgs)
    # # print(accelerations)
    # # print(X)
    # exit()
    y = gearratios
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8)
    # train_r2 = []
    # test_r2 = []
    # for degree in range(1,10):
    #     # model = make_pipeline(PolynomialFeatures(degree), Ridge())
    #     model = DecisionTreeRegressor(max_depth=degree)
    #     model.fit(X_train, y_train)
    #     test_r2.append(model.score(X_test, y_test))
    #     train_r2.append(model.score(X_train, y_train))
    #
    # plt.plot(np.arange(1, 10), train_r2, color='green', label='train')
    # plt.plot(np.arange(1, 10), test_r2, color='red', label='test')
    # plt.title("Max-depth vs R2 - Finding optimal degree \n Tree regression")
    # plt.ylabel('R2')
    # plt.xlabel('Max Depth')
    # plt.legend(loc='lower left')
    # plt.show()

    model = make_pipeline(PolynomialFeatures(2), Ridge())
    # model = DecisionTreeRegressor(max_depth=2)
    model.fit(X, y)
    # for i in range (0, 50):
    #     for j in range(1, 50):
    i = 0.1
    for j in range(10):
        points = getMpgForAcceleration(i)
        mpgs = []
        for point in points:
            val = float((point['vehiclespeed'])/(float(point['fuelrate']))*3.7854)
            if (val > 10):
                # Outlier, Ignore.
                continue
            mpgs.append(val)
        top = mpgs.copy()
        top.sort()
        top.reverse()
        length = round(len(top)/10)
        # print(top[:length])
        top = np.mean(top[:length])
        print(i, top, model.predict([i, top]))
        i += 0.1
    # predict = [[0.1, top],
    #            [0.2, top],
    #            [0.3, top],
    #            [0.4, top],
    #            [0.5, top],
    #            [0.6, top],
    #            [0.7, top],
    #            [0.8, top],
    #            [0.9, top],
    #            [1.0, top]]
    # print(model.predict(predict))

    closeDB()

scatterGearRatios()