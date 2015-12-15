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

conn = None
cur = None
idCounter = 0

'''
Gets the next set of up to 1000 points where the truck was going > 5mph
'''
def getNextPoints(metaId):
    global cur
    global conn
    global idCounter

    cur.execute("SELECT * FROM points WHERE metaId = %s AND id > %s AND vehicleSpeed > 10 ORDER BY id ASC LIMIT 5000", (metaId, idCounter));
    if cur.rowcount == 0:
        #This timestamp doesn't have any points, apparently
        return None

    points = cur.fetchall()
    idCounter = points[len(points)-1]['id']
    print("idCounter set to "+str(idCounter))
    return points

def uniqueDrivers():
    loadDB()
    global conn
    global cur
    cur.execute("SELECT DISTINCT(driver) FROM routeMetadata")
    drivers = cur.fetchall()
    for driver in drivers:
        print("INSERT INTO drivers VALUES (NULL,'" + driver['driver'] + "');")


def getAllSummary():
    global conn
    global cur
    cur.execute("SELECT * FROM point_summary")
    return cur.fetchall()

def getAllSummaryDriver():
    global conn
    global cur
    cur.execute("SELECT * FROM point_summary JOIN routeMetadata ON metaid = routeMetadata.id")
    return cur.fetchall()

def getAllSummaryDriverAndTruck():
    global conn
    global cur
    cur.execute("SELECT * FROM point_summary JOIN routeMetadata ON metaid = routeMetadata.id where truck IS NOT NULL")
    return cur.fetchall()

def getConstLoc():
    global conn
    global cur
    cur.execute("SELECT DISTINCT latitude, longitude FROM point_summary")
    return cur.fetchall()

def getAllPointsForLoc(lat, long):
    global conn
    global cur
    cur.execute("SELECT * FROM point_summary JOIN routeMetadata ON metaid = routeMetadata.id where truck is NOT NULL AND latitude=" + str(lat) + " AND longitude=" + str(long))
    return cur.fetchall()

def getDriverID(name):
    global conn
    global cur
    cur.execute("SELECT id FROM drivers where name = '" + name + "'")
    rows = cur.fetchall()
    for row in rows:
        return row['id']


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

def getFuleRateLabel(rate):
    if (rate < 10):
        return 1
    if rate < 20:
        return 2
    if rate < 30:
        return 3
    if rate < 40:
        return 4
    if rate < 50:
        return 5
    return 6

def main():
    global idCounter
    loadDB()
    points = getAllSummary()
    X = []
    Y = []
    i = 0
    for point in points:
        if (i == 0):
            print(point)
            i += 1
        Y.append(getFuleRateLabel(point['fuelrate']))
        del (point['id'])
        del (point['metaId'])
        del (point['fuelrate'])
        del (point['day'])
        X.append(point)
    X = pd.DataFrame(X)
    X = X.as_matrix()
    X = StandardScaler().fit_transform(X)

    i = 0
    for point in X:
        if (i == 1):
            break
        print(point)
        i += 1

    clf = DecisionTreeClassifier()
    clf.fit(X, Y)
    print(clf.feature_importances_)
    closeDB()

def classifyWithDriver():
    global idCounter
    loadDB()
    points = getAllSummaryDriver()
    X = []
    Y = []
    i = 0
    for point in points:
        point['driverid'] = getDriverID(point['driver'])
        if (i == 0):
            print(point)
            i += 1
        Y.append(getFuleRateLabel(point['fuelrate']))
        del (point['id'])
        del (point['metaId'])
        del (point['fuelrate'])
        del (point['day'])
        del (point['driver'])
        del (point['stamp'])
        del (point['truck'])
        X.append(point)
    X = pd.DataFrame(X)
    X = X.as_matrix()

    i = 0
    for point in X:
        if (i == 1):
            break
        print(point)
        i += 1
    X = StandardScaler().fit_transform(X)

    i = 0
    for point in X:
        if (i == 1):
            break
        print(point)
        i += 1

    clf = DecisionTreeClassifier()
    clf.fit(X, Y)
    print(clf.feature_importances_)
    closeDB()

def classifyWithDriverAndTruck():
    global idCounter
    loadDB()
    points = getAllSummaryDriverAndTruck()
    X = []
    Y = []
    trucks = []
    i = 0
    for point in points:
        point['driverid'] = getDriverID(point['driver'])
        if (i == 0):
            print(point)
            i += 1
        Y.append(getFuleRateLabel(point['fuelrate']))
        trucks.append(point['truck'])
        del (point['id'])
        del (point['metaId'])
        del (point['fuelrate'])
        del (point['day'])
        del (point['driver'])
        del (point['stamp'])
        del (point['routeMetadata.id'])
        X.append(point)

    le = preprocessing.LabelEncoder()
    le.fit(trucks)
    for point in X:
        point['truck'] = le.transform(point['truck'])
    X = pd.DataFrame(X)
    X = X.as_matrix()

    i = 0
    for point in X:
        if (i == 1):
            break
        print(point)
        i += 1
    X = StandardScaler().fit_transform(X)

    i = 0
    for point in X:
        if (i == 1):
            break
        print(point)
        i += 1

    clf = DecisionTreeClassifier()
    clf.fit(X, Y)
    print(clf.feature_importances_)
    closeDB()

def classifyConstLoc():
    global idCounter
    loadDB()
    locations = getConstLoc()
    for location in locations:
        points = getAllPointsForLoc(location['latitude'], location['longitude'])
        if (len(points) == 0):
            continue
        else:
            print(location['latitude'], location['longitude'])
            print("number of points - " + str(len(points)))
        X = []
        Y = []
        trucks = []
        i = 0
        for point in points:
            point['driverid'] = getDriverID(point['driver'])
            if (i == 0):
                print(point)
                i += 1
            Y.append(getFuleRateLabel(point['fuelrate']))
            trucks.append(point['truck'])
            del (point['id'])
            del (point['metaId'])
            del (point['fuelrate'])
            del (point['day'])
            del (point['driver'])
            del (point['stamp'])
            del (point['routeMetadata.id'])
            X.append(point)

        le = preprocessing.LabelEncoder()
        le.fit(trucks)
        for point in X:
            point['truck'] = le.transform(point['truck'])
        X = pd.DataFrame(X)
        X = X.as_matrix()

        i = 0
        for point in X:
            if (i == 1):
                break
            print(point)
            i += 1
        X = StandardScaler().fit_transform(X)

        i = 0
        for point in X:
            if (i == 1):
                break
            print(point)
            i += 1

        clf = DecisionTreeClassifier()
        clf.fit(X, Y)
        print(clf.feature_importances_)
    closeDB()
classifyConstLoc()