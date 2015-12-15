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
from sklearn.ensemble import ExtraTreesClassifier

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

def getAllPoints():
    global conn
    global cur
    cur.execute("SELECT * FROM points LIMIT 0, 100000")
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
    minLat = lat - 0.02
    maxLat = lat + 0.02
    minLong = long - 0.02
    maxLong = long + 0.02
    # print("SELECT * FROM point_summary JOIN routeMetadata ON metaid = routeMetadata.id where"
    #             " latitude<" + str(maxLat) + " AND latitude>" + str(minLat) + " AND longitude<" + str(maxLong) + " AND longitude>" + str(minLong))
    cur.execute("SELECT * FROM point_summary JOIN routeMetadata ON metaid = routeMetadata.id where"
                " latitude<" + str(maxLat) + " AND latitude>" + str(minLat) + " AND longitude<" + str(maxLong) + " AND longitude>" + str(minLong))
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

def ForestFeatures():
    global idCounter
    global cur
    loadDB()
    points = getAllPoints()
    X = []
    Y = []
    for point in points:
        slope = point['vehicleSpeed'] / point['enginespeed']
        sql = "Select * from gears ORDER BY ABS( slope - " + str(slope) + " ) ASC LIMIT 1"
        cur.execute(sql)
        res = cur.fetchall()
        for re in res:
            break
        point['gear'] = re['gear']
        if (point['gear'] == -1):
            # Ignore point, appropriate gear not found.
            continue
        Y.append(getFuleRateLabel(point['fuelrate']))
        del (point['id'])
        del (point['metaId'])
        del (point['fuelrate'])
        del (point['time'])
        X.append(point)

    i = 0
    for point in X:
        if (i == 1):
            break
        print(point)
        i += 1
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

    forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

    forest.fit(X, Y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    labels = ["Enginespeed", "Gear", "Latitude", "Longitude", "Vechilespeed"]
    print(indices)
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
           color="r", yerr=std[indices], align="center")
    finallabels = []
    for ind in np.nditer(indices):
        finallabels.append(labels[ind])
    # The label is in reverse order for some reason.
    finallabels.reverse()
    plt.xticks(range(X.shape[1]), finallabels)
    plt.xlim([-1, X.shape[1]])
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.show()
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
        if (len(points) < 100):
            continue
        else:
            print(location['latitude'], location['longitude'])
            print("number of points - " + str(len(points)))
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
            del (point['truck'])
            del (point['stamp'])
            del (point['routeMetadata.id'])
            del (point['latitude'])
            del (point['longitude'])
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
ForestFeatures()