import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors
import pymysql
from config import *
from matplotlib.collections import LineCollection
from matplotlib  import cm

from sklearn.cluster import DBSCAN, MeanShift, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph, DistanceMetric
from sklearn.manifold import SpectralEmbedding
from scipy.spatial.distance import cosine,mahalanobis

import time
from scipy.stats import linregress

import statsmodels.formula.api as sm

from collections import OrderedDict
import datetime

def loadDB():
	global conn
	global cur
	global idCounter
	idCounter = 0;
	conn = pymysql.connect(host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER, passwd=MYSQL_PASSWORD, db='mysql', charset='utf8')

	cur = conn.cursor(pymysql.cursors.DictCursor)
	cur.execute("USE trucking")

def closeDB():
	conn.close()

def getAllMetadata():
	global conn
	global cur
	cur.execute("SELECT * FROM routeMetadata where id > 1933 ORDER BY id DESC")
	return cur.fetchall()

def getTrucks():
	global cur
	global conn
	cur.execute("SELECT * FROM routeMetadata WHERE truck IS NOT NULL GROUP BY truck")
	return cur.fetchall()

def getDrivers():
	global cur
	global conn
	cur.execute("SELECT * FROM routeMetadata GROUP BY driver")
	return cur.fetchall()

def getPointsForTruck(truck, driver=None):
	global cur
	global conn
	if driver is not None:
		cur.execute("SELECT points.id as id, enginespeed, vehicleSpeed FROM points JOIN routeMetadata ON points.metaId = routeMetadata.id WHERE routeMetadata.driver = %s AND routeMetadata.truck IS NULL", (driver))
	else:	
		cur.execute("SELECT points.id as id, enginespeed, vehicleSpeed FROM points JOIN routeMetadata ON points.metaId = routeMetadata.id WHERE routeMetadata.truck = %s", (truck))
	return cur.fetchall()

#We can't get the full range of gears from a single sample
#This updates the sample with the "more correct" gear calculated
def updateShift(id, gear):
	cur.execute("UPDATE shifting SET gear = %s WHERE id = 1")

def getAllShiftTrucks():
	cur.execute("SELECT truck FROM shifting WHERE truck IS NOT NULL AND truck != \"\" GROUP BY truck")
	if cur.rowcount == 0:
		return None
	return cur.fetchall()
'''
Assigning shift patterns based on driver is less preferable to assigning them based
on truck. Avoid drivers with trucks because of this
'''
def getAllShiftDrivers():
	cur.execute("SELECT shifting.driver FROM shifting JOIN routeMetadata ON shifting.driver = routeMetadata.driver WHERE shifting.driver IS NOT NULL AND shifting.driver != \"\" AND routeMetadata.truck IS NULL GROUP BY shifting.driver")
	if cur.rowcount == 0:
		return None
	return cur.fetchall()

def getShiftsWithDriver(driver):
	cur.execute("SELECT * FROM shifting WHERE driver = %s", str(driver))
	if cur.rowcount == 0:
		return None
	return cur.fetchall()

def getTruckOnlyShifts(truck):
	cur.execute("SELECT * FROM shifting WHERE truck = %s", str(truck))
	if cur.rowcount == 0:
		return None
	return cur.fetchall()

def getShiftsWithTruck(truck):
	cur.execute("SELECT routeMetadata.truck, shifting.* FROM shifting JOIN routeMetadata ON shifting.driver = routeMetadata.driver AND shifting.day = DATE(routeMetadata.datetime) WHERE routeMetadata.truck = %s", (truck))
	if cur.rowcount == 0:
		return None
	return cur.fetchall()

def getShiftsWithoutTruck():
	cur.execute("SELECT routeMetadata.truck, shifting.* FROM shifting JOIN routeMetadata ON shifting.driver = routeMetadata.driver AND shifting.day = DATE(routeMetadata.datetime) WHERE routeMetadata.truck IS NULL")
	if cur.rowcount == 0:
		return None
	return cur.fetchall()

def getShifts():
	global cur
	global conn

	cur.execute("SELECT * FROM shifting")
	return cur.fetchall()

def getGears(truck, driver=None):
	global cur
	global conn
	if driver is not None:
		cur.execute("SELECT * FROM gears WHERE driver = %s", (driver))
	else:
		cur.execute("SELECT * FROM gears WHERE truck = %s", (truck))
	if cur.rowcount == 0:
		return None
	return cur.fetchall()

def addGear(truck, gear, slope, driver=None):
	global cur
	global conn
	if driver is not None:
		print("Adding gear for driver!")
		cur.execute("SELECT * FROM gears WHERE driver = %s AND gear = %s", (driver, int(gear)))
		if cur.rowcount == 0:
			cur.execute("INSERT INTO gears (driver, gear, slope) VALUES (%s, %s, %s)", (driver, int(gear), float(slope)))
		else:
			cur.execute("UPDATE gears SET slope = %s WHERE driver = %s AND gear = %s", (float(slope), driver, int(gear)))

	else:
		print("Adding gear! "+str(truck)+" "+str(gear)+" "+str(slope))
		cur.execute("SELECT * FROM gears WHERE truck = %s AND gear = %s", (truck, int(gear)))
		if cur.rowcount == 0:
			cur.execute("INSERT INTO gears (truck, gear, slope) VALUES (%s, %s, %s)", (truck, int(gear), float(slope)))
		else:
			cur.execute("UPDATE gears SET slope = %s WHERE truck = %s AND gear = %s", (float(slope), truck, int(gear)))
	conn.commit()


def populateDates():
	metadata = getAllMetadata()
	for data in metadata:
		print("Updating: "+str(data['id']))

		#year, month, day, hour=0, minute=0, second=0, microsecond=0, tzinfo=None
		cur.execute("UPDATE routeMetadata SET datetime = %s WHERE id = %s", (sqlDateTime, data['id']))
		conn.commit()

def find_nearest_gear(gears,array,value):
	#If it's less than half the lowest gear value, return -2
	if value < (array.argmin()/2):
		return -2

	idx = (np.abs(array-value)).argmin()

	return gears[idx]['gear']

def groupAndUpdate(dataDict, uniqueLabels, truck, driver=None):

	for label in uniqueLabels:
		shiftArr = dataDict[label]
		#print([x['slope'] for x in shiftArr])
		#Get weighted average of the slope, based on the r2 value
		avgSlope = np.average(np.array([x['slope'] for x in shiftArr]), weights=np.array([x['r2'] for x in shiftArr]))
		addGear(truck, label, avgSlope, driver=driver)

def doClustering(shifts, slopes, truck, driver=None):
	slopes = np.array(slopes)
	#slopes = slopes + (1/1348)*intercepts
	print("Clustering data")
	#Cluster on *just* the slopes
	#X = StandardScaler().fit_transform(slopes)
	#DBSCAN(eps=0.5, min_samples=5, metric='euclidean', algorithm='auto', leaf_size=30, p=None, random_state=None)
	db = DBSCAN(eps=0.001, min_samples=1).fit(slopes.reshape(-1, 1))
	labels = db.labels_

	uniqueLabels = list(sorted(set(labels)))
	dataDict = dict()
	for label in uniqueLabels:
		dataDict[label] = []
	for i in range(len(shifts)):
		dataDict[labels[i]].append(shifts[i])

	dataDict = OrderedDict(sorted(dataDict.items(), key=lambda t: np.mean([x['slope'] for x in t[1]]), reverse=True))

	sortedLabels = OrderedDict()
	newLabels = []
	returnShifts = []
	i = 0
	for key in dataDict.keys():
		sortedLabels[i] = dataDict[key]
		newLabels.extend([i]*len(dataDict[key]))
		returnShifts.extend(dataDict[key])
		i += 1

	groupAndUpdate(dataDict, uniqueLabels, truck, driver=driver)

	return returnShifts, newLabels


def twoDPlot(slopes, intercepts, labels, title = "Shift clustering"):
	print(title)
	#labelSet = sort(set(labels))
	cmap = plt.cm.prism
	cmaplist = [cmap(i) for i in range(cmap.N)]
	cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
	bounds = np.linspace(-1, 16, 18)
	norm = mpl.colors.BoundaryNorm(bounds, cmap.N)


	plt.suptitle(title, fontsize=20)
	plt.scatter(slopes, intercepts, marker='o', c=labels, cmap=cmap, norm=norm)
	plt.xlabel('Slope', fontsize=18)
	plt.ylabel('Intercept', fontsize=16)
	plt.colorbar(ticks=bounds)
	plt.show()


def updatePoints(idGearTuple):
	print("UPDATING Len of tuples is: "+str(len(idGearTuple)))
	try:
		cur.executemany("UPDATE points SET gear = %s WHERE id = %s ", idGearTuple)
		conn.commit()
	except TypeError:
		print("ERROR")
		print(idGearTuple)

def addGearsToPoints(pointsByDriver=False):
	#(vehicleSpeed/engineSpeed) is test slope

	if pointsByDriver:
		drivers = getDrivers()

		for driver in drivers:
			print(str(driver['driver']))
			gears = getGears(None, driver=driver['driver'])

			idGearTuples = []
			if gears is not None:
				maxGear = -2
				gearSlopes = np.array([x['slope'] for x in gears])
				print(gearSlopes)
				points = getPointsForTruck(None, driver=driver['driver'])
				for point in points:
					if point['enginespeed'] == float(0.0):
						gearId = maxGear
					else:
						gearId = find_nearest_gear(gears, gearSlopes, float(float(point['vehicleSpeed'])/float(point['enginespeed'])))
					idGearTuples.append((gearId,point['id']))

				updatePoints(idGearTuples)

	else:
		trucks = getTrucks()

		for truck in trucks:
			print(str(truck['truck']))
			gears = getGears(truck['truck'])
			print(gears)
			idGearTuples = []
			if gears is not None:
				maxGear = -2
				gearSlopes = np.array([x['slope'] for x in gears])
				print(gearSlopes)
				points = getPointsForTruck(truck['truck'])
				for point in points:
					if point['enginespeed'] == float(0.0):
						gearId = maxGear
					else:
						gearId = find_nearest_gear(gears, gearSlopes, float(point['vehicleSpeed']/point['enginespeed']))
					idGearTuples.append((gearId,point['id']))

				updatePoints(idGearTuples)


def main():
	'''
	shifts = getShiftsWithoutTruck()
	slopes = [x['slope'] for x in shifts]
	intercepts = [x['intercept'] for x in shifts]
	shifts, labels = doClustering(shifts, slopes, "")
	slopes = [x['slope'] for x in shifts]
	intercepts = [x['intercept'] for x in shifts]
	twoDPlot(slopes, intercepts, labels, "Shifting for unidentified trucks")

'''
	trucks = getTrucks()

	for truck in trucks:
		weights = []
		shifts = getShiftsWithTruck(truck['truck'])
		if shifts is not None:
			slopes = [x['slope'] for x in shifts]
			intercepts = [x['intercept'] for x in shifts]
			shifts, labels = doClustering(shifts, slopes, truck['truck'])
			slopes = [x['slope'] for x in shifts]
			intercepts = [x['intercept'] for x in shifts]
			twoDPlot(slopes, intercepts, labels, title = truck['truck'])

			

def runDriverOnlyShifts():
	drivers = getAllShiftDrivers()
	for driver in drivers:
		shifts = getShiftsWithDriver(driver['driver'])
		if shifts is not None and len(shifts) > 10:
			slopes = [x['slope'] for x in shifts]
			intercepts = [x['intercept'] for x in shifts]
			shifts, labels = doClustering(shifts, slopes, None, driver=driver['driver'])
			slopes = [x['slope'] for x in shifts]
			intercepts = [x['intercept'] for x in shifts]
			twoDPlot(slopes, intercepts, labels, "Shifting for driver "+str(driver['driver']))
		else: 
			print("Driver "+str(driver['driver']+" is not known"))


def runTruckOnlyShifts():
	trucks = getAllShiftTrucks()
	if trucks is not None:
		for truck in trucks:
			shifts = getTruckOnlyShifts(truck['truck'])
			if shifts is not None:
				slopes = [x['slope'] for x in shifts]
				intercepts = [x['intercept'] for x in shifts]
				shifts, labels = doClustering(shifts, slopes, truck['truck'])
				slopes = [x['slope'] for x in shifts]
				intercepts = [x['intercept'] for x in shifts]
				twoDPlot(slopes, intercepts, labels, "Shifting for unidentified trucks")
			else: 
				print("Truck "+str(truck['truck']+" is not known"))

'''
Once the shift equations have been created in the "shifting" table, use this to assign
the appropriate points
'''
loadDB()
#populateDates()
#runDriverOnlyShifts()
#addGearsToPoints(pointsByDriver=True)
main()
closeDB()
