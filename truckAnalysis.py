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

from sklearn.ensemble import BaggingClassifier
from sklearn.cluster import DBSCAN, dbscan, AgglomerativeClustering, SpectralClustering, ward_tree, Birch
from sklearn.metrics import r2_score
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph, DistanceMetric
from sklearn.manifold import SpectralEmbedding
from scipy.spatial.distance import cosine,mahalanobis

import time
from scipy.stats import linregress

import statsmodels.formula.api as sm

from collections import OrderedDict
import datetime

conn = None
cur = None
idCounter = 0
metaId = 0
minMetaId = 4406

'''
Gets the next set of up to 1000 points where the truck was going > 5mph
'''
def getNextPoints(metaId):
	global cur
	global conn
	global idCounter


	cur.execute("SELECT * FROM points WHERE metaId = %s AND id > %s AND vehicleSpeed > 5 ORDER BY id ASC LIMIT 5000", (metaId, idCounter));
	if cur.rowcount == 0:
		#This timestamp doesn't have any points, apparently
		return None

	points = cur.fetchall()
	idCounter = points[len(points)-1]['id']
	print("idCounter set to "+str(idCounter))
	return points
	
'''
Gets a set of all driver/dates for which we have legs
'''
def getDriverDates():
	global cur
	global conn

	cur.execute("SELECT SUBSTR(stamp, 1, 8) as stamp, driver, truck FROM routeMetadata WHERE id > %s GROUP BY driver, SUBSTR(stamp, 1, 8)", (int(minMetaId)))
	return cur.fetchall()

def getPointsForDriverDate(stampPrefix, driver):
	global cur
	global conn
	global minMetaId
	#print("Getting points "+str(stampPrefix)+" "+str(driver))
	allPoints = []
	cur.execute("SELECT COUNT(*) as count FROM points JOIN routeMetadata ON points.metaId = routeMetadata.id WHERE SUBSTR(routeMetadata.stamp, 1, 8) = %s AND routeMetadata.driver = %s AND routeMetadata.id > %s", (stampPrefix, driver, int(minMetaId)))
	count = cur.fetchone()['count']
	if count < 2000:
		return None

	for mph in range(1,14):
		cur.execute("SELECT points.* FROM points JOIN routeMetadata ON points.metaId = routeMetadata.id WHERE points.vehicleSpeed >= %s AND points.vehicleSpeed < %s AND SUBSTR(routeMetadata.stamp, 1, 8) = %s AND routeMetadata.driver = %s LIMIT 120", (mph*5, (mph+1)*5, stampPrefix, driver))
		mphCount = cur.rowcount
		if mphCount < 120:
			return None
		allPoints.extend(cur.fetchall())
	print("Retrieved Data")
	return allPoints

'''
Gets 3000 points, evenly distributed among a range of vehicle speeds from 5MPH to 70MPH
200 points per 5MPH increment
'''
def getEvenSpreadPoints(metaId):
	print("Searching for points: "+str(metaId))
	cur.execute("SELECT COUNT(*) as count FROM points WHERE metaId = %s", (metaId))
	count = cur.fetchone()['count']
	if(count < 2000):
		return None

	allPoints = []
	for mph in range(1,15):
		cur.execute("SELECT * FROM points WHERE metaId = %s AND vehicleSpeed >= %s AND vehicleSpeed < %s LIMIT 50", (metaId, mph*5, (mph+1)*5));
		if cur.rowcount < 40:
			#Look for "perfect sets" this time around
			return None
		allPoints.extend(cur.fetchall())

	print("SUCCESS!")
	return allPoints


def storeShift(position, slope, intercept, slopeConf, interceptConf, clustMeanX, clustMeanY, clustVals, r2score):
	global driver
	global date
	global cur
	global conn
	sqlDate = datetime.date(int(date[:4]), int(date[4:6]), int(date[6:8]))
	cur.execute("SELECT * FROM shifting WHERE driver = %s AND day = %s AND position = %s", (driver, sqlDate, position))
	if cur.rowcount != 0:
		print("Already present!")
		return
	cur.execute("INSERT INTO shifting (driver, day, position, slope, intercept, slopeConf, interceptConf, clustMeanX, clustMeanY, clustVals, r2) VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s,%s,%s)", (driver, sqlDate, int(position), float(slope), float(intercept), float(slopeConf), float(interceptConf), float(clustMeanX), float(clustMeanY), int(clustVals),float(r2score)))
	conn.commit()

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

'''
Tool used to provide logic for deciding if two linRegress arrays 
represent the same line
'''
def isSame(lower, higher):
	###TEST TECHNIQUE####
	meanSlope = (lower[5][1] - higher[5][1])/(lower[5][0] - higher[5][0])
	if(meanSlope < 0):
		return False
	meanTestCoefLower = (1-float(lower[4][0])/.2)*4
	meanTestCoefHigher = (1-float(higher[4][0])/.2)*4
	if abs(meanSlope - lower[0]) < meanTestCoefLower*lower[4][0] or abs(meanSlope - higher[0]) < meanTestCoefLower*lower[4][0]:
		print("Mean test passed!")
		return True
	if abs(meanSlope - lower[0]) < meanTestCoefHigher*higher[4][0] or abs(meanSlope - higher[0]) < meanTestCoefHigher*higher[4][0]:
		print("Mean test passed!")
		return True
	#Higher confidence interval coefficient = less certain that groups are the same
	#I've found that slope is a better indicator than intercept, use this for intercept
	confIntCoefficient = 1
	#Get the values with the least confidence (greatest confdence interval)
	# and compare based on that
	if(lower[4][0] > higher[4][0]):
		if abs(lower[0] - higher[0]) > lower[4][0]:
			return False
		if abs(lower[1] - higher[1]) > confIntCoefficient*lower[4][1]:
			return False
	else:
		if abs(lower[0] - higher[0]) > higher[4][0]:
			return False
		
		if abs(lower[1] - higher[1]) > confIntCoefficient*higher[4][1]:
			return False
	return True
				

'''
Uses the product of linearRegression and clustering to group clusters
based on a similar linear regression. Discards cluster groups that are too small
Returns final dictionary of labels/values
linRegress has format: slope, intercept, score, num_values, [slope confidence interval, intercept confidence interval]
'''
def combineClusters(segmentedValues, linRegress):
	#List of keys we need to remove/pop at the end
	popMe = []

	#First step: Throw out clusers with small numbers of points
	#Or R^2 scores under 80%
	i = 0
	for key in linRegress.keys():
		# Eliminate negative/too shallow slopes, low values, bad regression
		if linRegress[key][0] < 0 or float(linRegress[key][2]) < 0.80 or float(linRegress[key][3]) < 25:
			print("Bad or not enough data in "+str(key))
			segmentedValues[key] = None
			linRegress[key] = None
			popMe.append(key)


	print("Remaining keys are:")
	for key in linRegress.keys():
		if linRegress[key] is not None:
			print(str(key))

	for keyHigher in linRegress.keys():
		#Get a regress and compare it to all lower values
		#If a match is found, new slope/intercept is the weighted average
		#of the two
		for keyLower in linRegress.keys():
			#Because we're popping values left and right, we need to check to make sure the 
			#key is actually still in the dict
			#"keyLower > keyHigher" is confusing. We're referring to the average value of the points
			#Not the literal key value
			if (keyLower > keyHigher) and (linRegress[keyLower] is not None) and (linRegress[keyHigher] is not None):
				lower = linRegress[keyLower]
				higher = linRegress[keyHigher]
				#Check slopes (Can also check intercepts here)
				if isSame(lower, higher):
					#If they're substantially the same, combine and pop missng values
					print("Combining "+str(keyHigher)+" and "+str(keyLower))
					avgWeights = [linRegress[keyHigher][3], linRegress[keyLower][3]]
					linRegress[keyHigher][0] = np.average([higher[0],lower[0]], weights=avgWeights)
					linRegress[keyHigher][1] = np.average([higher[1],lower[1]], weights=avgWeights)
					#Hack: Take the uncertainty of whichever is highest
					linRegress[keyHigher][4] = higher[4] if higher[4][0] < lower[4][0] else lower[4]
					#Don't need to average score -- we're not using it anymore
					linRegress[keyHigher][3] += lower[3]
					segmentedValues[keyHigher].extend(segmentedValues[keyLower])
					linRegress[keyLower] = None
					segmentedValues[keyLower] = None
					popMe.append(keyLower)

	for key in popMe:
		print("Popping "+str(key))
		segmentedValues.pop(key)
		linRegress.pop(key)
	#We can record the linear regressions and do something useful with them here
	compressedX = OrderedDict()
	compressedRegress = OrderedDict()
	i = 0
	for key in segmentedValues.keys():
		compressedX[i] = segmentedValues[key]
		compressedRegress[i] = linRegress[key]
		i += 1

	for key in compressedRegress.keys():
		print("Key: "+str(key)+" Slope: "+str(compressedRegress[key][0])+" Int: "+str(compressedRegress[key][1])+" R2 "+str(compressedRegress[key][2])+" vals: "+str(compressedRegress[key][3])+" conf: "+str(compressedRegress[key][4][0])+", "+str(compressedRegress[key][4][1])+" mean: "+str(compressedRegress[key][5]))

	return compressedX, compressedRegress


'''
Although it often doesn't make sense for us to compute the 
"centroid" of a db scan cluster, given the shape of our
particular clusters, it does
'''
def linearRegression(segmentedValues):
	print("Linear regression")
	#regression = LinearRegression()
	linRegress = dict()
	for key in segmentedValues.keys():
		x = [x[0] for x in segmentedValues[key]]
		y = [x[1] for x in segmentedValues[key]]
		mean = [float(np.average(x)),float(np.average(y))]
		valuesDict = dict()
		valuesDict['x'] = x
		valuesDict['y'] = y
		valuesFrame = pd.DataFrame(valuesDict)
		try:
			rlmRes = sm.rlm(formula = 'y ~ x', data=valuesFrame).fit()
		except ZeroDivisionError:
			#I have no idea why this occurs. A problem with statsmodel
			#Return None
			print("divide by zero :( ")
			return None
		#Caclulate r2_score (unfortunately, rlm does not give this to us)
		x = np.array(x)
		y = np.array(y)
		#Get the predicted values of Y
		y_pred = x*rlmRes.params.x+rlmRes.params.Intercept
		score = r2_score(y, y_pred)
		#These should both be positive -- put in abs anyway
		slopeConfInterval = abs(float(rlmRes.params.x) - float(rlmRes.conf_int(.005)[0].x))
		intConfInterval = abs(float(rlmRes.params.Intercept) - float(rlmRes.conf_int(.005)[0].Intercept))
		#Slope, Intercept, R^2, num of values, confidenceIntervals, mean of cluster
		linRegress[key] = [rlmRes.params.x, rlmRes.params.Intercept, score, len(x), [slopeConfInterval, intConfInterval], mean]
		print("Key: "+str(key)+" Slope: "+str(rlmRes.params.x)+" Intercept: "+str(rlmRes.params.Intercept)+"R2 Score: "+str(score)+" Num vals: "+str(len(x))+" confidence: "+str(slopeConfInterval)+", "+str(intConfInterval)+" mean: "+str(mean))
	return linRegress

'''
Sort labels by average Y axis value of points, 
discard -1 values
'''
def rearrangeLabels(X, labels):
	labelSet = sorted(list(set(labels)))
	if -1 in labelSet:
		labelSet.remove(-1)
	segmentedValues = OrderedDict()
	for label in labelSet:
		segmentedValues[label] = []

	for i in range(len(labels)):
		if labels[i] in labelSet:
			segmentedValues[labels[i]].append(X[i])


	averagesDict = dict()
	for label in labelSet:
		averagesDict[label] = np.mean([x[1] for x in segmentedValues[label]])

	#Sort by highest average Y value -- may tweak this to highest average "X+Y" value or something
	averagesDict = OrderedDict(sorted(averagesDict.items(), key=lambda t: t[0]/2000+t[1]/70, reverse=True))
	
	sortedX = OrderedDict()
	i = 0
	for key in averagesDict.keys():
		sortedX[i] = segmentedValues[key]
		i += 1

	#Values are now sorted. linearRegression does not change sortedX
	linRegress = linearRegression(sortedX)
	if linRegress == None:
		#Mysterious divide by zero error
		return None

	filteredX, filteredRegress = combineClusters(sortedX, linRegress)

	#Create arrays to return
	newLabels = []
	newXVals = []
	i = 0
	for key in filteredX.keys():
	#linRegress[key] = [rlmRes.params.x, rlmRes.params.Intercept, score, len(x), [slopeConfInterval, intConfInterval], mean]
#storeShift(position, slope, intercept, slopeConf, interceptConf, clustMean, clustVals, r2score):
		storeShift(i, filteredRegress[key][0], filteredRegress[key][1], filteredRegress[key][4][0], filteredRegress[key][4][1], filteredRegress[key][5][0], filteredRegress[key][5][1], filteredRegress[key][3], filteredRegress[key][2])
		newXVals.extend(filteredX[key])
		newLabels.extend([i]*len(filteredX[key]))

		i += 1

	return newXVals, newLabels




def dbScanDistance(a, b):
	covInv = np.cov(np.vstack((a,b)).T)
	return mahalanobis(a, b, covInv)

def clusterData(X):
	original=X
	X = StandardScaler().fit_transform(X)
	coreSamples, labels = dbscan(X, min_samples=5, eps=.12, p=4)

	return rearrangeLabels(original, labels)



def threeDPlot(fuelrates, enginespeeds, vehiclespeeds, driver, metaId, truck, labels):

	fig = plt.figure(1)
	ax = fig.add_subplot(111, projection='3d')
	plt.suptitle(driver+", "+str(metaId)+", "+str(truck), fontsize=20)
	#ax.auto_scale_xyz([0, 113], [0, 100], [0, 2200])
	#ax.set_xlim3d([0, 113])
	#ax.set_ylim3d([0,100])
	#ax.set_zlim3d([0,2200])
	#ax.set_autoscale_on(False)
	cm = plt.get_cmap("Dark2")
	ax.scatter(enginespeeds, fuelrates, vehiclespeeds, marker='o', c=labels.tolist(), cmap=cm)
	#Max fuel rate: 113.00
	#Max VS: 155 (let's use 100. 155 is ridiculous)
	#Max engine speed: 2207



	ax.set_xlabel('Engine Speeds')
	ax.set_ylabel('Fuel Rates')
	ax.set_zlabel('Vehicle Speeds')

	plt.show()

def twoDPlot(enginespeeds, vehiclespeeds, driver, truck, stamp, labels):
	if(len(set(labels)) > 20):
		print("Too many labels!")
		return
	#labelSet = sort(set(labels))
	cmap = plt.cm.prism
	cmaplist = [cmap(i) for i in range(cmap.N)]
	cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
	bounds = np.linspace(0, 19, 20)
	norm = mpl.colors.BoundaryNorm(bounds, cmap.N)


	plt.suptitle(driver+", "+str(stamp)+", "+str(truck), fontsize=20)
	plt.scatter(enginespeeds, vehiclespeeds, marker='o', c=labels, cmap=cmap, norm=norm)
	plt.xlabel('Engine Speed', fontsize=18)
	plt.ylabel('Vehicle Speed', fontsize=16)
	plt.colorbar(ticks=bounds)
	plt.show()

def polyRegression(fuelrates, enginespeeds,driver, metaId, truck):
	print("Doing poly regression")
	p = np.polyfit(fuelrates, enginespeeds, 1)
	print(type(p))
	print(p)
	fig = plt.scatter(enginespeeds, fuelrates)
	plt.suptitle(driver+", "+str(metaId)+", "+str(truck), fontsize=20)
	plt.xlabel('Engine Speed', fontsize=18)
	plt.ylabel('Fuel Rate', fontsize=16)
	plt.show()

def knnGraph(X):
	knn_graph = kneighbors_graph(X, 9, include_self=False)

	for connectivity in (None, knn_graph):
	    for n_clusters in (9, 3):
	        plt.figure(figsize=(10, 4))
	        for index, linkage in enumerate(('average', 'complete', 'ward')):
	            plt.subplot(1, 3, index + 1)
	            model = AgglomerativeClustering(linkage=linkage,
	                                            connectivity=connectivity,
	                                            n_clusters=n_clusters)
	            t0 = time.time()
	            model.fit(X)
	            elapsed_time = time.time() - t0
	            plt.scatter(X[:, 0], X[:, 1], c=model.labels_,
	                        cmap=plt.cm.spectral)
	            plt.title('linkage=%s (time %.2fs)' % (linkage, elapsed_time),
	                      fontdict=dict(verticalalignment='top'))
	            plt.axis('equal')
	            plt.axis('off')

	            plt.subplots_adjust(bottom=0, top=.89, wspace=0,
	                                left=0, right=1)
	            plt.suptitle('n_cluster=%i, connectivity=%r' %
	                         (n_clusters, connectivity is not None), size=17)


plt.show()

def main():
	global driver
	global date
	loadDB()
	driverDates = getDriverDates()
	for driverDate in driverDates:
		driver = driverDate['driver']
		date = driverDate['stamp']
		points = getPointsForDriverDate(driverDate['stamp'], driverDate['driver'])
		if points is not None:
			clusteringData = []
			#Summarize and save points
			enginespeeds = []
			#fuelrates = []
			vehiclespeeds = []
			for point in points:
				clusteringData.append([point['enginespeed'],point['vehicleSpeed']])
				###UNCOMMENT THIS FOR 3D PLOTS ###
				enginespeeds.append(point['enginespeed'])
				#fuelrates.append(point['fuelrate'])
				vehiclespeeds.append(point['vehicleSpeed'])
			print("Clustering and labeling data")
			scanResult = clusterData(clusteringData)
			if scanResult is not None:
				print("Results are good")
				xVals, labels = scanResult

				#twoDPlot(enginespeeds, vehiclespeeds, driverDate['driver'], driverDate['truck'],driverDate['stamp'], labels)
				twoDPlot([x[0] for x in xVals], [x[1] for x in xVals], driverDate['driver'], driverDate['truck'],driverDate['stamp'], labels)
				#threeDPlot(fuelrates, enginespeeds, vehiclespeeds, driverDate['driver'], driverDate['stamp'], driverDate['truck'], labels)

	closeDB()


main()
#mainOld()
