import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pymysql
from config import *


from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler


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
	

def getAllMetadata():
	global conn
	global cur
	cur.execute("SELECT * FROM routeMetadata")
	return cur.fetchall()


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

def dbScan(X):
	X = StandardScaler().fit_transform(X)
	dbscan = DBSCAN(eps=.2)
	dbscan.fit(X)
	labels = dbscan.fit_predict(X)
	print(labels)
	print("PREDICTED!")



def threeDPlot(fuelrates, enginespeeds, vehiclespeeds, driver, metaId, truck):

	fig = plt.figure(1)
	ax = fig.add_subplot(111, projection='3d')
	plt.suptitle(driver+", "+str(metaId)+", "+str(truck), fontsize=20)
	#ax.auto_scale_xyz([0, 113], [0, 100], [0, 2200])
	#ax.set_xlim3d([0, 113])
	#ax.set_ylim3d([0,100])
	#ax.set_zlim3d([0,2200])
	#ax.set_autoscale_on(False)
	ax.scatter(fuelrates, enginespeeds, vehiclespeeds, c='r', marker='o')
	#Max fuel rate: 113.00
	#Max VS: 155 (let's use 100. 155 is ridiculous)
	#Max engine speed: 2207


	ax.set_xlabel('Fuel Rates')
	ax.set_ylabel('Engine Speeds')
	ax.set_zlabel('Vehicle Speeds')

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

def main():
	global idCounter
	loadDB()
	metadata = getAllMetadata()
	for meta in metadata:
		points = getNextPoints(meta['id'])
		while points is not None:
			if len(points) > 3000:
				clusteringData = []
				#Summarize and save points
				enginespeeds = []
				fuelrates = []
				vehiclespeeds = []
				for point in points:
					clusteringData.append([point['fuelrate'], point['vehicleSpeed']])
					###UNCOMMENT THIS FOR 3D PLOTS ###
					enginespeeds.append(point['enginespeed'])
					fuelrates.append(point['fuelrate'])
					vehiclespeeds.append(point['vehicleSpeed'])
				threeDPlot(enginespeeds, fuelrates,vehiclespeeds, meta['driver'], meta['id'],meta['truck'] )
				#dbScan(clusteringData)
				points = getNextPoints(meta['id'])


	closeDB()

main()
