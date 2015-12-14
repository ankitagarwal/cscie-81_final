from bs4 import BeautifulSoup
from os import listdir, walk
from os.path import isfile, join
import matplotlib.pyplot as plt
import pymysql
from config import *

#Datetime stuff
import pytz
import datetime
from tzwhere import tzwhere

tz = None
conn = None
cur = None

def getFirstPointTS(metaId):
	global cur
	global conn

	cur.execute("SELECT * FROM points WHERE metaId = %s ORDER BY id ASC LIMIT 1", (metaId));
	if cur.rowcount == 0:
		#This timestamp doesn't have any points, apparently
		return None
	return cur.fetchone()['time']


def putPointSummary(metaId, day, hour, dayofweek, latitude, longitude, enginespeed, fuelrate, vehicleSpeed):
	global cur
	global conn

	cur.execute("SELECT * FROM point_summary WHERE day = %s AND hour = %s AND metaId = %s AND latitude = %s AND longitude = %s", (day, hour, metaId, latitude, longitude))
	if cur.rowcount != 0:
		print("Not unique!")
		return
	cur.execute("INSERT INTO point_summary(metaId, day, hour, dayofweek, latitude, longitude, enginespeed, fuelrate, vehicleSpeed) VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s)", (metaId, day, hour, dayofweek, latitude, longitude, enginespeed, fuelrate, vehicleSpeed))
	conn.commit()


def getNextPoints(metaId, lowerTime, higherTime):
	global cur
	global conn
	points = []

	#Make sure we don't already have these
	cur.execute("SELECT metaId, TRUNCATE(latitude,4) as latitude, TRUNCATE(longitude, 4) as longitude, AVG(enginespeed) AS enginespeed, AVG(fuelrate) AS fuelrate, AVG(vehicleSpeed) as vehicleSpeed FROM points WHERE metaId = %s AND time >= %s AND time < %s GROUP BY TRUNCATE(latitude,4), TRUNCATE(longitude,4)", (metaId, lowerTime, higherTime))
	if cur.rowcount == 0:
		return None
	return cur.fetchall()
	

def getAllMetadata():
	global conn
	global cur
	cur.execute("SELECT * FROM routeMetadata where id > 2281 ORDER BY id ASC")
	return cur.fetchall()

#Epoch timestamp is in *seconds* not ms
def getLocalTimes(epochTimestamp, latitude, longitude):
	utc_date = datetime.datetime.fromtimestamp(epochTimestamp, tz=pytz.utc)
	tzName = tz.tzNameAt(latitude, longitude);
	localtz = pytz.timezone(tzName)
	local_date = utc_date.astimezone(localtz)
	weekday = local_date.weekday()
	hour = local_date.hour
	isoDate = local_date.isoformat()
	return weekday, hour, isoDate


def loadDB():
	global conn
	global cur
	global tz
	#1384210800000
	#Just add this in here
	tz = tzwhere.tzwhere(shapely=True, forceTZ=True)
	conn = pymysql.connect(host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER, passwd=MYSQL_PASSWORD, db='mysql', charset='utf8')

	cur = conn.cursor(pymysql.cursors.DictCursor)
	cur.execute("USE trucking")

def closeDB():
	conn.close()

def summarize(points, timestamp):
	metaId = points[0]['metaId']
	dayofweek, hour, isoDate = getLocalTimes(timestamp/1000, points[0]['latitude'], points[0]['longitude'])
	for point in points:
		putPointSummary(metaId, isoDate, hour, dayofweek, point['latitude'], point['longitude'], point['enginespeed'], point['fuelrate'], point['vehicleSpeed'])


#takes a metaId and returns the first timestamp, along with the 
#second timestamp, indicating the end of the hour
def getFirstTimestamp(metaId):
	start = getFirstPointTS(metaId)
	if start is None:
		return None
	end = start + (3600000 - start % 3600000)
	return start, end

loadDB()
metadata = getAllMetadata()
for meta in metadata:
	print("Summarizing "+str(meta['id']))
	times = getFirstTimestamp(meta['id'])
	if times is not None:
		start = times[0]
		end = times[1]
		points = getNextPoints(meta['id'], start, end)
		while points is not None:
			#Summarize and save points
			summarize(points, int((start+end)/2))
			start = end
			end = start + 3600000
			points = getNextPoints(meta['id'], start, end)


closeDB()
