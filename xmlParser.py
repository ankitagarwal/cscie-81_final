from bs4 import BeautifulSoup
from os import listdir, walk
from os.path import isfile, join
import matplotlib.pyplot as plt
import pymysql
from config import *


conn = None
cur = None
basePath = "data/"
currentPath = "data/"
dirList = None
visitedList =[]

def insertPoints(data):
	global cur
	global conn
	#Make sure we don't already have these
	cur.execute("SELECT * FROM points WHERE metaId = %s AND time = %s", (int(data[0][0]), int(data[0][2])))
	if cur.rowcount == 0:
		cur.executemany("INSERT INTO points (metaId, time, latitude, longitude, enginespeed, fuelrate, vehicleSpeed) VALUES (%s, %s, %s, %s, %s, %s, %s)",data)
		conn.commit()

	else:
		print("This data is already inserted!")



def getMetaDataId(driver, truck, stamp):
	global conn
	global cur
	cur.execute("SELECT * FROM routeMetadata WHERE driver = %s AND stamp = %s", (driver, int(stamp)))
	if cur.rowcount == 0:
		cur.execute("INSERT INTO routeMetadata(driver, truck, stamp) VALUES (%s, %s, %s)", (driver, truck, stamp))
		try:
			conn.commit()
			metaId = conn.insert_id()
			return metaId
		except:
			conn.rollback()

	else:
		return cur.fetchone()['id']


def addToDB(pathPrefix, driver, truck, stamp):
	i = 0
	path = pathPrefix+"/"+str(stamp)+"-"+str(i).zfill(4)+".xml"
	metaId = None
	#fuelRates = []
	#engineSpeeds = []
	#latitudes = []
	#longitudes = []
	#times = []
	data = []
	while isfile(path):
		f = open(path, 'r')
		xml = f.read()
		bsObj = BeautifulSoup(xml)
		daes = bsObj.findAll("dae")
		if len(daes) != 0 and metaId is None:
				metaId = getMetaDataId(driver, truck, stamp)
		for dae in daes:
			fuelRate = dae.find("fr")
			engineSpeed = dae.find("es")
			latitude = dae.find("lat")
			longitude = dae.find("lon")
			vehicleSpeed = dae.find("vs")
			time = dae.find("vne")

			if(fuelRate != None and engineSpeed != None and latitude != None and longitude != None and time.attrs['time'] != None and vehicleSpeed != None):
				fuelRate = float(fuelRate.get_text())
				engineSpeed = int(float(engineSpeed.get_text()))
				latitude = float(latitude.get_text())
				longitude = float(longitude.get_text())
				vehicleSpeed = float(vehicleSpeed.get_text())
				time = int(time.attrs['time'])

				if(vehicleSpeed > 0.3 and fuelRate > .01):
					#fuelRates.append(fuelRate)
					#engineSpeeds.append(engineSpeed)
					#latitudes.append(latitude)
					#longitudes.append(longitude)
					#times.append(time)
					data.append((metaId, time, latitude, longitude, engineSpeed, fuelRate, vehicleSpeed))
		i += 1
		path = pathPrefix+"/"+str(stamp)+"-"+str(i).zfill(4)+".xml"
	if(len(data) > 0):
		insertPoints(data)
	else:
		print("No data for leg!")


	#plt.scatter(fuelRates, engineSpeeds)
	#plt.show()

def loadDB():
	global conn
	global cur
	conn = pymysql.connect(host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER, passwd=MYSQL_PASSWORD, db='mysql', charset='utf8')

	cur = conn.cursor(pymysql.cursors.DictCursor)
	cur.execute("USE trucking")
def closeDB():
	conn.close()

def writeVisited(visited):
	with open("visited.txt", "a") as f:
		f.write(visited+"\n")

def getVisited():
	global visitedList
	with open("visited.txt") as f:
		visitedList = f.read().splitlines()

def buildDirList():
	global dirList
	global basePath
	global visitedList

	print(visitedList)
	print("Building directory list...")
	dirList = [x[0] for x in walk(basePath) if x not in visitedList]
	for directory in dirList:
		if ("heartbeats" in directory):
			dirList.remove(directory)
		if directory in visitedList:
			dirList.remove(directory)



'''
Returns driverName, timestamp, truckVin, from path+file-header.xml 
'''
def getMetaInfo(filePath):
	truck = None
	badVins = ["00000000000000000", "Unknown", ""]
	with open(filePath) as f:
		headerTxt = f.read()
		bsObj = BeautifulSoup(headerTxt)
		vinTag = bsObj.find("vin")
		if vinTag is not None:
			vin = vinTag.get_text()
			if vin not in badVins:
				print("Found vin tag! "+vin)
				truck = vin

	#data/pride/20150630/CliffordHuyck/20131111072506-header.xml
	fileParts = filePath.split("/")
	driverName = fileParts[3]
	if "-" in driverName:
		nameParts = driverName.split("-")
		driverName = nameParts[1]
	stamp = fileParts[4]
	stamp = stamp.replace("-header.xml", "")
	return driverName, stamp, truck

def findNextDataSets():
	global dirList
	global visitedList
	print("Getting list of xml files...")

	fileList = []
	newDir = None
	while len(fileList) == 0:
		newDir = dirList.pop(0)
		if newDir not in visitedList:
			print("Trying directory: "+newDir)
			fileList = [f for f in listdir(newDir) if (isfile(join(newDir, f)) and f.endswith("-header.xml"))]
		else:
			print("Already visited!")
	writeVisited(newDir)
	print(fileList)
	for header in fileList:
		driverName, stamp, truck = getMetaInfo(newDir+"/"+header)
		print("Driver name: "+driverName+", stamp: "+stamp+" truck: "+str(truck))
		addToDB(newDir, driverName, truck, stamp)



loadDB()
getVisited()
buildDirList()
for i in range(50):
	findNextDataSets()

#pathPrefix = "data/DON-HUMMER/20151130/CandaceMarley"
#driver = "CandaceMarley"
#truck = ""
#stamp = "20151130045436"
#addToDB(pathPrefix, driver, truck, stamp)
closeDB()
