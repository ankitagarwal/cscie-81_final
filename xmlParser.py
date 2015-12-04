from bs4 import BeautifulSoup
import os
import matplotlib.pyplot as plt
import pymysql
from config import *


conn = None
cur = None

def insertPoints(data):
     global cur
     global conn
     #Make sure we don't already have these
     cur.execute("SELECT * FROM points WHERE metaId = %s AND time = %s", (int(data[0][0]), int(data[0][2])))
     print("SELECT * FROM points WHERE metaId = "+str(data[0][0])+" AND time = "+str(data[0][2]))
     if cur.rowcount == 0:
          print("Rowcount is ZERO!")

          cur.executemany("INSERT INTO points (metaId, time, latitude, longitude, enginespeed, fuelrate) VALUES (%s, %s, %s, %s, %s, %s)",data)
          print("Executing")
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
     metaId = getMetaDataId(driver, truck, stamp)

     i = 0
     path = pathPrefix+"/"+str(stamp)+"-"+str(i).zfill(4)+".xml"
     print("Path is: "+path)
     fuelRates = []
     engineSpeeds = []
     #latitudes = []
     #longitudes = []
     #times = []
     data = []
     while os.path.isfile(path):
          print("Path is: "+path)
          f = open(path, 'r')
          xml = f.read()
          bsObj = BeautifulSoup(xml)
          daes = bsObj.findAll("dae")
          for dae in daes:
               fuelRate = dae.find("fr")
               engineSpeed = dae.find("es")
               latitude = dae.find("lat")
               longitude = dae.find("lon")
               time = dae.find("vne")
               if(fuelRate != None and engineSpeed != None and latitude != None and longitude != None and time.attrs['time'] != None):
                    fuelRate = float(fuelRate.get_text())
                    engineSpeed = int(float(engineSpeed.get_text()))
                    latitude = float(latitude.get_text())
                    longitude = float(longitude.get_text())
                    time = int(time.attrs['time'])

                    if(engineSpeed > 200 and fuelRate > .01):
                         fuelRates.append(fuelRate)
                         engineSpeeds.append(engineSpeed)
                         #latitudes.append(latitude)
                         #longitudes.append(longitude)
                         #times.append(time)
                         data.append((metaId, time, latitude, longitude, engineSpeed, fuelRate))
          i += 1
          path = pathPrefix+"/"+str(stamp)+"-"+str(i).zfill(4)+".xml"
     print(data[0])
     insertPoints(data)


     plt.scatter(fuelRates, engineSpeeds)
     plt.show()

def loadDB():
     global conn
     global cur
     conn = pymysql.connect(host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER, passwd=MYSQL_PASSWORD, db='mysql', charset='utf8')

     cur = conn.cursor(pymysql.cursors.DictCursor)
     cur.execute("USE trucking")
def closeDB():
     conn.close()

loadDB()
pathPrefix = "data/DON-HUMMER/20151130/CandaceMarley"
driver = "CandaceMarley"
truck = ""
stamp = "20151130045436"
addToDB(pathPrefix, driver, truck, stamp)
closeDB()
