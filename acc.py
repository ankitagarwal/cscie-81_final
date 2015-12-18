
from config import *
import pymysql
def loadDB():
    global conn
    global cur
    global idCounter
    idCounter = 0
    conn = pymysql.connect(host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER, passwd=MYSQL_PASSWORD, db='mysql', charset='utf8')

    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute("USE " + MYSQL_DATABASE)

loadDB()
for i in range(5, 1000):
    print(i)

    cur.execute("UPDATE point_summary JOIN"
    "(SELECT metaId, vehicleSpeed as firstSpeed, MAX(vehicleSpeed) as maxSpeed, MIN(vehicleSpeed) as minSpeed, MAX(time) - MIN(time) as timeDiff, TRUNCATE(latitude, 4) as latitude, TRUNCATE(longitude, 4) as longitude FROM points WHERE points.metaID="+ str(i)+" GROUP BY metaId, TRUNCATE(latitude,4), TRUNCATE(longitude,4) ORDER BY time ASC) as b"
    " ON point_summary.metaId = b.metaId AND point_summary.latitude = b.latitude AND point_summary.longitude = b.longitude"
    " SET acceleration = (((maxSpeed-minSpeed)*0.44704)/(timeDiff/1000)) WHERE"
    " firstSpeed = minSpeed AND maxSpeed > 10 AND timeDiff < 10000 AND timeDiff > 1 AND point_summary.metaId=" + str(i))