
# RA, 2018-10-29

# scp ~/repos/numpde/transport/pt2pt/*BusCal/buscal.py 192.168.100.201:~/BusCal-AS/

# pip3 install --upgrade google-api-python-client
from google.oauth2 import service_account

# For Scopes, see
# https://developers.google.com/calendar/auth
# https://stackoverflow.com/a/49225582
SCOPES = ['https://www.googleapis.com/auth/calendar']
SERVICE_ACCOUNT_FILE = './.credentials/UV/buscal-1540742972387-16e078047967.json'

credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)

import datetime as dt

from googleapiclient.discovery import build
service = build('calendar', 'v3', credentials=credentials)

# "BusCal -- AS" calendar ID
# https://developers.google.com/calendar/v3/reference/calendarList/list
calId = 'r1i54vcjb8g6dj54vrrjpk7fbs@group.calendar.google.com'

calendar = service.calendars().get(calendarId=calId).execute()

print(calendar['summary'])

def insert_arrivals(calId) :

	import urllib.request
	import urllib.parse
	import re

	import pytz
	TZ = pytz.timezone('Asia/Taipei')

	# Step 1: Determine stop

	stopname = "Academia Sinica"

	# Step 2: Get routes through stop

	url = "http://www.5284.com.tw/Aspx/dybus/arrivalInfo.aspx" + "?" + urllib.parse.urlencode({
		'lang' : 'eng',
		'ACTION' : '24',
		'PA_NUMB' : stopname,
	})

	with urllib.request.urlopen(url) as response :
		r = response.read().decode('utf-8')

	# |205,10181,3|212,10912,4|212Express,10911,4|212Night,16132,4|270,11841,3|270Shuttle,11842,3|276,11851,3|306,10473,3|306Shuttle,11853,3|620,15334,3|645,10461,3|645Sub,10462,3|679,16127,3|679Shuttle,17678,3|BL25,10941,1|S1,11051,2|S12,11052,2|S12Shuttle,11053,2|S5,11059,2|S5Shuttle,11061,2_@

	routes = re.findall('\|(\w+)', r)

	# Step 3 : get expected arrival times

	# routes = ['306']

	# Calendar events so far
	events = (
		service.events().list(calendarId=calId, singleEvents=True).execute()
	).get('items', [])

	for route in routes :

		for goback in [0, 1] :

			print("---------")

			print("Route {}-{}".format(route, goback))

			url = "http://www.5284.com.tw/Aspx/dybus/dybusRoute.aspx?" + urllib.parse.urlencode({
				'lang' : 'eng',
				'Route' : route,
				'goback' : goback,
			})

			with urllib.request.urlopen(url) as response:
				r = response.read().decode('utf-8')

			print("Response:", r)
			# 853-FU,0,1,0|260-U3,1,21,1|233-FU,1,30,0|@_|Luzhou Bus Terminal_,0_,3368_,1_|Wangye Temple Entrance_,1_,3478_,1_|National Open Univ._,2_,24_,0_|Zhongyuan Apartment_,3_,47_,0_|Luzhou Elementary School_,4_,101_,0_|Luzhou Motor Vehicles Office_,5_,166_,0_|Luzhou Police Substation_,6_,204_,0_|Xiqian_,7_,292_,0_|MRT St. Ignatius High School_,8_,342_,0_|St. Ignatius High School_,9_,396_,0_|Xingfu Theater_,10_,438_,0_|Jianhe New Village_,11_,472_,0_|MRT Sanhe Juior High School Sta._,12_,503_,0_|Sanhe Junior High School_,13_,559_,0_|Gezhi High School_,14_,599_,0_|Houde Police Substation_,15_,644_,0_|Delin Temple_,16_,678_,0_|Longmen Rd. Entrance_,17_,753_,0_|Sanan Li_,18_,792_,0_|Changshou W. St. Entrance_,19_,829_,0_|Sanchong Bus Co. Ltd._,20_,868_,0_|Zhengyi Chongxin Intersection_,21_,901_,0_|Sanchong Post Office_,22_,969_,0_|Guangxing Elementary School_,23_,996_,0_|Zhengyi S. Rd. (End)_,24_,1020_,0_|Tianhou Temple_,25_,1078_,0_|Fude S. Rd._,26_,1163_,0_|Liangzhou and Chongqing Intersection_,27_,1403_,0_|Minsheng and Chongqing Intersection_,28_,1480_,0_|Zhaoyang Park_,29_,1521_,0_|Taipei Circle(Nanjing)_,30_,1573_,0_|MRT Zhongshan Sta.(Zhiren High School)_,31_,71_,0_|Nanjing Linsen Intersection_,32_,184_,0_|Nanjing Jilin Intersection_,33_,269_,0_|MRT Songjiang Nanjing Stop_,34_,330_,0_|Nanjing Jianguo Intersection_,35_,397_,0_|Nanjing Longjiang Intersection_,36_,476_,0_|MRT Nanjing Fuxing Station_,37_,540_,0_|Nanjing Dunhua Intersection(Arena)_,38_,620_,0_|Nanjing Ningan Roads_,39_,754_,0_|Nanjing Sanmin Intersection_,40_,852_,0_|Nanjing Apartment(MRT Nanjing Sanmin)_,41_,962_,0_|Nansongshan(Nanjing)_,42_,996_,0_|Raohe St. Night Market (Tayou)_,43_,1044_,0_|Songshan Farmers Association_,44_,1157_,0_|Songshan Station(Bade)_,45_,1211_,0_|Yucheng Li_,46_,1280_,0_|Songshan Brick Factory_,47_,1373_,0_|Nangang Rd. Sec. 3_,48_,1401_,0_|Mercedes Benz Taiwan_,49_,1430_,0_|Yucheng Elementary School_,50_,1457_,0_|Tudigong Temple_,51_,1540_,0_|Dongming Li_,52_,1578_,0_|Nangang Tire_,53_,1599_,0_|Taifei New Village_,54_,1622_,0_|Nangang Dist. Admin. Center_,55_,1673_,0_|Nangang Vocational High School_,56_,1723_,0_|Nangang_,57_,1778_,0_|TWTC Nangang Exhibition Hall_,58_,1823_,0_|Chengzheng Junior High School_,59_,1910_,0_|Nangang Water Plant_,60_,1936_,0_|Yuangong Bridge_,61_,1979_,0_|Zhongyan New Village_,62_,2018_,0_|Academia Sinica_,63_,2054_,0_|Hushih Park_,64_,2106_,0_|Jiuru Li 1_,65_,2128_,0_|Lingyun Village 5_,66_,2160_,0_|Jiuru Li 2_,67_,2180_,0_|Academia Rd. Sec. 3_,68_,2201_,0_|Lingyun Village 5 Stop_,69_,2287_,0_|Jiuzhuang 1st. Stop_,70_,7186_,1_|Jiuzhuang Elementary School_,71_,7218_,1_|Jiuzhuang Bus Terminal_,72_,7265_,1_|Jiuzhang Stop_,73_,7317_,1@_|233-FU_,306_,121.515927_,25.053747_,0_,108_,1_,306_,2018/10/29 上午 05:09:53_,222233955_|853-FU_,306_,121.466492_,25.088418_,0_,145_,0_,306_,2018/10/29 上午 05:09:46_,222237691

			now = dt.datetime.now(tz=TZ)
			#
			def eta_to_time(eta) :
				eta = int(eta)
				if (eta < 0) : return '?'
				eta_dt = now + dt.timedelta(seconds=eta)
				return { 'datetime' : eta_dt, 'pretty' : '{0:%H:%M}'.format(eta_dt) }

			try :
				(busses, stops, location) = r.split('@')
				print(" o) Busses:", busses)
				print(" o) Stops:", stops)
				print(" o) Location:", location)

				stops = re.findall('\|([^_]+)_,(\d+)_,([^_]+)_,([^_@]+)', stops)

				if not stops : continue

				# ETA in seconds
				for k in range(len(stops)) :
					stops[k] = list(stops[k])
					stops[k][2] = int(stops[k][2])

				print("Parse stops:", stops)

				dest = stops[-1][0]

				# Extract the info about our stop of interest
				stop_index = [s[0] for s in stops].index(stopname)

				stop_info = stops[stop_index]

				next_stops = stops[stop_index:]

			except ValueError :
				# Stop not in list
				continue

			# Meaning unclear
			if (int(stop_info[3]) < 0) : continue

			eta = stop_info[2] # in seconds

			# This probably means "bus at depot":
			if (eta < 0) : continue

			(ta, ta_pretty) = (eta_to_time(eta)['datetime'], eta_to_time(eta)['pretty'])

			tb = ta + dt.timedelta(minutes=1)

			summary = '{} ~~> {}, ..., {}'.format(route, next_stops[1][0], next_stops[-1][0])

			print(summary, 'ETA {}'.format(ta_pretty))

			#
			for event in events:
				if (event['summary'] == summary) :
					try :
						service.events().delete(calendarId=calId, eventId=event['id']).execute()
					except Exception :
						pass

			for (i, (a, b)) in enumerate(zip(next_stops[:-1], next_stops[1:])) :
				if (a[2] > b[2]) :
					gap = (a[2] - b[2]) + 60 # seconds
					for j in range(i+1, len(next_stops)) :
						next_stops[j][2] += gap

			desc = "\n ".join(
				[ "ETA:" ] + [
					"[{}] {}".format(eta_to_time(s[2])['pretty'], s[0]) for s in next_stops
				]
			)

			event = {
				'summary': summary,
				'location': stopname,
				'description': desc,
				'start': {
					'dateTime': ta.isoformat(),
					'timeZone': TZ.zone,
				},
				'end': {
					'dateTime': tb.isoformat(),
					'timeZone': TZ.zone,
				},
				'reminders': {
					'useDefault': False,
					'overrides': [
						{'method': 'popup', 'minutes': 10},
					],
				},
			}

			event = service.events().insert(calendarId=calId, body=event).execute()

	# http://www.5284.com.tw/Aspx/dybus/busLine.aspx
	# ACTION=44
	# count=Y
	# Glid=205
	# goback=0
	# lang=eng
	# ranv=1002904011039
	#
	# Response: _|205_,China University of Science and Tec._,Dongyuan_,3_,0_|........BUS STOP COORDINATES

	# http://www.5284.com.tw/Aspx/dybus/busLine2.aspx
	# ACTION=44
	# Glid=205
	# goback=0
	# lang=eng
	# rane=1002904011039
	#
	# Response: _|205_,China University of Science and Tec._,Dongyuan_,3_,0_&_|China unzipped. of Science Technology(Main Entrance)_,0_,-1_,2_|China unzipped. of Science Technology_,1_,-1_,2_|Soldiers Public Cemetry_,2_,-1_,2_|Academia Rd. Sec. 3_,3_,-1_,2_|Jiuru Li 2_,4_,-1_,2_|Lingyun Village 5_,5_,-1_,2_|Jiuru Li 1_,6_,-1_,2_|Hushih Park_,7_,-1_,2_|Academia Sinica_,8_,-1_,2_|Zhongyan New Village_,9_,-1_,2_| ..........


def delete_past_events(calId, margin=-1) :

	# Delete all past events, plus margin minutes

	# Call the Calendar API
	now = (dt.datetime.utcnow() + dt.timedelta(minutes=margin)).isoformat() + 'Z'  # 'Z' indicates UTC time

	events = (
		service.events().list(calendarId=calId, timeMax=now, singleEvents=True).execute()
	).get('items', [])

	for event in events :
		service.events().delete(calendarId=calId, eventId=event['id']).execute()


delete_past_events(calId)
insert_arrivals(calId)
