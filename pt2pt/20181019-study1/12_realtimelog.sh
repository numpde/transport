#!/usr/bin/env bash

# Script for downloading realtime bus information
# RA, 2018-10-31

# City
C="Taipei"

# MOTC data category
Y="Bus/RealTimeByFrequency/City"

# Root output directory
OUT="OUTPUT/12/${C}/UV"

while true; do

	# Output directory: date stamp
	d=${OUT}/$(date +"%Y%m%d")

	# Create output directory, if needed
	mkdir -p ${d}

    # Output file: time stamp
	f=${d}/$(date +"%H%M%S").json
	echo "Downloading to file $f"

	# Authorization mechanism
	MSG="x-date: "$(LANG=US date -u +"%a, %d %b %Y %H:%M:%S GMT")
	KEY="FFFFFFFF-FFFF-FFFF-FFFF-FFFFFFFFFFFF"
	SIG=$(echo -n $MSG | openssl dgst -binary -sha1 -hmac $KEY | base64)

	#echo $MSG/$SIG

	echo ----

	sleep 1

	# Command obtained from
	# https://ptx.transportdata.tw/MOTC#!/CityBusApi/CityBusApi_RealTimeByFrequency
	curl -X GET \
		--header 'Accept: application/json' \
		--header 'Authorization: hmac username="'"${KEY}"'", algorithm="hmac-sha1", headers="x-date", signature="'"${SIG}"'"' \
		--header "${MSG}" \
		--header 'Accept-Encoding: gzip' \
		--compressed 'https://ptx.transportdata.tw/MOTC/v2/'"${Y}"'/'"${C}"'?$format=JSON' \
		> $f \
		&

	# Pause for a total of 30 seconds

	sleep 19

	echo ====

	sleep 10

done



exit

##### END OF SCRIPT #####



# The following was used to make this download script

# From
# https://ptx.transportdata.tw/MOTC#!/CityBusApi/CityBusApi_RealTimeByFrequency

# Example generated headers
# curl -items GET --header 'Accept: application/json' --header 'Authorization: hmac username="FFFFFFFF-FFFF-FFFF-FFFF-FFFFFFFFFFFF", algorithm="hmac-sha1", headers="x-date", signature="nUZfcRh631qkDEVdMCeMFXvbV8I="' --header 'x-date: Wed, 31 Oct 2018 10:22:54 GMT' --header 'Accept-Encoding: gzip' --compressed  'https://ptx.transportdata.tw/MOTC/v2/Bus/RealTimeByFrequency/City/Kaohsiung?$top=3&$format=JSON'

# Excerpt from the webpage source
function l() {
  var n = $("#AppID")[0].value,
    t = $("#AppKey")[0].value,
    r, i;
  (t === null || t.trim() === "") && (n === null || n.trim() === "") && (t = "FFFFFFFF-FFFF-FFFF-FFFF-FFFFFFFFFFFF", n = "FFFFFFFF-FFFF-FFFF-FFFF-FFFFFFFFFFFF");
  r = (new Date).toGMTString();
  i = new jsSHA("SHA-1", "TEXT");
  i.setHMACKey(t, "TEXT");
  i.update("x-date: " + r);
  var u = i.getHMAC("B64"),
    f = 'hmac username="' + n + '", algorithm="hmac-sha1", headers="x-date", signature="' + u + '"',
    e = new SwaggerClient.ApiKeyAuthorization("Authorization", f, "header"),
    o = new SwaggerClient.ApiKeyAuthorization("x-date", r, "header");
  window.swaggerUi.api.clientAuthorizations.add("Authorization", e);
  window.swaggerUi.api.clientAuthorizations.add("x-date", o)
}


# JSON format of the response:

BusA1Data {

    PlateNumb (string):
    車牌號碼 ,

    OperatorID (string, optional):
    營運業者代碼 ,

    RouteUID (string, optional):
    路線唯一識別代碼，規則為 {業管機關簡碼} + {RouteID}，其中 {業管機關簡碼} 可於Authority API中的AuthorityCode欄位查詢 ,

    RouteID (string, optional):
    地區既用中之路線代碼(為原資料內碼) ,

    RouteName (NameType, optional):
    路線名稱 ,

    SubRouteUID (string, optional):
    子路線唯一識別代碼，規則為 {業管機關簡碼} + {SubRouteID}，其中 {業管機關簡碼} 可於Authority API中的AuthorityCode欄位查詢 ,

    SubRouteID (string, optional):
    地區既用中之子路線代碼(為原資料內碼) ,

    SubRouteName (NameType, optional):
    子路線名稱 ,

    Direction (string, optional):
    去返程 = ['0: 去程', '1: 返程'],

    BusPosition (PointType, optional):
    車輛位置經度 ,

    Speed (number, optional):
    行駛速度(kph) ,

    Azimuth (number, optional):
    方位角 ,

    DutyStatus (string, optional):
    勤務狀態 = ['0: 正常', '1: 開始', '2: 結束'],

    BusStatus (string, optional):
    行車狀況 = ['0: 正常', '1: 車禍', '2: 故障', '3: 塞車', '4: 緊急求援', '5: 加油', '90: 不明', '91: 去回不明', '98: 偏移路線', '99: 非營運狀態', '100: 客滿', '101: 包車出租', '255: 未知'],
    Driving status = ['0: normal', '1: car accident', '2: fault', '3: traffic jam', '4: emergency help', '5: refueling', '90: unknown', '91: go back to unknown ', '98: offset route', '99: non-operational status', '100: full passenger', '101: charter rental', '255: unknown'],

    MessageType (string, optional):
    資料型態種類 = ['0: 未知', '1: 定期', '2: 非定期'],
    Type of data type = ['0: Unknown', '1: Regular', '2: Non-periodic'],

    GPSTime (DateTime):
    車機時間(ISO8601格式:yyyy-MM-ddTHH:mm:sszzz) ,

    TransTime (DateTime, optional):
    車機資料傳輸時間(ISO8601格式:yyyy-MM-ddTHH:mm:sszzz)[多數單位沒有提供此欄位資訊] ,

    SrcRecTime (DateTime, optional):
    來源端平台接收時間(ISO8601格式:yyyy-MM-ddTHH:mm:sszzz) ,

    SrcTransTime (DateTime, optional):
    來源端平台資料傳出時間(ISO8601格式:yyyy-MM-ddTHH:mm:sszzz)[公總使用TCP動態即時推播故有提供此欄位, 而非公總系統因使用整包資料更新, 故沒有提供此欄位] ,

    SrcUpdateTime (DateTime, optional):
    來源端平台資料更新時間(ISO8601格式:yyyy-MM-ddTHH:mm:sszzz)[公總使用TCP動態即時推播故沒有提供此欄位, 而非公總系統因提供整包資料更新, 故有提供此欄] ,

    UpdateTime (DateTime):
    本平台資料更新時間(ISO8601格式:yyyy-MM-ddTHH:mm:sszzz)

}

NameType {

    Zh_tw (string, optional):
    中文繁體名稱 ,

    En (string, optional):
    英文名稱

}

PointType {

    PositionLat (number, optional):
    位置緯度(WGS84) ,

    PositionLon (number, optional):
    位置經度(WGS84)

}
