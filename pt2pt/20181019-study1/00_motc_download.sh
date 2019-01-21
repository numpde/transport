#!/usr/bin/env bash

# Download bus route info from https://ptx.transportdata.tw
# RA, 2018-11-07

# Where to put the files
BASE_PATH="OUTPUT/00/ORIGINAL_MOTC"

# Cities of interest
declare -a CITIES=("Kaohsiung" "Taipei")

# Do we have those?
which curl    1> /dev/null || (echo curl not found    ; exit)
which base64  1> /dev/null || (echo base64 not found  ; exit)
which openssl 1> /dev/null || (echo openssl not found ; exit)

download() {

	# INPUT:

	# City
	C=$1

	# Type of information (Route, Shape, etc...)   
	X=$2

	# WORK:

	# Output directory for the response
	D="${BASE_PATH}/${C}"

	# Create output directory, if needed
	mkdir -p ${D}

	# Filename for output
	f="${D}/CityBusApi_${X}.json"
	echo "Downloading $X to file $f"

	# Authorization mechanism
	MSG="x-date: "$(LANG=US date -u +"%a, %d %b %Y %H:%M:%S GMT")
	KEY="FFFFFFFF-FFFF-FFFF-FFFF-FFFFFFFFFFFF"
	SIG=$(echo -n $MSG | openssl dgst -binary -sha1 -hmac $KEY | base64)

	#echo $MSG/$SIG

	# Command obtained from
	# https://ptx.transportdata.tw/MOTC#!/CityBusApi/CityBusApi_RealTimeByFrequency
	curl -X GET \
		--header 'Accept: application/json' \
		--header 'Authorization: hmac username="'"${KEY}"'", algorithm="hmac-sha1", headers="x-date", signature="'"${SIG}"'"' \
		--header "${MSG}" \
		--header 'Accept-Encoding: gzip' \
		--compressed 'https://ptx.transportdata.tw/MOTC/v2/Bus/'"${X}"'/City/'"${C}"'?$format=JSON' \
		> $f

	echo
}

# Loop through cities and items to download
for C in "${CITIES[@]}" ; do
	for X in Route Shape Station Stop StopOfRoute DisplayStopOfRoute Schedule Vehicle RouteFare Operator ; do
		download $C $X
		sleep 1
	done
done


