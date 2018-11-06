#!/usr/bin/bash

# Download bus route info from https://ptx.transportdata.tw
# RA, 2018-11-07

which curl 1> /dev/null || (echo curl not found; exit)

CITY=Kaohsiung

# Root output directory
OUT="OUTPUT/00/ORIGINAL_MOTC/${CITY}"

download() {
    X=$1

    # Output directory for the response
	D="${OUT}/CityBusApi_${X}"

	# Create output directory, if needed
	mkdir -p ${D}

	f="${D}/data.json"
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
		--compressed 'https://ptx.transportdata.tw/MOTC/v2/Bus/'"${X}"'/City/'"${CITY}"'?$format=JSON' \
		> $f

    echo
}

for X in Route Shape Station Stop StopOfRoute; do
    download $X
	sleep 1
done

