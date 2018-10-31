#!/usr/bin/bash

# Script for downloading realtime bus information in Kaohsiung
# RA, 2018-10-31

CITY=Kaohsiung

# Output directory
OUT="OUTPUT/12/${CITY}/UV"
# Create it, if needed
mkdir -p ${OUT}

while true; do

	f=${OUT}/$(date +"%Y%m%d-%H%M%S").json
	echo "Downloading to $f"

	# Authorization mechanism
	MSG="x-date: "$(LANG=US date -u +"%a, %d %b %Y %H:%M:%S GMT")
	KEY="FFFFFFFF-FFFF-FFFF-FFFF-FFFFFFFFFFFF"
	SIG=$(echo -n $MSG | openssl dgst -binary -sha1 -hmac $KEY | base64)

	#echo $MSG/$SIG

	sleep 1

	# Command obtained from
	# https://ptx.transportdata.tw/MOTC#!/CityBusApi/CityBusApi_RealTimeByFrequency
	curl -X GET \
		--header 'Accept: application/json' \
		--header 'Authorization: hmac username="'"${KEY}"'", algorithm="hmac-sha1", headers="x-date", signature="'"${SIG}"'"' \
		--header "${MSG}" \
		--header 'Accept-Encoding: gzip' \
		--compressed 'https://ptx.transportdata.tw/MOTC/v2/Bus/RealTimeByFrequency/City/'"${CITY}"'?$format=JSON' \
		> $f

	echo ----
	sleep 60

done




exit


# The following was used to make this download script

# From
# https://ptx.transportdata.tw/MOTC#!/CityBusApi/CityBusApi_RealTimeByFrequency

# Example generated headers
# curl -X GET --header 'Accept: application/json' --header 'Authorization: hmac username="FFFFFFFF-FFFF-FFFF-FFFF-FFFFFFFFFFFF", algorithm="hmac-sha1", headers="x-date", signature="nUZfcRh631qkDEVdMCeMFXvbV8I="' --header 'x-date: Wed, 31 Oct 2018 10:22:54 GMT' --header 'Accept-Encoding: gzip' --compressed  'https://ptx.transportdata.tw/MOTC/v2/Bus/RealTimeByFrequency/City/Kaohsiung?$top=3&$format=JSON'

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
