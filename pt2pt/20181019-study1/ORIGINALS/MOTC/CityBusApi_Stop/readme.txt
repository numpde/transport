# data.json obtained 2018-10-31 with
curl -X GET --header 'Accept: application/json' --header 'Authorization: hmac username="FFFFFFFF-FFFF-FFFF-FFFF-FFFFFFFFFFFF", algorithm="hmac-sha1", headers="x-date", signature="3FphaaydPI5a8N5R6jC4xVlW4k8="' --header 'x-date: Wed, 31 Oct 2018 09:12:28 GMT' --header 'Accept-Encoding: gzip' --compressed  'https://ptx.transportdata.tw/MOTC/v2/Bus/Stop/City/Kaohsiung?$format=JSON' > data.json

# Command from
# https://ptx.transportdata.tw/MOTC#!/CityBusApi/CityBusApi_Stop

