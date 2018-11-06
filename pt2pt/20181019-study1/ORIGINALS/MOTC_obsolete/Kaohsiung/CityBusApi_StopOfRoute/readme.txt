# data.json obtained 2018-10-31 using
curl -X GET --header 'Accept: application/json' --header 'Authorization: hmac username="FFFFFFFF-FFFF-FFFF-FFFF-FFFFFFFFFFFF", algorithm="hmac-sha1", headers="x-date", signature="6fb3fYdqz5hsX3B9PPDUuimCeQ0="' --header 'x-date: Wed, 31 Oct 2018 09:15:37 GMT' --header 'Accept-Encoding: gzip' --compressed  'https://ptx.transportdata.tw/MOTC/v2/Bus/StopOfRoute/City/Kaohsiung?$format=JSON' > data.json

# The command is from
# https://ptx.transportdata.tw/MOTC#!/CityBusApi/CityBusApi_StopOfRoute

