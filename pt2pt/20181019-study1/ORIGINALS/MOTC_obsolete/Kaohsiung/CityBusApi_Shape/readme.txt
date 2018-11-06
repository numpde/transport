# data.json obtained 2018-11-02 with
curl -X GET --header 'Accept: application/json' --header 'Authorization: hmac username="FFFFFFFF-FFFF-FFFF-FFFF-FFFFFFFFFFFF", algorithm="hmac-sha1", headers="x-date", signature="7yBJIUBhXChGLOTn7nuvCXzIe5k="' --header 'x-date: Thu, 01 Nov 2018 18:01:10 GMT' --header 'Accept-Encoding: gzip' --compressed  'https://ptx.transportdata.tw/MOTC/v2/Bus/Shape/City/Kaohsiung?$format=JSON' > data.json

# Command from 
# https://ptx.transportdata.tw/MOTC#!/CityBusApi/CityBusApi_Shape

