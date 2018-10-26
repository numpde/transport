# Query bus ETA by bus stop name
https://ibus.tbkc.gov.tw/KSBUSN/newAPI/CrossRoute.ashx?stopname=建國一路

# Query bus geo-locations by route ID (or IDs: 122_,1221)
https://ibus.tbkc.gov.tw/KSBUSN/NewAPI/RealRoute.ashx?type=GetBusInfo&data=122&Lang=Cht

# Purpose unknown
https://ibus.tbkc.gov.tw/KSBUSN/NewAPI/CountRoute.ashx?routeid=122&split=n&type=web

# Get all busses (currently circulating?)
https://ibus.tbkc.gov.tw/KSBUSN/NewAPI/RealRoute.ashx?type=GetAllBus

# Get all route descriptions
https://ibus.tbkc.gov.tw/KSBUSN/NewAPI/RealRoute.ashx?type=GetRoute

# Get all stops of a route group (=master route)
https://ibus.tbkc.gov.tw/KSBUSN/NewAPI/RealRoute.ashx?type=GetGroupStop&Data=98215397&Lang=Cht

# Get all stops of a route (direction 1 or 2)
https://ibus.tbkc.gov.tw/KSBUSN/NewAPI/RealRoute.ashx?type=GetStop&Data=1431_,1&Lang=Cht
https://ibus.tbkc.gov.tw/KSBUSN/NewAPI/RealRoute.ashx?type=GetStop&Data=1431_,2&Lang=Cht

