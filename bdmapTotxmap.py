import math


def bdmapTotxmap(location):
    '''
    将传入的百度地图坐标转换为腾讯地图的坐标
    @params：location：传入的百度地图的坐标
    '''
    
    lat = location['latitude']
    lng = location['longitude']
    pi = (3.14159265358979324 * 3000.0) / 180.0
    x = lng - 0.0065
    y = lat - 0.006
    z = math.sqrt(x * x + y * y) - 0.00002 * math.sin(y * pi)
    theta = math.atan2(y, x) - 0.000003 * math.cos(x * pi)
    lng = z * math.cos(theta)
    lat = z * math.sin(theta)
    
    # 返回数据给客户端
    result = {'lng': lng, 'lat': lat}
    #print(result)
    return result
'''
# 示例调用
event = {'latitude': 32.058801, 'longitude': 118.783740}
bdmapTotxmap(event)
'''