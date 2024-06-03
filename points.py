import threading
import time
import requests
import random,math
from numpy import mean

# 接口地址
url = "https://api.map.baidu.com/directionlite/v1/walking"
ak = "mOwHuqpTRisg9X2n6l90Q8FGvXi9t2PR"
ratio_list = []

RodeArray = [
    "32.129368,118.958233",  # 宿舍区0   0
    "32.115939,118.967152",  # 南门      1
    "32.117235,118.969093",  # 行政南楼  2
    "32.118837,118.967116",  # 图书馆    3
    "32.116791,118.965225",  # 教学楼    4
    "32.116463,118.962791",  # 实验楼    5
    "32.117877,118.962081",  # 体育馆    6
    "32.118603,118.96492",   # 活动中心  7
    "32.119081,118.961543",  # 宿舍区1   8
    "32.120305,118.960963",  # 宿舍区2   9
    "32.119632,118.963047",  # 快递中心  10
    "32.121356,118.960393",  # 校医院    11
    "32.121914,118.961839",  # 九食堂    12
    "32.122923,118.961327",  # 气象楼    13
    "32.123668,118.959485",  # 环境学院  14
    "32.121562,118.967391",  # 信息中心  15
    "32.121524,118.965392",  # 游泳馆    16
    "32.119143,118.968783",  # 美术馆    17
    "32.121715,118.96956",   # 宿舍区3   18
    "32.122174,118.963739",  # 宿舍区4   19
    "32.12337,118.96267",    # 宿舍区5   20
    "32.125438,118.960415",  # 医学院    21
    "32.128118,118.960375",  # 建设银行  22
    "32.126012,118.963115",  # 现工院    23

]


def GetJsonResult(start, end, timeout=0.2):
    ''' 通过网络请求进行路径规划
    @start: 起始点坐标(string类型)"lat,lng"
    @end: 终点序号(string类型)"lat,lng"
    @return: 路劲规划的返回值(dict), 注意关注格式
    '''
    params = {
        "origin": start,
        "destination": end,
        "ak":       ak,
    }
    response = requests.get(url=url, params=params)
    if response:
        # 注意拿到的response.json()是字典类型
        return response.json()


def GetRodeDistance(start, end, mean_square_diff):
    ''' 获取两点间路程距离
    @start: 起始点坐标(string类型)"lng,lat"
    @end: 终点序号(string类型)"lng,lat"
    @return: 两点之间的路程距离
    '''
    # 传入坐标需要是"lat,lng"
    start_str = str(start[1]) + ',' + str(start[0])
    end_str = str(end[1]) + ',' + str(end[0])
    # print(start_str, end_str)
    res = GetJsonResult(start_str, end_str, 0.1)
    # print(res)
    # 校验申请的数据是否有效
    if res['status'] == 0:
        distance = res['result']['routes'][0]['distance']
        # linevalue = distance / math.sqrt(mean_square_diff) / 1e3
        distance_forcast = 1e3 * math.sqrt(mean_square_diff) * 146.433506
        # ratio_list.append(linevalue)
        print(distance, mean_square_diff, distance_forcast)
        return distance
    else:
        return -1

def GetForcastDistance(mean_square_diff, ratio = 146.433506):
    distance_forcast = int(1e3 * math.sqrt(mean_square_diff) * ratio)
    return distance_forcast


def GetPathArray_str(steps):
    ''' 获取全部中间节点
    @steps: 路径规划返回值里的行走路径
    @return: 将steps中的path全部存到一个数组里返回,字符串形式
    '''
    tempArray = []
    cnt = 0
    length = len(steps)
    # 获取原始path数据的同时，进行抽样，缓解之后请求距离时间过长的压力
    for step in steps:
        cnt += 1
        paths = step['path'].split(';')
        # 先随机抽样
        sample_list = [paths[0]] + random.sample(paths[1:-1], int((len(paths) - 2)//1.5)) + [paths[-1]]
        # sample_list = paths
        # 通过列表推导式确保数据顺序
        tempArray += [x for x in paths if x in sample_list]
        if cnt < length:
            tempArray.pop()
    return tempArray

def GetPathArray(tempArray):
    ''' 获取全部中间节点
    @tempArray: 经过抽样处理并提取后的路径数组
    @return: 将tempArray中的字符串转为浮点数全部存到一个数组里返回,浮点数形式
    '''
    pathArray = []
    for path in tempArray:
        # 转为float且只保留小数点后6位
        path_num = [round(float(i), 6) for i in (path.split(','))]
        pathArray.append(path_num)
    return pathArray

def Convert2GD(pathArray):
    '''输出高德地图形式的集合,使用异步并发形式提高运行效率
    @pathArray: 经过抽样提取并估算插值后的路径数组
    @ereturn: 返回高德地图参考系的路径坐标
    '''
    start_time = time.time()
    PathArray_gaode = []
    local_length = len(pathArray)
    # for j in range(times + 1):
    #     locations_str = ""
    #     local_length = length if (j+1)*40 > length else (j+1)*40
    #     for i in range(j*40, local_length):
    locations_str = ""
    for i in range(local_length):
        lng_str = "{:.6f}".format(pathArray[i][0])
        lat_str = "{:.6f}".format(pathArray[i][1])
        locations_str += lng_str + ',' + lat_str
        if i < local_length - 1:
            locations_str += '|'
    params = {
        'locations': locations_str,
        'coordsys':'baidu',
        'output': 'json',
        'key': '91d3fd24d3add11f8401036da6e86879'
    }
    results = requests.get(url="https://restapi.amap.com/v3/assistant/coordinate/convert", params=params).json()['locations']
    result_list = results.split(';')
    for result in result_list:
        PathArray_gaode.append([round(float(i), 6) for i in (result.split(','))])
    print(f"坐标转换耗时：{time.time() - start_time}")
    return PathArray_gaode

def InsertNode(tempArray, freq):
    '''通过线性插值让路径数组按距离均匀分布
    @pathArray: 通过路径规划抽样获取的原始路径数组，字符串形式
    @return: 线性插值后获取的新路径数组
    '''
    start_time = time.time()
    pathArray = GetPathArray(tempArray)
    PathArray_baidu = []
    # 遍历百度地图返回的path点，进而插入新点
    for i in range(len(pathArray) - 1):
        PathArray_baidu.append(pathArray[i])
        # 根据均方误差粗略判断距离
        mean_square_diff = ((pathArray[i][0]-pathArray[i+1][0])**2 + (pathArray[i][1]-pathArray[i+1][1])**2) / 2
        # print(f"两点均方误差为：{mean_square_diff}")
        if mean_square_diff < 1e-8:
            # print("两点距离较近，不必估算中间点")
            continue
        local_distance = GetForcastDistance(mean_square_diff)
        # print(f'两点距离为：{local_distance}')
        # 判断距离是否较大，需要插入新的点
        if local_distance > (1.5 * freq):
            insertNum = local_distance // freq
            dlng = (pathArray[i+1][0] - pathArray[i][0]) / insertNum
            dlat = (pathArray[i+1][1] - pathArray[i][1]) / insertNum
            # 插入新估算出的点
            for j in range(1, insertNum):
                # print("生成：",[round((pathArray[i][0] + j*dlng), 6), round((pathArray[i][1] + j*dlat), 6)])
                PathArray_baidu.append([round((pathArray[i][0] + j*dlng), 6), round((pathArray[i][1] + j*dlat), 6)])
    # 放入最后一个没遍历的点
    PathArray_baidu.append(pathArray[-1])
    print(f"线性插值耗时：{time.time() - start_time}")
    return PathArray_baidu

def GetRealTimeArray(start, end, freq):
    res = GetJsonResult(start, end)
    distance_total = res['result']['routes'][0]['distance']
    PathArray_gaode = []
    PathArray_baidu = []
    # 校验数据是否有效
    if res['status'] == 0:
        # 得到路径数组
        steps = res['result']['routes'][0]['steps']
        tempArray = GetPathArray_str(steps)
        PathArray_baidu = InsertNode(tempArray, freq)
        PathArray_gaode = Convert2GD(PathArray_baidu)
    return {
        'distance':distance_total,
        'location_array':PathArray_baidu,
        'location_array_amap':PathArray_gaode
    }
