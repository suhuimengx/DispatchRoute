import json
import datetime
from flask import Flask, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO
from HuaweiIoT import HuaweiCloudObj
from time_sys import SystemClock, System_file
import threading, time
from points import RodeArray, GetPathArray, GetRealTimeArray
from tqdm import tqdm
import requests
import numpy as np
import xlrd as xd

"""
动态插入接口 start
"""
ArriveCar = [0,0,0]
dynamic_cnt = [0, 0, 0]
previous_cnt = [0, 0, 0]
myCustomer = [20, 20, 20]

'''
静态共乘调度结果
'''
dataDict = {}
# 时间窗
dataDict["Timewindow"] = [
    [0,0,0],
    [0, 10, 20], [0, 10, 20], [5, 15, 25], [5, 15, 25], [10, 25, 35], [10, 25, 35], [10, 25, 35], [15, 25, 40], [20, 30, 40], [20, 30, 40], [0, 10, 20], [0, 10, 20], [0, 10, 20], [5, 15, 30], [10, 20, 30], [15, 25, 35], [15, 25, 35], [20, 30, 40],[20, 30, 35], [20, 30, 40]
]
# 节点
dataDict["NodeCoor"] = [0,
             0, 0, 8, 8, 6, 6, 8, 4, 3, 16, 0, 0, 0, 6, 2, 17, 3, 5, 10, 16,
             4, 4, 4, 4, 3, 3, 1, 12, 18, 18, 4, 4, 4, 4, 12, 10, 5, 12, 12, 4]

# 车辆需求数
dataDict["Demand"] = [0,
             1, 2, 2, 2, 2, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 2, 2, 2, 1, 3,
             -1, -2, -2, -2, -2, -2, -3, -1, -2, -3, -1, -2, -3, -1, -2, -2, -2, -2, -1, -3]
dataDict['MaxLoad'] = 7
dataDict['nCustomer'] = len(dataDict['Timewindow']) - 1
dataDict['ServiceTime'] = 1
# 最佳个体
bestInd = [0, 2, 13, 4, 24, 33, 22, 16, 15, 8, 35, 28, 36, 20, 10, 30, 40, 0, 11, 1, 12, 31, 21, 3, 32, 23, 18, 19, 39, 38, 0, 7, 14, 5, 34, 27, 25, 17, 37, 6, 26, 9, 29]
# data = xd.open_workbook("F:/Lemon/Desktop/HikingUs/DispatchRoute/node_distance_1.xls")
data = xd.open_workbook("./node_distance_1.xls")


sheet = data.sheet_by_name('Sheet1')
DijDis = []
for r in range(sheet.nrows): #将表中数据按行逐步添加到列表中，最后转换为list结构
    data1 = []
    for c in range(sheet.ncols):
        data1.append(sheet.cell_value(r,c))
    DijDis.append(list(data1))   
for i in range(len(DijDis)):
    for j in range(i+1,len(DijDis)):
        DijDis[i][j]=DijDis[j][i]
nNodes=len(DijDis)
dataDict['speed'] = 1000/3 #车速为20km/h
nSAV = 3
DijTime = np.zeros((nNodes,nNodes))
for i in range(nNodes):
    for j in range(nNodes):
        DijTime[i,j] = DijDis[i][j]/dataDict['speed']

# 染色体解码
def decodeInd(ind):
    '''从染色体解码回路线片段，每条路径都是以0为开头'''
    indCopy = ind.copy() # 复制ind，防止直接对染色体进行改动
    indCopy.append(0)
    zeroIdx = [x for x, y in list(enumerate(indCopy)) if y == 0]
    routes = []
    for i,j in zip(zeroIdx[0::], zeroIdx[1::]):
        routes.append(ind[i:j])
    return routes

def combineInd(routes):
    # 合并函数，将生成的分段route合并为完整的染色体
    Ind = []
    for _ in routes:
        Ind += _
    return Ind

# 定义负载约束的惩罚函数
def loadPenalty(routes):
    penalty = 0
    for eachRoute in routes:
        routeLoad = 0
        i = 0
        while i<=len(eachRoute)-1:
            if i<len(eachRoute)-1 and dataDict['NodeCoor'][eachRoute[i+1]]==dataDict['NodeCoor'][eachRoute[i]] and dataDict['Demand'][eachRoute[i]]>0 and dataDict['Demand'][eachRoute[i+1]]<0:
                routeLoad = routeLoad + dataDict['Demand'][eachRoute[i]] + dataDict['Demand'][eachRoute[i+1]]
                i+=2
                if routeLoad > dataDict['MaxLoad']:
                    penalty += routeLoad - dataDict['MaxLoad'] # 容量超限还是有必要累加的，一步错步步错
            else:
                routeLoad += dataDict['Demand'][eachRoute[i]]
                if routeLoad > dataDict['MaxLoad']:
                    penalty += routeLoad - dataDict['MaxLoad'] # 容量超限还是有必要累加的，一步错步步错
                i+=1
    return penalty

# 辅助函数，根据给定路径，计算到达该路径上各个乘客的时间
def calcuRouteServiceTime(route, dataDict = dataDict):
    serviceTime = [0] * len(route)
    arrivalTime = 0
    for i in range(1, len(route)):
        if dataDict['NodeCoor'][route[i]]!=dataDict['NodeCoor'][route[i-1]]:
            arrivalTime += dataDict['ServiceTime']
        arrivalTime += DijTime[dataDict['NodeCoor'][route[i-1]]][dataDict['NodeCoor'][route[i]]]
        if route[i] <= dataDict['nCustomer']:
            arrivalTime = max(arrivalTime, dataDict['Timewindow'][route[i]][0])
        serviceTime[i] = arrivalTime
    return  serviceTime

def timeTable(distributionPlan, dataDict = dataDict):
    timeArrangement = []
    for eachRoute in distributionPlan:
        serviceTime = calcuRouteServiceTime(eachRoute)
        timeArrangement.append(serviceTime)
        # 将数组重新排序为与基因编码一致的排列方式
    realignedTimeArrangement = []
    for routeTime in timeArrangement:
            realignedTimeArrangement = realignedTimeArrangement + routeTime
    return realignedTimeArrangement

def timePenalty(ind, routes):
    """辅助函数，对不能按照pick-up和drop-off时间窗到达的染色体编码进行惩罚"""
    timeArrangement = timeTable(routes)
    desiredTime = [0]*len(ind)
    for i in range(len(ind)):
        if ind[i] <= dataDict['nCustomer']:
            desiredTime[i] = dataDict['Timewindow'][ind[i]][1]
        else:
            desiredTime[i] = dataDict['Timewindow'][ind[i]-dataDict['nCustomer']][2]
    # desireTime是将pick-up和drop-off的最晚时间糅合在一起，按照ind的排序，按照节点排序自动pickup/dropoff时间
    timeDelay = [max(timeArrangement[i]-desiredTime[i],0) for i in range(len(ind))]
    return np.sum(timeDelay)

def panevaluate(ind, c1=500.0, c2=100.0):
    routes = decodeInd(ind) # 将个体解码为路线
    return c1*loadPenalty(routes) + c2*timePenalty(ind,routes)

def insert(req, bestInd, dataDict):#req按照[起点，目的地，当前时刻，最晚上车时间，最晚下车时间，人数]来输入
    #首先先把起点目的地，时间窗插入到原来的集合数组里
    for i in range(len(bestInd)):
        if bestInd[i]>dataDict['nCustomer']:
            bestInd[i] += 1
    dataDict['nCustomer'] += 1
    dataDict['NodeCoor'].insert(dataDict['nCustomer'], req[0])
    dataDict['NodeCoor'].append(req[1])
    dataDict['Timewindow'].append([req[2],req[3],req[4]])
    dataDict['Demand'].insert(dataDict['nCustomer'], req[5])
    dataDict['Demand'].append(-req[5])
    distributionPlan = decodeInd(bestInd)
    bestEva = float("inf")
    for i in range(len(distributionPlan)):
        # 遍历三辆车
        Plan_copy = distributionPlan.copy()
        indexo = ArriveCar[i] + 1
        if indexo == distributionPlan[i][-1]:
            Plan_copy[i].append(dataDict['nCustomer'])
            Plan_copy[i].append(2*dataDict['nCustomer'])
            betterInd = combineInd(Plan_copy)
            betterEva = panevaluate(betterInd)
        else:
            betterEva = float("inf")
            routeEva = float("inf")
            for seq1 in range(indexo,len(Plan_copy[i])+1):
                for seq2 in range(seq1+1, len(Plan_copy[i])+2):
                    route_copy = Plan_copy[i].copy()
                    route_copy.insert(seq1, dataDict['nCustomer'])
                    route_copy.insert(seq2, 2*dataDict['nCustomer'])
                    if panevaluate(route_copy) < routeEva:
                        routeEva = panevaluate(route_copy)
                        betterRoute = route_copy
            Plan_copy[i] = betterRoute
            betterInd = combineInd(Plan_copy)
            betterEva = panevaluate(betterInd)
        if betterEva < bestEva:
            bestEva = betterEva
            bestInd = betterInd
    return bestInd,bestEva
"""
动态插入接口 end
"""

'''
华为云物联网平台客户端配置
'''

# # 项目id
# # project_id = "b9a181cca17a4c72874effe2cb82fe0e"
# project_id = "eb375db7c7674f0e99719592224806ee"
# # 上海一"cn-east-3"；北京四"cn-north-4"；华南广州"cn-south-4"
# region_id = "cn-north-4"  # 服务区域
# # 接入端点
# # endpoint = "cd5b167852.iot-mqtts.cn-north-4.myhuaweicloud.com"
# endpoint = "4c3dd8f578.st1.iotda-device.cn-north-4.myhuaweicloud.com"

# # 产品id
# # product_id = "649a72522a3b1d3de71e9e81"
# product_id = "66505a617dbfd46fabbd3225"

# 项目id
# project_id = "b9a181cca17a4c72874effe2cb82fe0e"
project_id = "eb375db7c7674f0e99719592224806ee"

# 上海一"cn-east-3"；北京四"cn-north-4"；华南广州"cn-south-4"
region_id = "cn-north-4"  # 服务区域
# 接入端点
# endpoint = "cd5b167852.iot-mqtts.cn-north-4.myhuaweicloud.com"
# endpoint = "4c3dd8f578.st1.iotda-device.cn-north-4.myhuaweicloud.com"
endpoint = "4c3dd8f578.st1.iotda-app.cn-north-4.myhuaweicloud.com"

# 产品id
# product_id = "649a72522a3b1d3de71e9e81"
product_id = "66505a617dbfd46fabbd3225"

# 设备id
# device_id = "649a72522a3b1d3de71e9e81_car01"
device_id = "66505a617dbfd46fabbd3225_led001"

# 服务id
# service_id = "car_01"
service_id = "hhhcar1"
# 设备id
# device_id_01 = "649a72522a3b1d3de71e9e81_car01"
# device_id_02 = "649a72522a3b1d3de71e9e81_car02"
# device_id_03 = "649a72522a3b1d3de71e9e81_car03"
# device_id_list = [device_id_01, device_id_02, device_id_03]

device_id_01 = "66505a617dbfd46fabbd3225_led001"
device_id_02 = "66505a617dbfd46fabbd3225_car02"
device_id_03 = "66505a617dbfd46fabbd3225_car03"
device_id_list = [device_id_01, device_id_02, device_id_03]

# 服务id
# service_id = "car_01"
service_id = "hhhcar1"


# 初始化华为云IOT对象
Cloud = HuaweiCloudObj(project_id, region_id, endpoint)

# 初始化系统时钟对象
Clock = SystemClock(0.325) # 0.065 -> 1s; 3.9 -> 1min
# 初始化系统文件对象
FileObj = System_file("./路径距离.xls")

# uniCloud云服务空间url
# uni_url_count = "https://fc-mp-9807717d-5acc-468e-a8a2-2b6850a56613.next.bspapp.com/getcount"
# uni_url_doc = "https://fc-mp-9807717d-5acc-468e-a8a2-2b6850a56613.next.bspapp.com/downdata"
# uni_url_id = ("https://fc-mp-9807717d-5acc-468e-a8a2-2b6850a56613.next.bspapp.com/getold")

# uni_url_count = "https://fc-mp-799372a1-cb92-40f1-8736-2de08a0afe18.next.bspapp.com/getcount"
# uni_url_doc = "https://fc-mp-799372a1-cb92-40f1-8736-2de08a0afe18.next.bspapp.com/downdata"
# uni_url_id = ("https://fc-mp-799372a1-cb92-40f1-8736-2de08a0afe18.next.bspapp.com/getold")
uni_url_count = "https://fc-mp-e8d46985-e50c-4bfb-af14-d4dfd054e7b5.next.bspapp.com/getCount"
uni_url_doc = "https://fc-mp-e8d46985-e50c-4bfb-af14-d4dfd054e7b5.next.bspapp.com/downloadData"
uni_url_id = ("https://fc-mp-e8d46985-e50c-4bfb-af14-d4dfd054e7b5.next.bspapp.com/getOld")
# 系统开始运行时 数据库中的数据量
init_count = json.loads(requests.get(uni_url_count).text)["total"]
print(init_count)
# 获取数据库中最后一个订单的id
init_id = json.loads(requests.get(uni_url_id, {"num":init_count}).text)
print(init_id)
init_id = json.loads(requests.get(uni_url_id, {"num":init_count}).text)["data"][0]["_id"]



# 三辆车的派送结果
Car_ServerLists = decodeInd(bestInd)
Car_ServerLists_Length = [len(Car_ServerLists[0]), len(Car_ServerLists[1]), len(Car_ServerLists[2])]
# Car_ServerLists = [
#     [0, 2, 13, 4, 24, 33, 22, 16, 15, 8, 35, 28, 36, 20, 10, 30, 40],
#     [0, 11, 1, 12, 31, 21, 3, 32, 23, 18, 19, 39, 38],
#     [0, 7, 14, 5, 34, 27, 25, 17, 37, 6, 26, 9, 29],
# ]

'''
数字孪生端websocket服务创建
'''
app = Flask(__name__)
# 解决前端跨域问题
CORS(app, supports_credentials=True)
app.config['SECRET_KEY'] = '123456789'
socketio = SocketIO(app, cors_allowed_origins='*')

# 定义路由处理websocket连接
@app.route('/socket.io/')
# @app.route('/sss')
def socket():
    return jsonify({})

@socketio.on('connect')
def socket_connect():
    """
    监听websocket连接事件
    """
    print('client connected')
    # 设置调度任务分别下发给三个小车的多线程任务
    for i in range(len(Car_ServerLists)):
        # threading.Thread()就是执行多线程任务，target是对应函数，args就是该函数所要求的参数
        thread = threading.Thread(target=ProcessSchRes, args=(device_id_list[i], Car_ServerLists[i]))
        thread.start()
    uni_thread = threading.Thread(target=QueryUniCloud, args=(uni_url_count, uni_url_doc))
    uni_thread.start()
    Clock.start()

@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected')


def GetLocation(interval, device_id):
    """9
    多线程定时服务，周期性更新车辆实时位置
    @param device_id: 要更新实时位置的设备id
    @param interval: 更新实时位置的周期间隔
    """
    # 第一个参数是间隔时间，第二是是执行的函数，也就是说间隔interval时间之后执行GetLocation函数
    t = threading.Timer(interval, GetLocation, args=[interval, device_id])
    t.start()
    location = Cloud.GetDeviceLocation(device_id)
    car_id = 0
    # 根据设备id发送对应小车的实时位置
    if device_id == device_id_01: car_id = 1
    elif device_id == device_id_02: car_id = 2
    else: car_id = 3
    data = {
        'system_ConvertedTime': Clock.get_current_ConvertedTime(),
        'system_time': Clock.get_current_time(),
        'car_id': car_id,
        'location': location,
    }
    socketio.emit('send_message_location', data)

"""这个calcutime函数好像没用上。。"""
def CalcuTime_dif(timestamp):
    """
    通过时间戳获取在系统参考系中的时间
    @param timestamp: 订单的时间戳
    @return: 与7：20的差值
    """
    timestamp = timestamp / 1000
    dt = datetime.datetime.fromtimestamp(timestamp)
    target_time = datetime.datetime(dt.year, dt.month, dt.day, 7, 20)
    time_dif = dt - target_time
    time_dif_min = time_dif.seconds // 60
    return time_dif_min

def QueryUniCloud(url_count, url_doc):
    """ 多线程任务
    轮询是否有新数据到来，即实时订单
    @param url_count: 查询数量
    @param url_doc: 获取新数据
    @return:
    """
    global init_count, init_id, dynamic_cnt
    global bestInd, dataDict, Car_ServerLists, Car_ServerLists_Length
    while True:
        # 查询数据库是否有新数据
        current_count = json.loads(requests.get(url_count).text)['total']
        # print("目前数量",current_count)
        if current_count > init_count:
            time.sleep(1)
            print("=============收到实时需求=============")

            # print(json.loads(requests.get(url_doc, {
            #     "old_id": init_id,
            # }, timeout=3).text))

            
            # new_doc = json.loads(requests.get(url_doc, {
            #     "old_id": init_id,
            # }, timeout=3).text)["data"][0]

            # 更新状态量
            init_count += 1

            new_doc = json.loads(requests.get(uni_url_id, {"num":init_count}).text)["data"][0]

            print(new_doc)

            init_id = new_doc["_id"]
            originId = int(new_doc["originId"])
            destId = int(new_doc["destId"])
            # originTime = CalcuTime_dif(new_doc["DepartureTime"])
            originTime = Clock.get_current_ConvertedTime()
            originTime_latest = originTime + new_doc["waitTime"]
            # destTime = CalcuTime_dif(new_doc["ArrivalTime"])
            destTime = originTime_latest + 8
            numDemand = new_doc["numDemand"]
            req = [originId, destId, originTime, originTime_latest, destTime, numDemand]
            print("需求信息：",req)
            # dataDict为引用传递，调用函数时自动修改
            bestInd, fitness = insert(req, bestInd, dataDict)
            print(bestInd, fitness, dataDict["nCustomer"])
            # 刷新车辆服务列表
            Car_ServerLists = decodeInd(bestInd)
            # 前端显示实时订单由哪个车辆服务
            insert_car_id = 0
            if len(Car_ServerLists[0]) > Car_ServerLists_Length[0]:
                insert_car_id = 1
            elif len(Car_ServerLists[1]) > Car_ServerLists_Length[1]:
                insert_car_id = 2
            else:
                insert_car_id = 3
            socketio.emit("send_message_realtimeDemand", {
                'car_id': insert_car_id,
                'originId': originId,
                'destId': destId,
            })
            Car_ServerLists_Length = [len(Car_ServerLists[0]), len(Car_ServerLists[1]), len(Car_ServerLists[2])]
            for i in range(len(dynamic_cnt)):
                dynamic_cnt[i] += 1
        time.sleep(2.5)
        

def ProcessSchRes(device_id, car_ServerList):
    """ 多线程任务
    按照逻辑与时序，下发共乘调度规划结果
    并将实时信息同步给数字孪生平台
    @param device_id: 小车设备id
    @param car_ServerList: 共乘调度结果中车辆的路线
    """
    global dataDict, Car_ServerLists, Car_ServerLists_Length
    global dynamic_cnt, previous_cnt, myCustomer
    index = 1
    start_points = []
    end_points = []
    CarLoad = 0
    CarStopFlag = False
    initFlag = False
    car_id = 1 if device_id == device_id_01 else (2 if device_id == device_id_02 else 3)
    car_ServerList = Car_ServerLists[car_id - 1]
    # 获取初始出发点
    start_points.append(car_ServerList[0])
    while dataDict["NodeCoor"][start_points[-1]] == dataDict["NodeCoor"][car_ServerList[index]]:
        start_points.append(car_ServerList[index])
        index += 1
    while True:
        # 解析完整行驶路线，根据车辆行驶状态更新出发点和目标点
        # if index < len(car_ServerList):
        if index < Car_ServerLists_Length[car_id - 1]:
            # 若动态插入端出现变动，及时做出响应
            if dynamic_cnt[car_id - 1] != previous_cnt[car_id - 1]:
                for point_index in range(len(start_points)):
                    if start_points[point_index] > myCustomer[car_id - 1]:
                        start_points[point_index] += 1
                myCustomer[car_id - 1] = dataDict["nCustomer"]
                previous_cnt[car_id - 1] = dynamic_cnt[car_id - 1]
            car_ServerList = Car_ServerLists[car_id - 1]
            print(Car_ServerLists_Length[car_id - 1])
            # 获取同终点的订单集合
            end_points.append(car_ServerList[index])
            index += 1
            if index < len(car_ServerList):
                while dataDict["NodeCoor"][end_points[-1]] == dataDict["NodeCoor"][car_ServerList[index]]:
                    end_points.append(car_ServerList[index])
                    index += 1
                    if index >= Car_ServerLists_Length[car_id - 1]: break
            # 记录小车即将到达的终点
            ArriveCar[car_id - 1] = car_ServerList.index(end_points[-1])
            # 获取本次行程可出发时间
            departure_time = 0
            for point in start_points:
                if point <= dataDict["nCustomer"]:
                    if dataDict["Timewindow"][point][0] > departure_time: departure_time = dataDict["Timewindow"][point][0]
            # 获取本段路的出发点和到达点
            start = dataDict["NodeCoor"][start_points[-1]]
            end = dataDict["NodeCoor"][end_points[-1]]
            # 处理动态插入bug：插入到了即将到达的点
            if start == end:
                increase_load = 0
                # 更新车载人数并跳过本次循环
                for point in start_points:
                    if dataDict["Demand"][point] > 0:
                        increase_load += dataDict["Demand"][point]
                CarLoad += increase_load
                message = {
                    'car_id': car_id,
                    'PassengerNum': CarLoad,
                    'type': 'get_in_car',
                    'various_num': increase_load,
                    'time_stamp': Clock.get_current_time(),
                }
                socketio.emit('send_message_carLoad', message)
                start_points = end_points
                end_points = []
                # 跳过本次循环
                continue
            # 从xls文件读取路网信息
            result = FileObj.get_direct_array(start, end)
            distance_total, location_array_total_str, location_array_amap_total_str = [value for value in result.values()]
            location_array_total = GetPathArray(location_array_total_str.split(';'))
            location_array_amap_total = GetPathArray(location_array_amap_total_str.split(';'))
            # 判断是否已达到可出发时间，否则阻塞该线程
            while departure_time > Clock.get_current_ConvertedTime():
                time.sleep(0.1)
            # 刷新上车后的载客量
            increase_load = 0
            for point in start_points:
                if dataDict["Demand"][point] > 0:
                    increase_load += dataDict["Demand"][point]
            CarLoad += increase_load
            # 通过websocket将载客量及变化消息刷新到web端
            message = {
                'car_id': car_id,
                'PassengerNum': CarLoad,
                'type': 'get_in_car',
                'various_num': increase_load,
                'time_stamp': Clock.get_current_time(),
            }
            socketio.emit('send_message_carLoad', message)
            print(f"{car_id}号车辆 载客数: {CarLoad}", start_points, end_points, start, end, "系统时钟：", Clock.get_current_ConvertedTime())
            # 通过华为云物联网平台下发路径命令
            # res = Cloud.SendArrayCommand(device_id, "car_01", location_array_total)
            res = Cloud.SendArrayCommand(device_id, "hhhcar1", location_array_total)

            print(f"{car_id}号车任务及行驶信息下发状态：{res}")
            # 设置更新3辆小车实时位置的多线程周期性任务
            if not initFlag:
                time.sleep(0.5)
                if car_id == 1: GetLocation(0.45, device_id_01)
                elif car_id == 2: GetLocation(0.45, device_id_02)
                else: GetLocation(0.45, device_id_03)
                initFlag = True
            time.sleep(0.25)
            # 通过websocket服务器将数据发送至数字孪生平台
            data = {
                'car_id': car_id,
                'distance': distance_total,
                'location_array':location_array_total,
                'location_array_amap':location_array_amap_total,
                'PassengerNum': CarLoad
            }
            socketio.emit('send_message', data)
            # 如何判断可以继续下发下一跳分段路径?
            while not CarStopFlag:
                property_flag = Cloud.GetDeviceFlag(device_id)
                if property_flag == 1:
                    CarStopFlag = True
                    res = Cloud.ResetStopFlag(device_id)
                    # print(f"{car_id}号车辆状态重置：",res)
                    time.sleep(2.5)
            CarStopFlag = False
            # 刷新下车后的载客量
            decrease_load = 0
            for point in end_points:
                if dataDict["Demand"][point] < 0:
                    decrease_load += dataDict["Demand"][point]
            CarLoad += decrease_load
            message = {
                'car_id': car_id,
                'PassengerNum': CarLoad,
                'type': 'get_out_car',
                'various_num': decrease_load,
                'time_stamp': Clock.get_current_time(),
            }
            socketio.emit('send_message_carLoad', message)
            start_points = end_points
            end_points = []
            time.sleep(0.5)
            # 刷新路线片段
            car_ServerList = Car_ServerLists[car_id - 1]


if __name__ == '__main__':
    # socketio.run(app, host='0.0.0.0',port=5000, allow_unsafe_werkzeug=True)
    socketio.run(app,port=5000, allow_unsafe_werkzeug=True)




