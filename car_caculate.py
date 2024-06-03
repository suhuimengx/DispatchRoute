# 时间窗
TimeWindows = [
    [0,0,0], [0, 10, 20], [0, 10, 20], [5, 15, 25], [5, 15, 25], [10, 25, 35], [10, 25, 35], [10, 25, 35], [15, 25, 40], [20, 30, 40], [20, 30, 40], [0, 10, 20], [0, 10, 20], [0, 10, 20], [5, 15, 30], [10, 20, 30], [15, 25, 35], [15, 25, 35], [20, 30, 40],[20, 30, 35], [20, 30, 40]
]
# 节点
pathArray = [0,
             0, 0, 8, 8, 6, 6, 8, 4, 3, 16, 0, 0, 0, 6, 2, 17, 3, 5, 10, 16,
             4, 4, 4, 4, 3, 3, 1, 12, 18, 18, 4, 4, 4, 4, 12, 10, 5, 12, 12, 4]
# 车辆需求数
NumDemand = [0,
             1, 2, 2, 2, 2, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 2, 2, 2, 1, 3,
             -1, -2, -2, -2, -2, -2, -3, -1, -2, -3, -1, -2, -3, -1, -2, -2, -2, -2, -1, -3, 1, -1]
# 三辆车的派送结果
Car_ServerLists = [
    [0, 2, 13, 4, 24, 33, 22, 16, 15, 8, 35, 28, 36, 20, 10, 30, 40],
    [0, 11, 1, 12, 31, 21, 3, 32, 23, 18, 19, 39, 38],
    [0, 7, 14, 5, 34, 27, 25, 17, 37, 6, 26, 9, 29],
]

# print(len(Car_ServerLists[0]))
# print(len(Car_ServerLists[1]))
# print(NumDemand[41])

# for i in Car_ServerLists[2]:
#     if i<= 20:
#         print(TimeWindows[i])

start_points = []
end_points = []
carLoad = 0
index = 1
car_ServerList = Car_ServerLists[2]
ServerList_length = len(car_ServerList)
# 获取初始出发点
start_points.append(car_ServerList[0])
while pathArray[start_points[-1]] == pathArray[car_ServerList[index]]:
    start_points.append(car_ServerList[index])
    index += 1
# 解析完整行驶路线，根据车辆行驶状态更新出发点和目标点
while index < ServerList_length:
    # 获取同终点的订单集合
    end_points.append(car_ServerList[index])
    index += 1
    if index < ServerList_length:
        while pathArray[end_points[-1]] == pathArray[car_ServerList[index]]:
            end_points.append(car_ServerList[index])
            index += 1
            if index >= ServerList_length: break
    # 获取本段路的出发点和到达点
    start = pathArray[start_points[-1]]
    end = pathArray[end_points[-1]]
    for point in start_points:
        carLoad += NumDemand[point]
    timew = []
    for point in start_points:
        if point <= 20:
            timew.append(TimeWindows[point])
    print("本段路程：", start_points, end_points, start, end, timew, f"载客量：{carLoad}")
    start_points = end_points
    end_points = []