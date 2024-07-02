import xlrd as xd
import numpy as np



data = xd.open_workbook("F:/Lemon/Desktop/HikingUs/DispatchRoute/node_distance_1.xls")
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
dataDict = {}
dataDict['speed'] = 1000/3 #车速为20km/h
nSAV = 3
DijTime = np.zeros((nNodes,nNodes))
for i in range(nNodes):
    for j in range(nNodes):
        DijTime[i,j] = DijDis[i][j]/dataDict['speed']
dataDict['NodeCoor'] =   [0,0,0,8,8,6,6,8,4,3,16,0,0,0,6,2,17,3,5,10,16,
                4,4,4,4,3,3,1,12,18,18,4,4,4,4,12,10,5,12,12,4]
dataDict['Timewindow'] = [[0,0,0],[0,10,20],[0,10,20],[5,15,25],[5,15,25],[10,25,35],[10,25,35],[10,25,35],[15,25,40],[20,30,40],[20,30,40],
                          [0,10,20],[0,10,20],[0,10,20],[5,15,30],[10,20,30],[15,25,35],[15,25,35],[20,30,40],[20,30,35],[20,30,40]]
dataDict['ServiceTime'] = 1
dataDict['Demand'] =   [0,1,2,2,2,2,2,3,1,2,3,1,2,3,1,2,2,2,2,1,3,
                        -1,-2,-2,-2,-2,-2,-3,-1,-2,-3,-1,-2,-3,-1,-2,-2,-2,-2,-1,-3]
dataDict['MaxLoad'] = 7
dataDict['nCustomer'] = len(dataDict['Timewindow'])-1

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
# print(decodeInd([0, 2, 13, 4, 24, 33, 22, 16, 15, 8, 35, 28, 36, 20, 10, 30, 40, 0, 11, 1, 12, 31, 21, 3, 32, 23, 18, 19, 39, 38, 0, 7, 14, 5, 34, 27, 25, 17, 37, 6, 26, 9, 29]))

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
"""
routes = [[0,1,2,3,4,5,6]]
load = loadPenalty(routes)
print(load)
"""

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

# print(DijTime)
# print(calcuRouteServiceTime([0,7,6,16,17,9,8,19,3,18,13],dataDict))

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

# print(timeTable([[0, 7, 6, 16, 17, 9, 8, 19, 3, 18, 13], [0, 2, 12], [0, 1, 11, 5, 15], [0, 10, 20, 4, 14]]))
# print(len(timeTable([[0, 7, 6, 16, 17, 9, 8, 19, 3, 18, 13], [0, 2, 12], [0, 1, 11, 5, 15], [0, 10, 20, 4, 14]])))

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

## print(timePanalty([0, 7, 6, 16, 17, 9, 8, 19, 3, 18, 13, 0, 2, 12, 0, 1, 11, 5, 15, 0, 10, 20, 4, 14],
##                  [[0, 7, 6, 16, 17, 9, 8, 19, 3, 18, 13], [0, 2, 12], [0, 1, 11, 5, 15], [0, 10, 20, 4, 14]]))

'''
def calRouteLen(routes,dataDict=dataDict):
    totalDistance = 0 # 记录各条路线的总长度
    for eachRoute in routes:
        # 从每条路径中抽取相邻两个节点，计算节点距离并进行累加
        for i,j in zip(eachRoute[0::], eachRoute[1::]):
            totalDistance += DijDis[dataDict['NodeCoor'][i]][dataDict['NodeCoor'][j]]
    return totalDistance
'''
"""
print(DijDis[0,10])
print(calRouteLen([[0, 0],[0, 1]]))
print(calRouteLen([[0, 7, 6, 16, 17, 9, 8, 19, 3, 18, 13], [0, 2, 12], [0, 1, 11, 5, 15], [0, 10, 20, 4, 14]]))
"""

def combineInd(routes):
    # 合并函数，将生成的分段route合并为完整的染色体
    Ind = []
    for _ in routes:
        Ind += _
    return Ind

'''
def evaluate(ind, c1=500.0, c2=10.0):
    routes = decodeInd(ind) # 将个体解码为路线
    totalDistance = calRouteLen(routes)
    return (totalDistance + c1*loadPenalty(routes) + c2*timePenalty(ind,routes)),
#print(evaluate([0, 3, 13, 0, 2, 6, 4, 1, 12, 10, 20, 11, 5, 14, 16, 8, 15, 7, 18, 17, 0, 9, 19]))


def singlevaluate(ind, c1=500.0, c2=10.0):
    routes = decodeInd(ind) # 将个体解码为路线
    totalDistance = calRouteLen(routes)
    return totalDistance + c1*loadPenalty(routes) + c2*timePenalty(ind,routes)
'''

def panevaluate(ind, c1=500.0, c2=100.0):
    routes = decodeInd(ind) # 将个体解码为路线
    return c1*loadPenalty(routes) + c2*timePenalty(ind,routes)
'''
def removeDropoff(ind):
    # 辅助函数，将生成的染色体还原为插入下车点前的染色体
    indPickup = [i for i in ind if i<=dataDict['nCustomer']]
    return indPickup
# print(removeDropoff([0, 7, 6, 16, 17, 9, 8, 19, 3, 18, 13, 0, 2, 12, 0, 1, 11, 5, 15, 0, 10, 20, 4, 14]))

def sequence(ind):
    # 辅助函数，将染色体中多余的0去掉，确定次序
    ind_seq = []
    for n in ind:
        if n not in ind_seq:
            ind_seq.append(n)
    return ind_seq


bestInd =[0, 2, 13, 4, 24, 33, 22, 16, 15, 8, 35, 28, 36, 20, 10, 30, 40, 0, 11, 1, 12, 31, 21, 3, 32, 23, 18, 19, 39, 38, 0, 7, 14, 5, 34, 27, 25, 17, 37, 6, 26, 9, 29]
print('最佳个体为：'+str(bestInd))
print('最佳适应度为：'+str(singlevaluate(bestInd)))


def calLoad(routes):
    loads = []
    for eachRoute in routes:
        routeLoad = 0
        eachload = []
        i=0
        while i<=len(eachRoute)-1:
            if i<len(eachRoute)-1 and dataDict['NodeCoor'][eachRoute[i+1]]==dataDict['NodeCoor'][eachRoute[i]] and dataDict['Demand'][eachRoute[i]]>0 and dataDict['Demand'][eachRoute[i+1]]<0:
                routeLoad = routeLoad + dataDict['Demand'][eachRoute[i]] + dataDict['Demand'][eachRoute[i+1]]
                eachload = eachload +[routeLoad]+[routeLoad]
                i+=2
            else:
                routeLoad += dataDict['Demand'][eachRoute[i]]
                eachload.append(routeLoad) # 容量超限还是有必要累加的，一步错步步错
                i+=1
        loads.append(eachload)
    return loads
distributionPlan = decodeInd(bestInd)

print('总运输距离为：'+str(singlevaluate(bestInd,c1=0,c2=0)))

print('各辆车上负载为：'+str(calLoad(distributionPlan)))

timeArrangement = timeTable(distributionPlan) # 对给定路线，计算到达每个客户的时间
# 索引给定的最迟到达时间
desiredTime = [0] * len(bestInd)
for i in range(len(bestInd)):
    if bestInd[i] <= dataDict['nCustomer']:
        desiredTime[i] = dataDict['Timewindow'][bestInd[i]][1]
    else:
        desiredTime[i] = dataDict['Timewindow'][bestInd[i] - dataDict['nCustomer']][2]
    # desireTime是将pick-up和drop-off的最晚时间糅合在一起，按照ind的排序，按照节点排序自动pickup/dropoff时间

timeDelay = [max(timeArrangement[i] - desiredTime[i], 0) for i in range(len(bestInd))]
timeDelay_Seq = [0]*(1+2*dataDict['nCustomer'])
for i in range(len(timeDelay)):
    timeDelay_Seq[bestInd[i]] = timeDelay[i]
print(timeDelay)

print('到达各客户的延迟为：'+str(timeDelay_Seq))
print('到达各个上/下车点时间为：'+str(timeTable(decodeInd(bestInd))))
'''

ArriveCar1 = [9,4,2]

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
        indexo = ArriveCar1[i]
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
print(insert([0,9,5,25,35,1],
      [0, 2, 13, 4, 24, 33, 22, 16, 15, 8, 35, 28, 36, 20, 10, 30, 40, 0, 11, 1, 12, 31, 21, 3, 32, 23, 18, 19, 39, 38, 0, 7, 14, 5, 34, 27, 25, 17, 37, 6, 26, 9, 29],
      dataDict))


