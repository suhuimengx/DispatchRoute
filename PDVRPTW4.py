import random
import sys
import copy
import json
import numpy as np
import matplotlib.pyplot as plt
from deap import base, tools, creator, algorithms
from pprint import pprint
from copy import deepcopy
params = {
    'font.family': 'serif',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 12,
    'legend.fontsize': 'small'
}
plt.rcParams.update(params)
path = 'C:/Users/Ryfan.Li/Desktop/project/graph.json'
nCustomer = 20 #定义乘客数量
nSAV = 6 #定义服务的SAV数量
speed = 8 #车辆速度是每分钟0.5km

def json_traverse(data: dict):
    """
    遍历字典，并返回ndarray格式的元组
    :param data: 要遍历的地图数据
    :return: 图的邻接矩阵
    """
    graph_items = [(item[0], item[1]) for item in data[0].items()]
    point_num = len(graph_items)
    adjacency_matrix = [[float('inf') for j in range(point_num)] for i in range(point_num)]
    for k, v in data[0].items():
        row_index = ord(k) - ord('A')
        adjacency_matrix[row_index][row_index] = 0
        for edge, dis in v.items():
            column_index = ord(edge) - ord('A')
            adjacency_matrix[row_index][column_index] = dis
            adjacency_matrix[column_index][row_index] = dis
        # print(k, v)
    return adjacency_matrix


with open(path, 'r', encoding='utf-8') as f:
    graph = json.load(f)
graph = json_traverse(graph)

def Dijkstra(network, s, d):  # 迪杰斯特拉算法算s-d的最短路径，并返回该路径和值
    path = []  # 用来存储s-d的最短路径
    n = len(network)  # 邻接矩阵维度，即节点个数
    fmax = float('inf')
    w = [[0 for _ in range(n)] for j in range(n)]  # 邻接矩阵转化成维度矩阵，即0→max

    book = [0 for _ in range(n)]  # 是否已经是最小的标记列表
    dis = [fmax for i in range(n)]  # s到其他节点的最小距离
    book[s - 1] = 1  # 节点编号从1开始，列表序号从0开始
    midpath = [-1 for i in range(n)]  # 上一跳列表
    for i in range(n):
      for j in range(n):
        if network[i][j] != 0:
          w[i][j] = network[i][j]  # 0→max
        else:
          w[i][j] = fmax
        if i == s - 1 and network[i][j] != 0:  # 直连的节点最小距离就是network[i][j]
          dis[j] = network[i][j]
    for i in range(n - 1):  # n-1次遍历，除了s节点
      min = fmax
      for j in range(n):
        if book[j] == 0 and dis[j] < min:  # 如果未遍历且距离最小
          min = dis[j]
          u = j
      book[u] = 1
      for v in range(n):  # u直连的节点遍历一遍
        if dis[v] > dis[u] + w[u][v]:
          dis[v] = dis[u] + w[u][v]
          midpath[v] = u + 1  # 上一跳更新
    j = d - 1  # j是序号
    path.append(d)  # 因为存储的是上一跳，所以先加入目的节点d，最后倒置
    while (midpath[j] != -1):
      path.append(midpath[j])
      j = midpath[j] - 1
    path.append(s)
    path.reverse()  # 倒置列表
    #print("path:",path)
    # print(midpath)
    #print("dis:",dis)
    return [path, dis[d-1]]


nNodes = int(len(graph))
DijDis = np.zeros((nNodes,nNodes))
for i in range(nNodes):
    for j in range(nNodes):
        DijDis[i,j] = Dijkstra(graph, i+1, j+1)[1]
for i in range(nNodes):
    DijDis[i,i] = 0
# DijDis矩阵用以存储各个节点之间的最短距离，在后续route节点排序出来后，直接将相邻节点距离求和即可得到该route的总里程

DijRoulist = [[[] for j in range(nNodes)] for i in range(nNodes)]
for i in range(nNodes):
    for j in range(nNodes):
        DijRoulist[i][j] = Dijkstra(graph, i+1, j+1)[0]
# DijRou = np.array(DijRoulist)
# DijRou用来存储各个节点之间最短距离的路线

DijTime = np.zeros((nNodes,nNodes))
for i in range(nNodes):
    for j in range(nNodes):
        DijTime[i,j] = DijDis[i,j]/speed
# DijTime用来存储各个节点之间最短距离行驶的时间
"""
print(DijRou[1,2])
print(DijRou[2,1])
print(DijDis)
print(DijTime)
"""

dataDict = {}
# NodeCoor的格式应为[0]+[上车点的节点序号]+[下车点序号]，定义地图中的0节点是停车场节点，对应编码中车辆从停车场出发，这样编码的数与节点索引就对应起来了
"""
dataDict['NodeCoor'] = [0]*(1+2*nCustomer)
for i in range(1, nCustomer+1):
    upoff = random.sample(range(nNodes),2)
    dataDict['NodeCoor'][i] = upoff[0]
    dataDict['NodeCoor'][i+nCustomer] = upoff[1]
print(dataDict['NodeCoor'])
print(len(dataDict['NodeCoor']))
"""
dataDict['NodeCoor'] = [0, 7, 5, 21, 9, 25, 25, 3, 7, 13, 14, 8, 9, 11, 25, 9, 6, 2, 14, 3, 8,
                        25, 11, 9, 16, 1, 24, 2, 4, 22, 7, 0, 21, 19, 5, 13, 1, 23, 16, 5, 24]

"""
产生nCustomer个需求随机数，排除第一个0，前nCustom个是正数，因为是上车点；后半是负数，因为是下车点，这样就可以直接用索引引用求和判断是否超负载
for _ in range(nCustomer):
    d = random.randint(1,4)
    dataDict['Demand'].append(d)
for de in range(1,nCustomer+1):
    dataDict['Demand'].append(-dataDict['Demand'][de])
print(dataDict['Demand'])
产生的需求结果如下
"""
dataDict['Demand'] = [0, 2, 2, 4, 1, 3, 3, 3, 3, 3, 1, 3, 1, 3, 3, 1, 3, 3, 1, 4, 2,
                      -2, -2, -4, -1, -3, -3, -3, -3, -3, -1, -3, -1, -3, -3, -1, -3, -3, -1, -4, -2]
dataDict['MaxLoad'] = 7 # 运载车最大容量，我们在校园里需求按常规来说为1-4

# 按照VRPTW代码中的时间窗抽取nCustomer个固定最大等待时间为2min的出发时间窗，和最晚到达时间，组成一个乘客的整体时间窗
# ***源代码中这里时间窗用的是括号表示的
dataDict['Timewindow'] = [[0, 0, 0], [3, 5, 13], [2, 4, 12], [9, 11, 19], [14, 16, 24], [20, 22, 30], [11, 13, 21], [6, 8, 16], [13, 15, 23], [7, 9, 17], [13, 15, 23],
                          [11, 13, 21], [4, 6, 14], [18, 20, 28], [3, 5, 13], [17, 19, 27], [9, 11, 19], [20, 22, 30], [19, 21, 29], [7, 9, 17], [12, 14, 22]]
# 用ServiceTime表示在节点上下车的时间
dataDict['ServiceTime'] = 1

# 生成染色体
def genInd(dataDict = dataDict):
    index0 = 0
    ind0 = np.random.permutation(nCustomer) + 1
    ind = ind0.tolist()
    ind.insert(0, 0)  # 先插入第一辆车
    n = nSAV-1 # 生成插入车辆的数量随机数（排除第一辆车）
    SAV_loc = []
    if n != 0:
        i = 1
        while i < n + 1:
            sav = random.randint(2, nCustomer)
            if sav in SAV_loc:
                continue
            else:
                SAV_loc.append(sav)
                i += 1

    SAV_loc.sort()  ## 把位置按从升序排列
    SAV_loc_shift = SAV_loc.copy()
    j = 0
    for loc in SAV_loc_shift:
        SAV_loc_shift[j] += index0
        index0 += 1
        j += 1

    for loc in SAV_loc_shift:
        ind.insert(loc, 0)

    indCopy = ind.copy()  # 复制染色体，防止直接对染色体进行改动
    SAV_loc_shift.insert(0, 0)
    SAV_loc_shift.append(len(indCopy))
    # indcopy是染色体复制 idxlist是染色体长度的一个顺序排列 zeroidx是染色体中0的顺序标号的排列
    routes = []
    for p, q in zip(SAV_loc_shift[0::], SAV_loc_shift[1::]):
        routes.append(ind[p:q])

    ## 开始插入destination节点
    routesCopy = []
    for rou in routes:
        rCopy = rou.copy()
        for c in range(1, len(rou)):
            index1 = rCopy.index(rou[c])
            d = random.randint(index1 + 1, len(rCopy))
            if d == len(rCopy):
                rCopy.append(rou[c] + nCustomer)
            else:
                rCopy.insert(d, rou[c] + nCustomer)
        routesCopy.append(rCopy)

    # 将生成的分段routes合并成最终随机生成的、包含上车点和下车点的染色体
    ind_f = []
    for rou_f in routesCopy:
        for r_f in rou_f:
            ind_f.append(r_f)
    ind_com = []
    ind_com.append(ind)
    ind_com.append(routes)
    ind_com.append(ind_f)
    ind_com.append(routesCopy)
    return ind_f

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
# print(decodeInd([0, 7, 6, 16, 17, 9, 8, 19, 3, 18, 13, 0, 2, 12, 0, 1, 11, 5, 15, 0, 10, 20, 4, 14]))

# 定义负载约束的惩罚函数
def loadPenalty(routes):
    penalty = 0
    for eachRoute in routes:
        routeLoad = 0
        for i in eachRoute:
            routeLoad += dataDict['Demand'][i]
            if routeLoad > dataDict['MaxLoad']:
                penalty += routeLoad - dataDict['MaxLoad'] # 容量超限还是有必要累加的，一步错步步错
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
        arrivalTime += DijTime[dataDict['NodeCoor'][route[i-1]]][dataDict['NodeCoor'][route[i]]]
        if route[i] <= nCustomer:
            arrivalTime = max(arrivalTime, dataDict['Timewindow'][route[i]][0])
        serviceTime[i] = arrivalTime
        arrivalTime += dataDict['ServiceTime']
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
        if ind[i] <= nCustomer:
            desiredTime[i] = dataDict['Timewindow'][ind[i]][1]
        else:
            desiredTime[i] = dataDict['Timewindow'][ind[i]-nCustomer][2]
    # desireTime是将pick-up和drop-off的最晚时间糅合在一起，按照ind的排序，按照节点排序自动pickup/dropoff时间
    timeDelay = [max(timeArrangement[i]-desiredTime[i],0) for i in range(len(ind))]
    return np.sum(timeDelay)

## print(timePanalty([0, 7, 6, 16, 17, 9, 8, 19, 3, 18, 13, 0, 2, 12, 0, 1, 11, 5, 15, 0, 10, 20, 4, 14],
##                  [[0, 7, 6, 16, 17, 9, 8, 19, 3, 18, 13], [0, 2, 12], [0, 1, 11, 5, 15], [0, 10, 20, 4, 14]]))

def calRouteLen(routes,dataDict=dataDict):
    '''辅助函数，返回给定路径的总长度'''
    totalDistance = 0 # 记录各条路线的总长度
    for eachRoute in routes:
        # 从每条路径中抽取相邻两个节点，计算节点距离并进行累加
        for i,j in zip(eachRoute[0::], eachRoute[1::]):
            totalDistance += DijDis[dataDict['NodeCoor'][i], dataDict['NodeCoor'][j]]
    return totalDistance
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

def evaluate(ind, c1=500.0, c2=10.0):
    '''评价函数，返回解码后路径的总长度，c1, c2分别为车辆超载与不能服从给定时间窗口的惩罚系数'''
    routes = decodeInd(ind) # 将个体解码为路线
    totalDistance = calRouteLen(routes)
    return (totalDistance + c1*loadPenalty(routes) + c2*timePenalty(ind,routes)),
#print(evaluate([0, 3, 13, 0, 2, 6, 4, 1, 12, 10, 20, 11, 5, 14, 16, 8, 15, 7, 18, 17, 0, 9, 19]))

def singlevaluate(ind, c1=500.0, c2=10.0):
    routes = decodeInd(ind) # 将个体解码为路线
    totalDistance = calRouteLen(routes)
    return totalDistance + c1*loadPenalty(routes) + c2*timePenalty(ind,routes)

def removeDropoff(ind):
    # 辅助函数，将生成的染色体还原为插入下车点前的染色体
    indPickup = [i for i in ind if i<=nCustomer]
    return indPickup
# print(removeDropoff([0, 7, 6, 16, 17, 9, 8, 19, 3, 18, 13, 0, 2, 12, 0, 1, 11, 5, 15, 0, 10, 20, 4, 14]))

def sequence(ind):
    # 辅助函数，将染色体中多余的0去掉，确定次序
    ind_seq = []
    for n in ind:
        if n not in ind_seq:
            ind_seq.append(n)
    return ind_seq

def genChild(ind1, ind2, nTrail=5): #ind对应ind_com[2]
    indPick1 = removeDropoff(ind1)
    indPick2 = removeDropoff(ind2) #仅有上车点的染色体，对应的是ind_com[0]
    indPickRou1 = decodeInd(indPick1)
    indPickRou2 = decodeInd(indPick2) #将上车点打断为车辆路径，对应的是ind_com[1]
    routes1 = decodeInd(ind1)
    routes2 = decodeInd(ind2) #解码后的单车路径列表,对应的是ind_com[3]
    ind_seq2 = sequence(ind2)
    # 将最终的节点序列除多余的0，以备后续子代新路线按照父代2的顺序排列
    numSubroute1 = len(routes1)  # 父代1的子路径数量
    ranRou1 = np.random.randint(0, numSubroute1)
    subroute = []
    subroute1 = routes1[ranRou1]  # 随机选取一段完整子路径作为子代1的开头
    # 将subroute1中没有出现的乘客按照其在ind2中的顺序排列成一个序列
    unvisited = set(indPick1) - set(indPickRou1[ranRou1])
    unvisitedPerm = [dight for dight in indPick2 if dight in unvisited]
    # 开始对上车点多次重复打断，选取适应度最好的个体
    bestInd = None
    bestFit = np.inf
    if numSubroute1 > 2:
        for _ in range(nTrail):
            # 将排好的上车点序列随机打断为numSubroute1-1条子路径
            breakPos = [0] + random.sample(range(1, len(unvisitedPerm)), numSubroute1 - 2)
            breakPos.append(len(unvisitedPerm))
            breakPos.sort()
            breakSubroute = []
            for i, j in zip(breakPos[0::], breakPos[1::]):
                breakSubroute.append([0] + unvisitedPerm[i:j])
            # 按照ind2中的顺序将下车点插入各breakRoute
            for breakRou in breakSubroute:
                for k in range(1, len(breakRou)):
                    breakRou.append(breakRou[k] + nCustomer)
                rouPerm = [dight for dight in ind_seq2 if dight in breakRou]
                subroute.append(rouPerm)
            subroute.insert(0, subroute1)  # 插入最先抽取的一段子路径即生成完整的交叉后的子代，循环5次，选取适应度最高的子代作为子代
            # 评价生成的子路径
            subInd = combineInd(subroute)
            routesFit = singlevaluate(subInd)
            if routesFit < bestFit:
                bestInd = subInd
                bestFit = routesFit

    return bestInd
#print(genchild([0, 5, 15, 6, 16, 7, 9, 2, 17, 12, 8, 10, 18, 19, 3, 13, 20, 0, 1, 4, 11, 14],
               #[0, 3, 13, 0, 9, 4, 19, 14, 10, 20, 2, 12, 0, 1, 11, 0, 6, 7, 17, 8, 18, 5, 15, 16], nTrail=5))

def crossover(ind1, ind2):
    ind1[:], ind2[:] = genChild(ind1, ind2), genChild(ind2, ind1)
    return ind1, ind2
#print(crossover([0, 5, 15, 6, 16, 7, 9, 2, 17, 12, 8, 10, 18, 19, 3, 13, 20, 0, 1, 4, 11, 14],
#               [0, 3, 13, 0, 9, 4, 19, 14, 10, 20, 2, 12, 0, 1, 11, 0, 6, 7, 17, 8, 18, 5, 15, 16]))
'''
(ind1,ind2)=crossover([0, 5, 15, 0, 6, 16, 0, 7, 9, 2, 17, 12, 8, 10, 18, 19, 3, 13, 20, 0, 1, 4, 11, 14],
               [0, 3, 13, 0, 9, 4, 19, 14, 10, 20, 2, 12, 0, 1, 11, 0, 6, 7, 17, 8, 18, 5, 15, 16])
print(evaluate(ind1),evaluate(ind2))
'''

##突变操作，目标寻找局部最优
def mutate(ind):
    bestInd = ind
    bestFit = singlevaluate(ind)
    for i in range(1, nCustomer):
        index1 = ind.index(i)
        index2 = ind.index(i+nCustomer)
        for j in range(i+1, nCustomer+1):
            ind_opt = ind.copy() # 复制染色体，防止对原染色体进行修改
            index3 = ind.index(j)
            index4 = ind.index(j+nCustomer)
            ind_opt[index1], ind_opt[index3] = ind[index3], ind[index1]
            ind_opt[index2], ind_opt[index4] = ind[index4], ind[index2]
            optFit = singlevaluate(ind_opt)
            if optFit < bestFit:
                bestInd = ind_opt
                bestFit = optFit
    ind = bestInd
    return ind,
#print(mutate([0, 3, 13, 0, 9, 4, 19, 14, 10, 20, 2, 12, 0, 1, 11, 0, 6, 7, 17, 8, 18, 5, 15, 16]))

## 问题定义
def GA_improved(npop = 200,cxpb = 0.7,mutpb = 0.2,ngen = 200):
    creator.create('FitnessMin', base.Fitness, weights=(-1.0,)) #最小化问题
    creator.create('Individual', list, fitness = creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register('individual', tools.initIterate, creator.Individual, genInd)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    toolbox.register('evaluate', evaluate)
    toolbox.register('select', tools.selTournament, tournsize=2)
    toolbox.register('mate', crossover)
    toolbox.register('mutate', mutate)

    ## 生成初始族群
    pop = toolbox.population(npop)

    ## 记录迭代数据
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register('min', np.min)
    stats.register('avg', np.mean)
    stats.register('std', np.std)
    logbook = tools.Logbook()
    logbook.header = 'gen', 'nevals', 'avg', 'std', 'min'

    for gen in range(1, ngen + 1):
        # 配种选择
        offspring = toolbox.select(pop, 2 * npop)
        # 复制，否则在交叉和突变这样的原位操作中，会改变所有select出来的同个体副本
        offspring_copy = list(map(toolbox.clone, offspring))

        # 变异操作-交叉
        for child1, child2 in zip(offspring_copy[::2], offspring_copy[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

    # 变异操作-突变
        for mutant in offspring_copy:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

    # 对于被改变的个体，重新计算其适应度
        invalid_ind = [ind for ind in offspring_copy if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # 环境选择-保留精英,保持种群规模
        pop = tools.selBest(offspring_copy, npop, fit_attr='fitness')

        # 记录数据
        # compile(sequence)# 将每个注册功能应用于输入序列数据，并将结果作为字典返回
        record = stats.compile(pop)
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        bestInd = tools.selBest(pop, k=1)[0]

    return pop, logbook, bestInd

def plotTour(logbookGA_improved):
    gen = logbookGA_improved.select('gen')
    min = logbookGA_improved.select('min')
    avg = logbookGA_improved.select('avg')
    plt.plot(gen,min,'r-',label='Minimum Fitness')
    plt.plot(gen,avg,'b-',label='Average Fitness')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.legend(loc='upper right')
    plt.title('GA iterations fitness convergence',fontsize = 15)
    plt.tight_layout()
    plt.show()

resultPopGA_improved,logbookGA_improved,bestInd = GA_improved(npop = 400,cxpb = 0.7,mutpb = 0.2,ngen = 200)
print(logbookGA_improved)
print('最佳个体为：'+str(bestInd))
print('最佳适应度为：'+str(singlevaluate(bestInd)))
plotTour(logbookGA_improved)


def calLoad(routes):
    loads = []
    for eachRoute in routes:
        routeLoad = 0
        eachload = []
        for i in eachRoute:
            routeLoad += dataDict['Demand'][i]
            eachload.append(routeLoad)
        loads.append(eachload)
    return loads
distributionPlan = decodeInd(bestInd)
print('最佳分派计划为：'+str(distributionPlan))

print('总运输距离为：'+str(singlevaluate(bestInd,c1=0,c2=0)))

print('各辆车上负载为：'+str(calLoad(distributionPlan)))

timeArrangement = timeTable(distributionPlan) # 对给定路线，计算到达每个客户的时间
# 索引给定的最迟到达时间
desiredTime = [0] * len(bestInd)
for i in range(len(bestInd)):
    if bestInd[i] <= nCustomer:
        desiredTime[i] = dataDict['Timewindow'][bestInd[i]][1]
    else:
        desiredTime[i] = dataDict['Timewindow'][bestInd[i] - nCustomer][2]
    # desireTime是将pick-up和drop-off的最晚时间糅合在一起，按照ind的排序，按照节点排序自动pickup/dropoff时间

timeDelay = [max(timeArrangement[i] - desiredTime[i], 0) for i in range(len(bestInd))]
timeDelay_Seq = [0]*(1+2*nCustomer)
for i in range(len(timeDelay)):
    timeDelay_Seq[bestInd[i]] = timeDelay[i]
print(timeDelay)

print('到达各客户的延迟为：'+str(timeDelay_Seq))










