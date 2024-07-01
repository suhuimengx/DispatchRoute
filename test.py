import requests
import time
from points import Convert2GD, RodeArray, GetJsonResult, GetPathArray_str
import threading
import time
from tqdm import tqdm

start_time1 = time.time()
cnt = 0
for i in tqdm(range(1,50), desc='Processing Real-time order'):
    time.sleep(0.01)
print(f"1耗时：{time.time() - start_time1}")

# a = []
# b = []
# def task1():
#     start_time1 = time.time()
#     a.append(1)
#     print("Task 1 started")
#     time.sleep(1)
#     print("Task 1 finished")
#     print(f"1耗时：{time.time() - start_time1}")
#
# def task2():
#     start_time1 = time.time()
#     a.append(2)
#     print("Task 2 started")
#     time.sleep(2)
#     print("Task 2 finished")
#     print(f"2耗时：{time.time() - start_time1}")
#
# def task():
#     start_time = time.time()
#     threads = []
#     t1 = threading.Thread(target=task1, )
#     threads.append(t1)
#     t2 = threading.Thread(target=task2, )
#     threads.append(t2)
#     for t in threads:
#         t.start()
#     for t in threads:
#         t.join()
#     print(f"耗时：{time.time() - start_time}")
#
# task()
# print(a,b)
# for i in range(24):
#     for j in range(i+1, 24):
#         res = GetJsonResult(RodeArray[i], RodeArray[j])
#         steps = res['result']['routes'][0]['steps']
#         tempArray = GetPathArray_str(steps)
#         if len(tempArray) > 38:
#             print(len(tempArray))






