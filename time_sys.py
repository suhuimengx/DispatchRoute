import time, threading
from datetime import datetime, timedelta
import xlrd

#生成时钟对象，从当前日期的7:20开始，每0.325秒更新一次，定义了当前时间的属性，是相对于7:20开始的，更新频率是interval
class SystemClock:
    def __init__(self, interval):
        self.begin_time = datetime(datetime.today().year,datetime.today().month, datetime.today().day, 7, 20)
        self.current_time = datetime(datetime.today().year,datetime.today().month, datetime.today().day, 7, 20)
        self.update_interval = interval #更新间隔，单位为秒
        self.is_running = False
        self.clock_thread = None

    def start(self):
        if not self.is_running:
            self.is_running = True
            self.clock_thread = threading.Thread(target=self._update_time)
            self.clock_thread.start()

    def stop(self):
        if self.is_running:
            self.is_running = False
            self.clock_thread.join()

    def get_current_time(self):
        real_time = self.current_time.time()
        # 读取时分秒
        return real_time.strftime('%H:%M:%S')

    # 得到当前系统运行的时间，以分钟为单位（目的是与时间窗单位对齐）
    def get_current_ConvertedTime(self):
        return (self.current_time -self.begin_time).seconds // 60

    def _update_time(self):
        while self.is_running:
            time.sleep(self.update_interval)
            self.current_time += timedelta(seconds=5)


class System_file:
    def __init__(self, file_path):
        self.excel = xlrd.open_workbook(file_path)
        self.sheet_list  = []
        for i in range(23):
            self.sheet_list.append(self.excel.sheet_by_index(i))
    def get_process_points(self, start, end):
        inverseFlag = False
        if start > end:
            temp = start
            start = end
            end = temp
            inverseFlag = True
        # 获取工作簿
        sheet = self.sheet_list[start]
        res = sheet.cell_value(1 + end - start, 8)
        if not isinstance(res, str):
            res = str(int(res))
        if res != '直达':
            process_list = [int(x) for x in res.split(',')]
            process_list = [start] + process_list + [end]
            # 若倒序，则翻转列表
            if inverseFlag:
                process_list.reverse()
            return process_list
        else:
            return [start, end]

    def get_direct_array(self, start, end):
        inverseFlag = False
        if start > end:
            temp = start
            start = end
            end = temp
            inverseFlag = True
        # 获取工作簿
        sheet = self.sheet_list[start]
        if inverseFlag:
            return {
                'distance': int(sheet.cell_value(1 + end - start, 6)),
                'location_array': sheet.cell_value(1 + end - start, 11),
                'location_array_amap': sheet.cell_value(1 + end - start, 13)
            }
        else:
            return {
                'distance': int(sheet.cell_value(1 + end - start, 6)),
                'location_array': sheet.cell_value(1 + end - start, 10),
                'location_array_amap': sheet.cell_value(1 + end - start, 12)
            }





# FileObj = System_file("./路径距离.xls")
# start_time = time.time()
# print(FileObj.get_process_points(2, 3))
# print(f"耗时{time.time() - start_time}")

# clock = SystemClock(1)
# clock.start()
# print(clock.get_current_ConvertedTime())
# print(type(clock.get_current_ConvertedTime()))