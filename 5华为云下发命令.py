import HuaweiIoT
from HuaweiIoT import HuaweiCloudObj
from time_sys import SystemClock, System_file
from points import RodeArray, GetPathArray, GetRealTimeArray

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
# device_id = "66505a617dbfd46fabbd3225_led001"
# device_id = "66505a617dbfd46fabbd3225_car02"
device_id = "66505a617dbfd46fabbd3225_car03"


# 服务id
# service_id = "car_01"
service_id = "hhhcar1"
FileObj = System_file("./路径距离.xls")

start= 6
end = 8
# 从xls文件读取路网信息
result = FileObj.get_direct_array(start, end)
distance_total, location_array_total_str, location_array_amap_total_str = [value for value in result.values()]
location_array_total = GetPathArray(location_array_total_str.split(';'))

# 初始化华为云IOT对象
Cloud = HuaweiCloudObj(project_id, region_id, endpoint)
res = Cloud.SendArrayCommand(device_id, "hhhcar1", location_array_total)
print(f"3号车任务及行驶信息下发状态：{res}")

# print(Cloud.GetDeviceLocation(device_id))
# print(Cloud.ShowDeviceShadow(device_id))
# print(Cloud.ShowDeviceShadow(device_id)['shadow'][1]['reported']['properties']['location'])
# print(Cloud.ShowDeviceShadow(device_id)['shadow'][0]['reported']['properties']['location'])
