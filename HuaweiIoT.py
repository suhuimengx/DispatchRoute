# coding: utf-8
from huaweicloudsdkcore.auth.credentials import BasicCredentials
from huaweicloudsdkiotda.v5.region.iotda_region import IoTDARegion
from huaweicloudsdkcore.exceptions import exceptions
from huaweicloudsdkiotda.v5 import *
import pprint
import json
from flask import Flask, jsonify
from flask_cors import CORS
import time
from points import RodeArray
from points import GetRealTimeArray


class HuaweiCloudObj:
    def __init__(self, project_id, region_id, endpoint):
        self.ak = "BFWTFWN6TQZMEHI0RB9M"
        self.sk = "N3mzBciTW3GTavoidOa0ADfexIHaBFG6sjvzBizm"
        self.project_id = project_id
        self.region_id = region_id
        self.endpoint = endpoint
        # 创建认证对象
        credentials = BasicCredentials(self.ak, self.sk, self.project_id)
        # 创建客户端
        self.client = IoTDAClient.new_builder() \
            .with_credentials(credentials) \
            .with_region(IoTDARegion.CN_NORTH_4) \
            .build()

    def ShowProduct(self, product_id):
        ''' 展示产品的全部参数
        @product_id: 产品的id(创建产品时自动生成)
        '''
        try:
            request = ShowProductRequest()
            request.product_id = product_id
            # 通过client发送请求, 返回dict类型数据
            return self.client.show_product(request).to_json_object()
        except exceptions.ClientRequestException as e:
            print(e.status_code, e.request_id, e.error_code)
            print(e.error_msg)

    def ShowDevice(self, device_id):
        ''' 显示设备的信息
        @device_id: 查询设备的id
        '''
        try:
            start_time = time.time()
            request = ShowDeviceRequest()
            request.device_id = device_id
            res = self.client.show_device(request).to_json_object()
            end_time = time.time()
            print(f"网络请求花费时间为: {end_time - start_time} s")
            return res
        except exceptions.ClientRequestException as e:
            print(e.status_code, e.request_id, e.error_code)
            print(e.error_msg)

    def ShowDeviceShadow(self, device_id):
        """
        获取设备的设备影子响应体
        @param device_id: 设备id
        @return: 设备影子响应体
        """
        try:
            request = ShowDeviceShadowRequest()
            request.device_id = device_id
            return self.client.show_device_shadow(request).to_json_object()
        except exceptions.ClientRequestException as e:
            print(e.status_code, e.request_id, e.error_code)
            print(e.error_msg)

    def GetDeviceFlag(self, device_id):
        """
        获取小车是否处于停止状态
        @param device_id: 小车的设备id
        @return: 小车的停止标志；0->未停止，1->已停止
        """
        return self.ShowDeviceShadow(device_id)['shadow'][0]['reported']['properties']['car_stop']


    def GetDeviceLocation(self, device_id):
        """
        获取小车的实时位置
        @param device_id: 小车设备的id
        @return: 小车的实时位置，百度api的经纬度形式，如[lng, lat] = [118.213791, 32.213131]
        """
        return self.ShowDeviceShadow(device_id)['shadow'][0]['reported']['properties']['location']

    def ResetStopFlag(self, device_id):
        """
        重置小车的停车状态，标志小车已经可以由停止转为继续运行状态
        @param device_id: 小车设备的id
        @return: 指令执行的结果
        """
        try:
            request = UpdatePropertiesRequest()
            request.device_id = device_id
            request.body = DevicePropertiesRequest(
                services=[{"service_id":"car_01","properties":{"car_stop":0}}]
            )
            return self.client.update_properties(request).to_json_object()
        except exceptions.ClientRequestException as e:
            print(e.status_code, e.request_id, e.error_code)
            print(e.error_msg)

    def SendArrayCommand(self, device_id, service_id, location_array):
        """
        将路径点数组通过命令下发传递给小车节点
        @param device_id: 小车的设备id
        @param service_id: 设备服务的id
        @param location_array: 传递的路径点数组
        @return: 命令下发的执行结果
        """
        try:
            request = CreateCommandRequest()
            request.device_id = device_id
            data = {"location_array": location_array}
            request.body = DeviceCommandRequest(
                command_name="SendArray",
                paras=json.dumps(data),
                service_id=service_id
            )
            return self.client.create_command(request).to_json_object()
        except exceptions.ClientRequestException as e:
            print(e.status_code, e.request_id, e.error_code)
            print(e.error_msg)


if __name__ == "__main__":
    # 项目id
    project_id = "b9a181cca17a4c72874effe2cb82fe0e"
    # 上海一"cn-east-3"；北京四"cn-north-4"；华南广州"cn-south-4"
    region_id = "cn-north-4"  # 服务区域
    # 接入端点
    endpoint = "cd5b167852.iot-mqtts.cn-north-4.myhuaweicloud.com"
    # 初始化华为云IOT对象
    Cloud = HuaweiCloudObj(project_id, region_id, endpoint)

    # 产品id
    product_id = "649a72522a3b1d3de71e9e81"
    # 设备id
    device_id = "649a72522a3b1d3de71e9e81_car01"
    # 服务id
    service_id = "car_01"

    start = RodeArray[7]
    end = RodeArray[1]
    # location_array = [[118.96607,32.11583],[118.965736,32.115793],[118.965402,32.115756]]
    # start_time = time.time()
    res = GetRealTimeArray(start, end, 25)
    print(res)
    # distance, location_array, location_array_amap = [value for value in res.values()]
    # print(len(location_array), len(location_array_amap))
    # print(location_array_amap)
    # print(f"耗时：{time.time() - start_time}")


    # res = Cloud.ResetStopFlag(device_id="649a72522a3b1d3de71e9e81_car02")
    # # print(res, f"花费时间为：{time.time() - start_time}")
    # print(type(res))
    # pprint.pprint(res)

    # res = Cloud.ShowDeviceShadow(device_id)
    # location = res['shadow'][0]['reported']['properties']['location']
    # print(type(location), location)
