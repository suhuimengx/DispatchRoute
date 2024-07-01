from huaweicloudsdkcore.exceptions import exceptions
from huaweicloudsdkcore.region.region import Region
from huaweicloudsdkiotda.v5 import *
from huaweicloudsdkcore.auth.credentials import BasicCredentials
from huaweicloudsdkcore.auth.credentials import DerivedCredentials
import json


if __name__ == "__main__":
    # 认证用的ak和sk直接写到代码中有很大的安全风险，建议在配置文件或者环境变量中密文存放，使用时解密，确保安全；
    #本示例以ak和sk保存在环境变量中为例，运行本示例前请先在本地环境中设置环境变量HUAWEICLOUD_SDK_AK和HUAWEICLOUD_SDK_SK。
    ak = "TO9AGL3KUTMENTE520KD"
    sk = "uMt7dxooFDsvQkHm9le1CR9CM9pElX7s2tP1mOvL"
    project_id = "eb375db7c7674f0e99719592224806ee"
    # region_id：如果是上海一，请填写"cn-east-3"；如果是北京四，请填写"cn-north-4"；如果是华南广州，请填写"cn-south-1"
    region_id = "cn-north-4"
    # endpoint：请在控制台的"总览"界面的"平台接入地址"中查看"应用侧"的https接入地址
    # endpoint = "4c3dd8f578.st1.iotda-device.cn-north-4.myhuaweicloud.com"
    endpoint = "4c3dd8f578.st1.iotda-app.cn-north-4.myhuaweicloud.com"
    # device_id= "66505a617dbfd46fabbd3225_led001"
    device_id = "66505a617dbfd46fabbd3225_car03"

    # 标准版/企业版：需自行创建Region对象
    REGION = Region(region_id, endpoint)

    # 创建认证
    # 创建BasicCredentials实例并初始化
    credentials = BasicCredentials(ak, sk, project_id)
    
    # 标准版/企业版需要使用衍生算法，基础版请删除配置"with_derived_predicate"
    credentials.with_derived_predicate(DerivedCredentials.get_default_derived_predicate())
    
    # 基础版：请选择IoTDAClient中的Region对象 如： .with_region(IoTDARegion.CN_NORTH_4)
    # 标准版/企业版：需要使用自行创建的Region对象
    client = IoTDAClient.new_builder() \
        .with_credentials(credentials) \
        .with_region(REGION) \
        .build()

    try:
        request = ShowDeviceShadowRequest()
        request.device_id = device_id
        # print(request)
        response = client.show_device_shadow(request).to_json_object()
        print(response)
    except exceptions.ClientRequestException as e:
        print(e.status_code)
        print(e.request_id)
        print(e.error_code)
        print(e.error_msg)
        # print(f"Error Response: {e.error_response}")