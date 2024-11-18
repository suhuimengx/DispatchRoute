import requests
import time
lantitude_const = 32.114475
longitude_const = 118.960298
car_ID = 2

#发送报文同步UniCloud的小车信息
def post_carinfo(car_id,latitude,longitude):
    '''
    发送报文更新UniCloud的小车位置信息
    @param:car_id:小车id
    @param:latitude:纬度
    @param：longitude：经度
    '''
    url = "https://fc-mp-6a266bfc-120f-42dc-9c71-1e9d6f643dfa.next.bspapp.com/updateMarkers"
    data = {
        "marker_car": 
            {
                "car_id":car_id,
                "latitude": latitude,
                "longitude": longitude
            }
        }
    try:
        # 发送POST请求
        response = requests.post(url, json=data)
        
        #print(response.json())
    
    except requests.RequestException as e:
        print(f"An error occurred: {e}")

def post_carid(orderid,car_id,server_id):
    '''
    发送报文同步订单分配的小车信息
    @param:orderid:订单id
    @param:car_id:小车id
    '''
    url = "https://fc-mp-6a266bfc-120f-42dc-9c71-1e9d6f643dfa.next.bspapp.com/updateCarId"
    data = {
        "info": 
            {
                "car_id":car_id,
                "order_id":orderid,
                "server_id":server_id
            }
        }
    try:
        # 发送POST请求
        response = requests.post(url, json=data)
        
        #print(response.json())
    
    except requests.RequestException as e:
        print(f"An error occurred: {e}")

def post_serverid(car_id,server_id):
    '''
    发送报文更新UniCloud的小车服务对象
    @param:car_id:小车id
    @param:server_id:小车本段路径的服务对象（数组）
    '''
    url = "https://fc-mp-6a266bfc-120f-42dc-9c71-1e9d6f643dfa.next.bspapp.com/updateServerId"
    data = {
        "info": 
            {
                "car_id":car_id,
                "server_id":server_id
            }
        }
    try:
        # 发送POST请求
        response = requests.post(url, json=data)
        
        #print(response.json())
    
    except requests.RequestException as e:
        print(f"An error occurred: {e}")   

def post_updateCSL(nCustomer):
    '''
    更新接单小车服务对象
    '''
    url = 'https://fc-mp-6a266bfc-120f-42dc-9c71-1e9d6f643dfa.next.bspapp.com/updateCSL'
    data = {
        "info":
        {
            "nCustomer":nCustomer
        }
    }
    try:
        # 发送POST请求
        response = requests.post(url, json=data)
        
        #print(response.json())
    
    except requests.RequestException as e:
        print(f"An error occurred: {e}")