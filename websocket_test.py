from flask import Flask, jsonify#jsonify将python转换为json字符串，并创建flask response对象，可直接返回
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import time, threading

app = Flask(__name__)
# 解决前端跨域问题
CORS(app)
initFlag = False

app.config['SECRET_KEY'] = '123456789'
socketio = SocketIO(app, cors_allowed_origins='*')

# 定义路由处理websocket连接
@app.route('/socket.io/')
def index():
    return jsonify({'name':'zhangsan'})

@socketio.on('connect')
def test_connect():
    global initFlag
    initFlag = True
    print('client connected')

@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected')

@socketio.on('message')
def test_message(data):
    print('Received message: ' + data)

@socketio.on('emit')
def test_emit(data):
    emit('response', data)

def thread_task():
    while True:
        print("okk")
        if initFlag:
            data = {
                'id':1,
                'message':'adad'
            }
            print("okk")
            socketio.emit('send_message', data)
        time.sleep(1.0)

thread = threading.Thread(target=thread_task)

socketio.run(app, port=5000, allow_unsafe_werkzeug=True)
thread.start()





