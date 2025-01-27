import sys

import socket

import random
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

from pickle import load
from scipy.spatial.transform import Rotation as R

# from models import CNN2DGRU
# from models import PatchTSMixer

import time

import concurrent.futures
from multiprocessing import Pool
import multiprocessing

import warnings
warnings.simplefilter(action='ignore')

import cv2

from models.DeepSEEModels import MultiModalCrossAttentionConfig, DeepSEEModel
from transformers import PatchTSMixerConfig
from transformers import TimesformerConfig

from ts2vec.ts2vec import TS2Vec

## Global Features
feature_name = ["Counter", "TimeStamp", "TrackMode", 
                "Brightness", "Contrast", "Entropy", "Laplacian",
                "AvgMPDepth", "VarMPDepth", "PrePOKeyMapLoss", "PostPOOutlier", "MatchedInlier",
                "DX", "DY", "DZ", "Yaw", "Pitch", "Roll", 
                "local_visual_BA_Err"]

class TimeSeriesBuffer():
    def __init__(self, window=30, num_channel=16):
        self.window = window
        self.num_channel = num_channel
        self.buffer = np.zeros([window,num_channel])
        self.valid_length = 0
        self.ready_flag = False

    @staticmethod
    def preprocess(str_data):
        isValid = False
        feature_list = str_data.split(",")[0:22]#str_data.split(",")
        try:
            feature_list = [float(x) for x in feature_list]
            feature_list = np.array(feature_list)
            # "Counter", "TimeStamp", "TrackMode",
            # "Brightness", "Contrast", "Entropy", "Laplacian",
            # "AvgMPDepth", "VarMPDepth", "PrePOKeyMapLoss", "PostPOOutlier", "MatchedInlier",
            # "DX", "DY", "DZ", "Yaw", "Pitch", "Roll", 
            # "local_visual_BA_Err"
            ## !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            feature = feature_list[np.r_[3:19]]
        except:
            print("Invalid features")
            isValid = False
            return None, isValid
        else:
            feature[0] = feature[0]/160.0   # Brightness
            feature[1] = feature[1]/70.0   # Contrast
            feature[2] = feature[2]/8.0   # Entropy
            feature[3] = feature[3]/90.0   # Laplacian
            feature[4] = feature[4]*1.2   # AvgMPDepth
            feature[5] = feature[5]*4.0   # VarMPDepth
            feature[6] = feature[6]/600.0   # PrePOKeyMapLoss
            feature[7] = feature[7]/100.0   # PostPOOutlier
            feature[8] = feature[8]/400.0   # MatchedInlier
            feature[9] = feature[9]*100.0   # "DX"
            feature[10] = feature[10]*100.0   # "DY"
            feature[11] = feature[11]*100.0   # "DZ"
            feature[12] = np.sin(feature[12])   # "Yaw"
            feature[13] = np.sin(feature[13])   # "Pitch"
            feature[14] = np.sin(feature[14])   # "Roll"
            feature[15] = np.log1p(feature[15])/10.0   # "local_visual_BA_Err"
            
            isValid = feature_list[2] == 2.0
            feature = np.array(feature).reshape([1,16])
            return feature, isValid
    
    def push(self, str_data):
        feature, isValid = TimeSeriesBuffer.preprocess(str_data)
        if isValid == True:
            assert feature.shape == (1,16)
            self.buffer = np.roll(self.buffer, -1, axis=0)
            self.buffer[-1] = feature
            if self.valid_length < self.window:
                self.valid_length += 1
        else:
            self.valid_length = 0

    def sample(self):
        if self.valid_length < self.window:
            self.ready_flag = False
            return None, self.ready_flag
        else:
            self.ready_flag = True
            return np.expand_dims(self.buffer, axis=0), self.ready_flag
            # shape (1, 30, 16)
        
    def reset(self):
        self.__init__()


class PointDistBuffer():
    def __init__(self, frames=30, width=128, height=96, channel=3):
        self.frames = frames
        self.width = width
        self.height = height
        self.channel=channel
        self.buffer = np.zeros([frames,channel,height,width], dtype=np.uint8)
        # target shape [30,3,96,128]
        self.valid_length = 0
        self.ready_flag = False

    @staticmethod
    def preprocess(raw_byte_data):
        # raw_byte_data is from socket

        isValid = False
        try:
            # Deserialize image data
            nparr = np.frombuffer(raw_byte_data, np.uint8)
            pointDist = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            # pointDist.shape == (96, 128, 3)
        except:
            print("Invalid features")
            isValid = False
            return None, isValid
        else:
            isValid = True
            pointDist = pointDist.transpose(2,0,1) # (96, 128, 3) --> (3, 96, 128)
            pointDist = np.expand_dims(pointDist, axis=0) # (3, 96, 128) --> (1, 3, 96, 128)
            return pointDist, isValid
    
    def push(self, raw_byte_data):
        pointDist, isValid = PointDistBuffer.preprocess(raw_byte_data)
        if isValid == True:
            assert pointDist.shape == (1, self.channel,self.height, self.width)
            self.buffer = np.roll(self.buffer, -1, axis=0)
            self.buffer[-1] = pointDist
            if self.valid_length < self.frames:
                self.valid_length += 1
        else:
            self.valid_length = 0

    def sample(self):
        if self.valid_length < self.frames:
            self.ready_flag = False
            return None, self.ready_flag
        else:
            self.ready_flag = True
            samples = [self.buffer[0], self.buffer[9], self.buffer[19], self.buffer[29]]
            samples =  np.concatenate(samples, axis=0).reshape(4,self.channel,self.height,self.width)
            samples = np.expand_dims(samples, axis=0)
            # samples.shape == (1, 4, 3, 96, 128)
            return samples, self.ready_flag
        
    def reset(self):
        self.__init__()


def Udp2TimeSeriesBuffer_Process(receive_lock, receive_share_data):

    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # Make sure the socket can be reused
    udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    udp_socket.bind(('127.0.0.1', 11111))
    print("Udp2TimeSeriesBuffer process started")

    while(True):
        update_time = None
        timeSeriesBuffer = TimeSeriesBuffer(window=30, num_channel=16)

        SendTimeStamp = None
        
        while True:
            # Keep running forever 
            # Receive from UDP
            recv_byte = udp_socket.recvfrom(1024)[0] # read data
            if recv_byte != None and len(recv_byte) > 10:
                if update_time is not None:
                    if time.time() - update_time >= 5:
                        print("Current Session End")
                        break
                
                update_time = time.time()

                # Preprocess and buffer
                recv_str = recv_byte.decode('utf-8')
                print("Receive Data: {}".format(recv_str))
                
                timeSeriesBuffer.push(recv_str)
                try:
                    SendTimeStamp = float(recv_str.split(",")[-2])
                except Exception as e: 
                    print("Get SendTimeStamp Failed")
                    print(e)
                    SendTimeStamp = None

                with receive_lock:
                    receive_share_data['HasNewTimeSeriesData'] = True
                    receive_share_data['TimeSeriesBuffer'] = timeSeriesBuffer
                    receive_share_data['SendTimeStamp'] = SendTimeStamp
            time.sleep(0.015)
    udp_socket.close()
    print("udp2buffer process stopped")


def Udp2PointDistBuffer_Process(receive_lock, receive_share_data):
    ## Settings
    # Declare a local udp socket
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    #udp_socket.bind(('127.0.0.1', 65535))
    #udp_socket.bind(('192.168.1.46', 65535))
    #udp_socket.bind(('192.168.0.103', 65535))  # Edge Server IP
    udp_socket.bind(('192.168.1.21', 65535))
    
    while(True):
        update_time = None
        pointDistBuffer = PointDistBuffer(frames=30, width=128, height=96, channel=3)
        print("Udp2PointDistBuffer process started")
        while True:
            # Keep running forever 
            data, addr = udp_socket.recvfrom(65535)  # Buffer size is 65535 bytes

            if data != None and len(data) > 10:
                #print("Receive Data: {}".format(data))
                if update_time is not None:
                    if time.time() - update_time >= 5:
                        print("Current Session End")
                        break
                update_time = time.time()
                # Preprocess and buffer
                with receive_lock:
                    pointDistBuffer.push(data)
                    receive_share_data['HasNewPointDistData'] = True
                    receive_share_data['PointDistBuffer'] = pointDistBuffer

            time.sleep(0.015)
    udp_socket.close()


def VisualizePointDist_Process(receive_lock, receive_share_data):
    while(True):
        ready_flag = False
        with receive_lock:
            if receive_share_data['PointDistBuffer'] is not None:
                pointDistSamples, ready_flag = receive_share_data['PointDistBuffer'].sample()
        if ready_flag:
            # (1, 4, 3, 96, 128) --> (96,128,3)
            img = pointDistSamples[0][-1].transpose(1,2,0)
            cv2.imshow('Received Image', img)
            cv2.waitKey(1)
        time.sleep(0.015)


def load_DeepSEE_model(device='cuda:0'):
    timeSeriesEmbeddedLayer = TS2Vec(input_dims=16,
                            device=0,
                            batch_size=1,
                            output_dims=64
                            )
    
    config_PD = TimesformerConfig(image_size = 128,
                             patch_size = 8,
                             num_channels = 3,
                             num_frames = 4,
                             num_hidden_layers = 3,
                             num_attention_heads = 12,
                             hidden_size = 192,
                             intermediate_size = 256,
                             #hidden_dropout_prob = 0.1)
                             hidden_dropout_prob = 0)
    
    config_TS = PatchTSMixerConfig(context_length = 30, 
                                prediction_length = 1,
                                num_input_channels = 64, 
                                d_model = 192,
                                patch_len = 10,
                                patch_stride = 5,
                                use_positional_encoding = True,
                                ca_d_model = 128,
                                num_layers = 3,
                                drop_out = 0.0
                            )
    
    config_CA = MultiModalCrossAttentionConfig(
                                        ts2vec_only = True,
                                        ts2vec_dim = 64,
                                        # Time Series
                                        ts_context_length = 30,
                                        ts_patch_len = 5,
                                        ts_num_input_channels = 64,
                                        ts_patch_stride = 5,
                                        ts_d_model = 192,
                                        ts_time_step = 33,
                                        # PD parameter
                                        pd_d_model = 192,
                                        pd_time_step = 33*10,
                                        pe_max_len = 10000, 
                                        # CA
                                        ca_d_model = 128, 
                                        ca_num_head = 16,
                                        ca_num_layers = 2,
                                        ca_dropout = 0, # !!!!!!!!!
                                        ca_time_series_only=False, # !!!!!!!!!!!!!!!!!!!!!,
                                        output_range=None #[0,1]None
                                    )
    
    deepSEEModel = DeepSEEModel(config_PD, config_TS, config_CA)

    state_dict = torch.load("./saved_model/pretrained_model.pkl")
    timeSeriesEmbeddedLayer.net.load_state_dict(state_dict)

    checkpoint = torch.load("./saved_model/SupervisedFinetune_best_model.pth")
    #deepSEEModel.load_state_dict(checkpoint['model_state_dict'])
    deepSEEModel.load_state_dict(checkpoint)
    
    deepSEEModel.to(device).eval()
    #deepSEEModel = torch.compile(deepSEEModel, mode="reduce-overhead")

    print("Successfully loaded and compiled DeepSEE models")
    return timeSeriesEmbeddedLayer, deepSEEModel

    
def Buffer2DeepSEE_Process(receive_lock, receive_share_data,
                      label_lock, label_share_data):
    # check GPU availability               
    print("Check CUDA availability")
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print("Run on {}...".format(device))
    
    timeSeriesEmbeddedLayer, deepSEEModel = load_DeepSEE_model(device)

    # Dummy inputs for activating the models
    x_ts = np.random.rand(1, 30, 16)
    x_pd = torch.rand((1, 4, 3, 96, 128)).to(device)
    x_ts = torch.tensor(timeSeriesEmbeddedLayer.encode(x_ts, causal=True, sliding_length=1, sliding_padding=5)).to(device)
    
    y = deepSEEModel.forward(x_pd, x_ts)

    print("Model Ready")
    counter = 0
    start_time = time.time()
    while(counter < 100):
        x_ts = np.random.rand(1, 30, 16)
        x_pd = torch.rand((1, 4, 3, 96, 128)).to(device)
        x_ts = torch.tensor(timeSeriesEmbeddedLayer.encode(x_ts)).to(device)
        y = deepSEEModel.forward(x_pd, x_ts)
        counter += 1
    print("Elapsed time: {}".format(time.time()-start_time))
    # average @ 1000 minibatch of size = 1: 0.00467 second --> 4.67 ms/round

    while True:

        hasNewPointDistData = False
        hasNewTimeSeriesData = False

        sendTimeStamp = None
        with receive_lock:
            hasNewPointDistData = receive_share_data['HasNewPointDistData']
            hasNewTimeSeriesData = receive_share_data['HasNewTimeSeriesData']

        if hasNewPointDistData and hasNewPointDistData:
            with receive_lock:
                pointDistBuffer = receive_share_data['PointDistBuffer']
                timeSeriesBuffer = receive_share_data['TimeSeriesBuffer']
                sendTimeStamp = receive_share_data['SendTimeStamp']
                receive_share_data['HasNewPointDistData'] = False
                receive_share_data['HasNewTimeSeriesData'] = False
            
            if (pointDistBuffer is not None) and (timeSeriesBuffer is not None):
                x_pd_np, pd_valid_flag = pointDistBuffer.sample()
                x_ts_np, ts_valid_flag = timeSeriesBuffer.sample()

                if (pd_valid_flag and ts_valid_flag):
                    #print(x_ts_np)
                    # print(x_ts_np.shape)

                    x_ts = torch.tensor(timeSeriesEmbeddedLayer.encode(x_ts_np)).to(device, 
                                                                                    dtype=torch.float)
                    x_pd = torch.tensor(x_pd_np).to(device, dtype=torch.float)
                    y = deepSEEModel.forward(x_pd, x_ts).detach().cpu().numpy()[0][0]

                    estimatedPRE = (np.exp(y)-1)/10000

                    #print("Estimated Pose Error {:.5f}".format(y))
                    with label_lock:
                        label_share_data['EstimatedRPE'] = estimatedPRE
                        label_share_data['SendTimeStamp'] = sendTimeStamp

        time.sleep(0.1)

def DeepSEE2UDP_Process(label_lock, label_share_data):
    # Target device (that runs Nvidia Jetson)'s ip and port
    #target_ip, taret_port = "192.168.0.102", 5005
    target_ip, taret_port = "192.168.0.25", 5005

    # Create a UDP socket
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    while(True):       
        message = str.encode(str(np.nan) + ',' + str(np.nan)) #str.encode(str(random.random()*0.06))
        with label_lock:
            if label_share_data['EstimatedRPE'] is not None:
                message = str.encode(str(label_share_data['EstimatedRPE']) + ',' + str(label_share_data['SendTimeStamp']))
        print(message)
        udp_socket.sendto(message, (target_ip, taret_port)) # Mobile Device IP
      
        time.sleep(0.03)



def Timer_Process():
    while True:
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        print(current_time)
        time.sleep(1)
      
        

if __name__ == "__main__":
    
    manager = multiprocessing.Manager()
    print("Initialize manager")

    receive_lock = manager.Lock()
    receive_share_data = manager.dict()
    receive_share_data['HasNewTimeSeriesData'] = False
    receive_share_data['TimeSeriesBuffer'] = None
    receive_share_data['HasNewPointDistData'] = False
    receive_share_data['PointDistBuffer'] = None
    receive_share_data['SendTimeStamp'] = None

    label_lock = manager.Lock()
    label_share_data = manager.dict()
    label_share_data['EstimatedRPE'] = None

    udp2TimeSeriesBuffer_process = multiprocessing.Process(
        target=Udp2TimeSeriesBuffer_Process,
        args=(receive_lock, receive_share_data)
    )

    udp2PointDistBuffer_process = multiprocessing.Process(
        target=Udp2PointDistBuffer_Process,
        args=(receive_lock, receive_share_data)
    )
    
    Buffer2DeepSEE_process =  multiprocessing.Process(
        target=Buffer2DeepSEE_Process,
        args=(receive_lock, receive_share_data,label_lock, label_share_data)
    )
    
    visualizePointDist_process = multiprocessing.Process(
        target=VisualizePointDist_Process,
        args=(receive_lock, receive_share_data)
    )
    

    DeepSEE2UDP_process = multiprocessing.Process(target=DeepSEE2UDP_Process,
                                                  args=(label_lock, label_share_data))

    timer_process = multiprocessing.Process(target=Timer_Process)
    
    # Activate all the processes
    udp2TimeSeriesBuffer_process.start()
    udp2PointDistBuffer_process.start()
    Buffer2DeepSEE_process.start()
    #visualizePointDist_process.start()
    DeepSEE2UDP_process.start()
    timer_process.start()


    # Wait for all the processes to end
    udp2TimeSeriesBuffer_process.join()
    udp2PointDistBuffer_process.join()
    Buffer2DeepSEE_process.join()
    #visualizePointDist_process.join()

    DeepSEE2UDP_process.join()
    timer_process.join()

