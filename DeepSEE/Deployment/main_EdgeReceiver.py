import sys

import socket

import random
import numpy as np

from pickle import load

import time
import datetime

import concurrent.futures
from multiprocessing import Pool
import multiprocessing

import warnings
warnings.simplefilter(action='ignore')

import csv

def DeepSEE_IO_Process(receive_DeepSEE_lock, receive_DeepSEE_data):
    udp_receiver_address = '127.0.0.1'
    udp_receiver_port = 11100
    
    while(True):
        udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Make sure the socket can be reused
        udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        udp_socket.bind((udp_receiver_address, udp_receiver_port))
        print("DeepSEE_IO_Process started")
        updated_time = time.time()

        while(True):
            # Receive from UDP
            recv_str = None
            estimatedRPE = None
            recv_byte = udp_socket.recvfrom(1024)[0] # read data
            
            if recv_byte != None and len(recv_byte) > 1:
                recv_str = recv_byte.decode('utf-8')
                #print("DeepSEE_IO_Process receive: {}".format(recv_str))
                if recv_str != 'Nan':
                    estimatedRPE = float(recv_str)
                    
            with receive_DeepSEE_lock:
                receive_DeepSEE_data["EstimatedRPE"] = estimatedRPE
                #receive_DeepSEE_data["HasNewEstimatedRPE"] = True

            if time.time() - updated_time > 5.0:
                # reset the session
                break

            time.sleep(0.015)


def get_datetime():
    # Get the current date and time
    current_datetime = datetime.datetime.now()

    # Format the datetime as a string
    formatted_datetime = current_datetime.strftime("%Y-%m-%d-%H-%M-%S")
    return formatted_datetime

def SLAM_IO_Process(receive_SLAM_lock, receive_SLAM_data, 
                   transmit_SLAM_lock, transmit_SLAM_data):
    # Configure the server IP and port
    #server_ip = "127.0.0.1"  # Listen on all available network interfaces
    #server_ip = "192.168.0.103"	# Edge Server IP
    server_ip = "192.168.1.21"	# Edge Server IP
    
    server_port = 10000
    
    # Target is the DeepSEE process
    target_host, target_port = "127.0.0.1", 11111

    while(True):

        print("SLAM_IO_Process started")

        # Create a TCP socket for receiving time series 
        tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        tcp_socket.bind((server_ip, server_port))


        # Create a UDP socket for transmitting time series
        udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        try:
            tcp_socket.listen(5)
            print(f"Listening on {server_ip}:{server_port}")
            print("Waiting for a connection...")
        except:
            break

        client_socket, client_address = tcp_socket.accept()
        print(f"Connection from {client_address}")

        default_action_str = "0.9,0,30"
        action_updated_time = time.time()
        counter = 0

        while(True):
            # Receive time series from DeepSEE-SLAM
            recv_byte = client_socket.recv(1024)
            current_time = time.time()
            if recv_byte is not None and len(recv_byte) > 10:

                # Forward the receivcd time series to the DeepSEE model
                udp_socket.sendto(recv_byte, (target_host, target_port))

                # Decode received bytes to strings
                recv_str = recv_byte.decode('utf-8')
                #if counter % 30 == 0:
                print("receive:{}".format(recv_str))
                print(" ")

                with receive_SLAM_lock:
                    receive_SLAM_data['HasNewData'] = True
                    receive_SLAM_data['HasNewRLData'] = True
                    receive_SLAM_data['RawStringInfo'] = recv_str

                SLAM_updated_time = current_time
                if(time.time()-SLAM_updated_time >= 5.0):
                    print("Current Connection End")
                    break

                counter += 1
            
            time.sleep(0.001)

        client_socket.close()
        tcp_socket.close()
        is_connected = False

        print("Current Trajectory End!")     

def Timer_Process():
    while True:
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        print(current_time)
        time.sleep(1)
      
        

if __name__ == "__main__":
    
    manager = multiprocessing.Manager()
    print("Initialize manager")

    receive_SLAM_lock = manager.Lock()
    receive_SLAM_data = manager.dict()
    receive_SLAM_data['HasNewData'] = False
    receive_SLAM_data['HasNewRLData'] = False
    receive_SLAM_data['RawStringInfo'] = None

    transmit_SLAM_lock = manager.Lock()
    transmit_SLAM_data = manager.dict()
    transmit_SLAM_data['HasNewData'] = False
    transmit_SLAM_data['ActionString'] = None

    receive_DeepSEE_lock = manager.Lock()
    receive_DeepSEE_data = manager.dict()
    receive_DeepSEE_data['EstimatedRPE'] = None
    
    SLAM_IO_process = multiprocessing.Process(target=SLAM_IO_Process, 
                                               args=(receive_SLAM_lock, receive_SLAM_data,
                                                     transmit_SLAM_lock, transmit_SLAM_data))

    DeepSEE_IO_process = multiprocessing.Process(target=DeepSEE_IO_Process,
                                                 args=(receive_DeepSEE_lock, receive_DeepSEE_data))
    
    timer_process = multiprocessing.Process(target=Timer_Process)
    
    # Activate all the processes
    SLAM_IO_process.start()
    DeepSEE_IO_process.start()
    timer_process.start()

    # Wait for all the processes to end
    SLAM_IO_process.join()
    DeepSEE_IO_process.join()
    timer_process.join()

