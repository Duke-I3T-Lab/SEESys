# SEESys Deployment on Edge Server
To deploy SEESys on an edge server, follow these steps:

Transfer this folder to your edge server.

Run the following commands in two separate terminals:

In terminal1:
```
python3 main_EdgeReceiver.py
```
This script handles the communication between the edge server and the mobile device.

Then, in terminal2:
```
python3 main_DeepSEE.py
```
This script is responsible for loading the DeepSEE model, preprocessing the data, and performing model inference.
