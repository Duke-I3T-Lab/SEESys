import os
import numpy as np
import pandas as pd
import copy
import tools

class SenseTime:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.benchmark = "SenseTime"
        self.scriptTemplate = "{}/Examples/Monocular/mono_tum_vi {}/Vocabulary/ORBvoc.txt {}/Examples/Monocular/SenseTime.yaml {}/datasets/SenseTime/{}/camera/images {}/Examples/Monocular/SenseTime_TimeStamps/{}.txt"
        self.trajectories = ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", \
                              "B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7"]
        
        self.scriptList = []

    def generate_script(self):
        self.scriptList = []
        for i, trajectory in enumerate(self.trajectories):
            scriptDict = {}
            scriptDict["benchmark"] = self.benchmark
            scriptDict["trajectory"] = trajectory
            scriptTemp = copy.deepcopy(self.scriptTemplate)
            scriptTemp = scriptTemp.format(self.root_dir, self.root_dir, self.root_dir, \
                                           self.root_dir, trajectory, \
                                           self.root_dir, trajectory)
            scriptDict["script"] = scriptTemp
            self.scriptList.append(scriptDict)

    def get_script(self):
        return self.scriptList
    
    def copy_ground_truth_traj(self, trajectory):
        gt_csv_path = "{}/datasets/SenseTime/{}/groundtruth/data.csv"
        gt_csv_path = gt_csv_path.format(self.root_dir, trajectory)
        df = pd.read_csv(gt_csv_path)
        columns = ["#t[s:double]","p.x[m:double]", "p.y[m:double]", "p.z[m:double]", \
                "q.x[double]", "q.y[double]", "q.z[double]", "q.w[double]"]
        df = df[columns]
        df.to_csv("{}/logs/ground_truth.txt".format(self.root_dir), sep=' ', index=False, header=False)

    def copy_raw_SLAM_data(self, benchmark,trajectory,trial):
        raw_data_path = "{}/logs/log.csv"
        raw_data_path = raw_data_path.format(self.root_dir)
        destination_folder_path = "{}/data/raw".format(self.root_dir)
        new_filename = "raw_{}_{}_{}.csv".format(benchmark,trajectory,trial)
        tools.copy_file_and_rename(raw_data_path, destination_folder_path, new_filename)

    def load_raw_SLAM_data(self, benchmark, trajectory, trial,
                           raw_data_path = "{}/datasets/Raw_SLAM_Data/{}/raw/raw_{}_{}_{}.csv"):
        data_path = raw_data_path.format(self.root_dir, benchmark, benchmark, trajectory, trial)
        destination_folder_path = "{}/logs".format(self.root_dir)
        new_filename = "log.csv"
        tools.copy_file_and_rename(data_path, destination_folder_path, new_filename)

    def load_n_save_estimated_traj(self):
        # Step 0: Read the CSV file
        df = pd.read_csv("{}/logs/log.csv".format(self.root_dir))
        
        for index, value in enumerate(df['TimeStamp'][1:-1], start=1):
            if(value == df.at[index+1, 'TimeStamp'] or value == 0 or value < df.at[index-1, 'TimeStamp']):
                df.at[index, 'TimeStamp'] = 0.5*(df.at[index-1, 'TimeStamp'] + df.at[index+1, 'TimeStamp'])
        #assert(df["TimeStamp"].is_monotonic_increasing)
        #assert(df["TimeStamp"].is_unique)

        # Step 2: Drop the rows of initialization
        target = 2
        initial_rows = (df["TrackMode"] == target).idxmax()
        initial_rows += 2
        df = df.iloc[initial_rows:].reset_index(drop=True)

        # Step 3: Select several columns by their names
        df_traj = copy.deepcopy(df)
        selected_columns = ["TimeStamp","PX","PY","PZ","QX","QY","QZ","QW"]  # Replace with your desired column names
        df_traj = df_traj[selected_columns]

        # Step 4: Save the DataFrame as a text (TXT) file
        df_traj.to_csv("{}/logs/trajectory.txt".format(self.root_dir), \
                       sep=' ', index=False, header=False)  
        # You can change the separator as needed
        return df, df_traj, initial_rows

    
class LivingRoom:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.benchmark = "LivingRoom"
        self.scriptTemplate = "{}/Examples/Monocular/mono_tum_vi {}/Vocabulary/ORBvoc.txt {}/datasets/LivingRoom/config.yaml {}/datasets/LivingRoom/{}/mav0/cam0/data {}/datasets/LivingRoom/{}/{}.txt"
        self.trajectories = ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7"]
        self.scriptList = []

    def generate_script(self):
        self.scriptList = []
        for i, trajectory in enumerate(self.trajectories):
            scriptDict = {}
            scriptDict["benchmark"] = self.benchmark
            scriptDict["trajectory"] = trajectory
            scriptTemp = copy.deepcopy(self.scriptTemplate)
            scriptTemp = scriptTemp.format(self.root_dir, self.root_dir, self.root_dir, \
                                           self.root_dir, trajectory, \
                                           self.root_dir, trajectory, trajectory)
            scriptDict["script"] = scriptTemp
            self.scriptList.append(scriptDict)

    def get_script(self):
        return self.scriptList
    
    def copy_ground_truth_traj(self, trajectory):
        gt_csv_path = "{}/datasets/LivingRoom/{}/groundtruth/data.csv"
        gt_csv_path = gt_csv_path.format(self.root_dir, trajectory)
        df = pd.read_csv(gt_csv_path)
        columns = ["#t[s:double]","p.x[m:double]", "p.y[m:double]", "p.z[m:double]", \
                "q.x[double]", "q.y[double]", "q.z[double]", "q.w[double]"]
        df = df[columns]
        df.loc[:,"#t[s:double]"] = df.loc[:,"#t[s:double]"] / 10**9
        df.to_csv("{}/logs/ground_truth.txt".format(self.root_dir), sep=' ', index=False, header=False)
    
    def copy_raw_SLAM_data(self, benchmark,trajectory,trial):
        raw_data_path = "{}/logs/log.csv"
        raw_data_path = raw_data_path.format(self.root_dir)
        destination_folder_path = "{}/data/raw".format(self.root_dir)
        new_filename = "raw_{}_{}_{}.csv".format(benchmark,trajectory,trial)
        tools.copy_file_and_rename(raw_data_path, destination_folder_path, new_filename)

    def load_raw_SLAM_data(self, benchmark, trajectory, trial,
                           raw_data_path = "{}/datasets/Raw_SLAM_Data/{}/raw/raw_{}_{}_{}.csv"):
        data_path = raw_data_path.format(self.root_dir, benchmark, benchmark, trajectory, trial)
        destination_folder_path = "{}/logs".format(self.root_dir)
        new_filename = "log.csv"
        tools.copy_file_and_rename(data_path, destination_folder_path, new_filename)

    def load_n_save_estimated_traj(self):
        # Step 0: Read the CSV file
        df = pd.read_csv("{}/logs/log.csv".format(self.root_dir))
        
        for index, value in enumerate(df['TimeStamp'][1:-1], start=1):
            if(value == df.at[index+1, 'TimeStamp'] or value == 0 or value < df.at[index-1, 'TimeStamp']):
                df.at[index, 'TimeStamp'] = 0.5*(df.at[index-1, 'TimeStamp'] + df.at[index+1, 'TimeStamp'])
        #assert(df["TimeStamp"].is_monotonic_increasing)
        #assert(df["TimeStamp"].is_unique)

        # Step 2: Drop the rows of initialization
        target = 2
        initial_rows = (df["TrackMode"] == target).idxmax()
        initial_rows += 2
        df = df.iloc[initial_rows:].reset_index(drop=True)

        # Step 3: Select several columns by their names
        df_traj = copy.deepcopy(df)
        selected_columns = ["TimeStamp","PX","PY","PZ","QX","QY","QZ","QW"]  # Replace with your desired column names
        df_traj = df_traj[selected_columns]

        # Step 4: Save the DataFrame as a text (TXT) file
        df_traj.to_csv("{}/logs/trajectory.txt".format(self.root_dir), \
                       sep=' ', index=False, header=False)  
        # You can change the separator as needed
        return df, df_traj, initial_rows


class Hall:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.benchmark = "Hall"
        self.scriptTemplate = "{}/Examples/Monocular/mono_tum_vi {}/Vocabulary/ORBvoc.txt {}/datasets/Hall/config.yaml {}/datasets/Hall/{}/mav0/cam0/data {}/datasets/Hall/{}/{}.txt"
        self.trajectories = ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7"]
        self.scriptList = []

    def generate_script(self):
        self.scriptList = []
        for i, trajectory in enumerate(self.trajectories):
            scriptDict = {}
            scriptDict["benchmark"] = self.benchmark
            scriptDict["trajectory"] = trajectory
            scriptTemp = copy.deepcopy(self.scriptTemplate)
            scriptTemp = scriptTemp.format(self.root_dir, self.root_dir, self.root_dir, \
                                           self.root_dir, trajectory, \
                                           self.root_dir, trajectory, trajectory)
            scriptDict["script"] = scriptTemp
            self.scriptList.append(scriptDict)

    def get_script(self):
        return self.scriptList
    
    def copy_ground_truth_traj(self, trajectory):
        gt_csv_path = "{}/datasets/Hall/{}/groundtruth/data.csv"
        gt_csv_path = gt_csv_path.format(self.root_dir, trajectory)
        df = pd.read_csv(gt_csv_path)
        columns = ["#t[s:double]","p.x[m:double]", "p.y[m:double]", "p.z[m:double]", \
                "q.x[double]", "q.y[double]", "q.z[double]", "q.w[double]"]
        df = df[columns]
        df.loc[:,"#t[s:double]"] = df.loc[:,"#t[s:double]"] / 10**9
        df.to_csv("{}/logs/ground_truth.txt".format(self.root_dir), sep=' ', index=False, header=False)

    def copy_raw_SLAM_data(self, benchmark,trajectory,trial):
        raw_data_path = "{}/logs/log.csv"
        raw_data_path = raw_data_path.format(self.root_dir)
        destination_folder_path = "{}/data/raw".format(self.root_dir)
        new_filename = "raw_{}_{}_{}.csv".format(benchmark,trajectory,trial)
        tools.copy_file_and_rename(raw_data_path, destination_folder_path, new_filename)

    def load_raw_SLAM_data(self, benchmark, trajectory, trial,
                           raw_data_path = "{}/datasets/Raw_SLAM_Data/{}/raw/raw_{}_{}_{}.csv"):
        data_path = raw_data_path.format(self.root_dir, benchmark, benchmark, trajectory, trial)
        destination_folder_path = "{}/logs".format(self.root_dir)
        new_filename = "log.csv"
        tools.copy_file_and_rename(data_path, destination_folder_path, new_filename)

    def load_n_save_estimated_traj(self):
        # Step 0: Read the CSV file
        df = pd.read_csv("{}/logs/log.csv".format(self.root_dir))
        
        for index, value in enumerate(df['TimeStamp'][1:-1], start=1):
            if(value == df.at[index+1, 'TimeStamp'] or value == 0 or value < df.at[index-1, 'TimeStamp']):
                df.at[index, 'TimeStamp'] = 0.5*(df.at[index-1, 'TimeStamp'] + df.at[index+1, 'TimeStamp'])
        #assert(df["TimeStamp"].is_monotonic_increasing)
        #assert(df["TimeStamp"].is_unique)

        # Step 2: Drop the rows of initialization
        target = 2
        initial_rows = (df["TrackMode"] == target).idxmax()
        initial_rows += 2
        df = df.iloc[initial_rows:].reset_index(drop=True)

        # Step 3: Select several columns by their names
        df_traj = copy.deepcopy(df)
        selected_columns = ["TimeStamp","PX","PY","PZ","QX","QY","QZ","QW"]  # Replace with your desired column names
        df_traj = df_traj[selected_columns]

        # Step 4: Save the DataFrame as a text (TXT) file
        df_traj.to_csv("{}/logs/trajectory.txt".format(self.root_dir), \
                       sep=' ', index=False, header=False)  
        # You can change the separator as needed
        return df, df_traj, initial_rows


class Lab:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.benchmark = "Lab"
        self.scriptTemplate = "{}/Examples/Monocular/mono_tum_vi {}/Vocabulary/ORBvoc.txt {}/datasets/Lab/config.yaml {}/datasets/Lab/{}/mav0/cam0/data {}/datasets/Lab/{}/{}.txt"
        self.trajectories = ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7"]
        
        self.scriptList = []

    def generate_script(self):
        self.scriptList = []
        for i, trajectory in enumerate(self.trajectories):
            scriptDict = {}
            scriptDict["benchmark"] = self.benchmark
            scriptDict["trajectory"] = trajectory
            scriptTemp = copy.deepcopy(self.scriptTemplate)
            scriptTemp = scriptTemp.format(self.root_dir, self.root_dir, self.root_dir, \
                                           self.root_dir, trajectory, \
                                           self.root_dir, trajectory, trajectory)
            scriptDict["script"] = scriptTemp
            self.scriptList.append(scriptDict)

    def get_script(self):
        return self.scriptList
    
    def copy_ground_truth_traj(self, trajectory):
        gt_csv_path = "{}/datasets/Lab/{}/groundtruth/data.csv"
        gt_csv_path = gt_csv_path.format(self.root_dir, trajectory)
        df = pd.read_csv(gt_csv_path)
        columns = ["#t[s:double]","p.x[m:double]", "p.y[m:double]", "p.z[m:double]", \
                "q.x[double]", "q.y[double]", "q.z[double]", "q.w[double]"]
        df = df[columns]
        df.loc[:,"#t[s:double]"] = df.loc[:,"#t[s:double]"]
        df.to_csv("{}/logs/ground_truth.txt".format(self.root_dir), sep=' ', index=False, header=False)

    def copy_raw_SLAM_data(self, benchmark,trajectory,trial):
        raw_data_path = "{}/logs/log.csv"
        raw_data_path = raw_data_path.format(self.root_dir)
        destination_folder_path = "{}/data/raw".format(self.root_dir)
        new_filename = "raw_{}_{}_{}.csv".format(benchmark,trajectory,trial)
        tools.copy_file_and_rename(raw_data_path, destination_folder_path, new_filename)

    def load_raw_SLAM_data(self, benchmark, trajectory, trial,
                           raw_data_path = "{}/datasets/Raw_SLAM_Data/{}/raw/raw_{}_{}_{}.csv"):
        data_path = raw_data_path.format(self.root_dir, benchmark, benchmark, trajectory, trial)
        destination_folder_path = "{}/logs".format(self.root_dir)
        new_filename = "log.csv"
        tools.copy_file_and_rename(data_path, destination_folder_path, new_filename)

    def load_n_save_estimated_traj(self):
        # Step 0: Read the CSV file
        df = pd.read_csv("{}/logs/log.csv".format(self.root_dir))
        
        for index, value in enumerate(df['TimeStamp'][1:-1], start=1):
            if(value == df.at[index+1, 'TimeStamp'] or value == 0 or value < df.at[index-1, 'TimeStamp']):
                df.at[index, 'TimeStamp'] = 0.5*(df.at[index-1, 'TimeStamp'] + df.at[index+1, 'TimeStamp'])
        #assert(df["TimeStamp"].is_monotonic_increasing)
        #assert(df["TimeStamp"].is_unique)

        # Step 2: Drop the rows of initialization
        target = 2
        initial_rows = (df["TrackMode"] == target).idxmax()
        initial_rows += 2
        df = df.iloc[initial_rows:].reset_index(drop=True)

        # Step 3: Select several columns by their names
        df_traj = copy.deepcopy(df)
        selected_columns = ["TimeStamp","PX","PY","PZ","QX","QY","QZ","QW"]  # Replace with your desired column names
        df_traj = df_traj[selected_columns]

        # Step 4: Save the DataFrame as a text (TXT) file
        df_traj.to_csv("{}/logs/trajectory.txt".format(self.root_dir), \
                       sep=' ', index=False, header=False)  
        # You can change the separator as needed
        return df, df_traj, initial_rows

class Lab2:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.benchmark = "Lab2"
        self.scriptTemplate = "{}/Examples/Monocular/mono_tum_vi {}/Vocabulary/ORBvoc.txt {}/datasets/Lab2/config.yaml {}/datasets/Lab2/{}/mav0/cam0/data {}/datasets/Lab2/{}/{}.txt"
        self.trajectories = ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7"]
        self.scriptList = []

    def generate_script(self):
        self.scriptList = []
        for i, trajectory in enumerate(self.trajectories):
            scriptDict = {}
            scriptDict["benchmark"] = self.benchmark
            scriptDict["trajectory"] = trajectory
            scriptTemp = copy.deepcopy(self.scriptTemplate)
            scriptTemp = scriptTemp.format(self.root_dir, self.root_dir, self.root_dir, \
                                           self.root_dir, trajectory, \
                                           self.root_dir, trajectory, trajectory)
            scriptDict["script"] = scriptTemp
            self.scriptList.append(scriptDict)

    def get_script(self):
        return self.scriptList
    
    def copy_ground_truth_traj(self, trajectory):
        gt_csv_path = "{}/datasets/Lab2/{}/groundtruth/data.csv"
        gt_csv_path = gt_csv_path.format(self.root_dir, trajectory)
        df = pd.read_csv(gt_csv_path)
        columns = ["#t[s:double]","p.x[m:double]", "p.y[m:double]", "p.z[m:double]", \
                "q.x[double]", "q.y[double]", "q.z[double]", "q.w[double]"]
        df = df[columns]
        df.loc[:,"#t[s:double]"] = df.loc[:,"#t[s:double]"]
        df.to_csv("{}/logs/ground_truth.txt".format(self.root_dir), sep=' ', index=False, header=False)

    def copy_raw_SLAM_data(self, benchmark,trajectory,trial):
        raw_data_path = "{}/logs/log.csv"
        raw_data_path = raw_data_path.format(self.root_dir)
        destination_folder_path = "{}/data/raw".format(self.root_dir)
        new_filename = "raw_{}_{}_{}.csv".format(benchmark,trajectory,trial)
        tools.copy_file_and_rename(raw_data_path, destination_folder_path, new_filename)

    def load_raw_SLAM_data(self, benchmark, trajectory, trial,
                           raw_data_path = "{}/datasets/Raw_SLAM_Data/{}/raw/raw_{}_{}_{}.csv"):
        data_path = raw_data_path.format(self.root_dir, benchmark, benchmark, trajectory, trial)
        destination_folder_path = "{}/logs".format(self.root_dir)
        new_filename = "log.csv"
        tools.copy_file_and_rename(data_path, destination_folder_path, new_filename)

    def load_n_save_estimated_traj(self):
        # Step 0: Read the CSV file
        df = pd.read_csv("{}/logs/log.csv".format(self.root_dir))
        
        for index, value in enumerate(df['TimeStamp'][1:-1], start=1):
            if(value == df.at[index+1, 'TimeStamp'] or value == 0 or value < df.at[index-1, 'TimeStamp']):
                df.at[index, 'TimeStamp'] = 0.5*(df.at[index-1, 'TimeStamp'] + df.at[index+1, 'TimeStamp'])
        #assert(df["TimeStamp"].is_monotonic_increasing)
        #assert(df["TimeStamp"].is_unique)

        # Step 2: Drop the rows of initialization
        target = 2
        initial_rows = (df["TrackMode"] == target).idxmax()
        initial_rows += 2
        df = df.iloc[initial_rows:].reset_index(drop=True)

        # Step 3: Select several columns by their names
        df_traj = copy.deepcopy(df)
        selected_columns = ["TimeStamp","PX","PY","PZ","QX","QY","QZ","QW"]  # Replace with your desired column names
        df_traj = df_traj[selected_columns]

        # Step 4: Save the DataFrame as a text (TXT) file
        df_traj.to_csv("{}/logs/trajectory.txt".format(self.root_dir), \
                       sep=' ', index=False, header=False)  
        # You can change the separator as needed
        return df, df_traj, initial_rows

class StorageRoom:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.benchmark = "StorageRoom"
        self.scriptTemplate = "{}/Examples/Monocular/mono_tum_vi {}/Vocabulary/ORBvoc.txt {}/datasets/StorageRoom/config.yaml {}/datasets/StorageRoom/{}/mav0/cam0/data {}/datasets/StorageRoom/{}/{}.txt"
        self.trajectories = ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7"]
        self.scriptList = []

    def generate_script(self):
        self.scriptList = []
        for i, trajectory in enumerate(self.trajectories):
            scriptDict = {}
            scriptDict["benchmark"] = self.benchmark
            scriptDict["trajectory"] = trajectory
            scriptTemp = copy.deepcopy(self.scriptTemplate)
            scriptTemp = scriptTemp.format(self.root_dir, self.root_dir, self.root_dir, \
                                           self.root_dir, trajectory, \
                                           self.root_dir, trajectory, trajectory)
            scriptDict["script"] = scriptTemp
            self.scriptList.append(scriptDict)

    def get_script(self):
        return self.scriptList
    
    def copy_ground_truth_traj(self, trajectory):
        gt_csv_path = "{}/datasets/StorageRoom/{}/groundtruth/data.csv"
        gt_csv_path = gt_csv_path.format(self.root_dir, trajectory)
        df = pd.read_csv(gt_csv_path)
        columns = ["#t[s:double]","p.x[m:double]", "p.y[m:double]", "p.z[m:double]", \
                "q.x[double]", "q.y[double]", "q.z[double]", "q.w[double]"]
        df = df[columns]
        df.loc[:,"#t[s:double]"] = df.loc[:,"#t[s:double]"]
        df.to_csv("{}/logs/ground_truth.txt".format(self.root_dir), sep=' ', index=False, header=False)

    def copy_raw_SLAM_data(self, benchmark,trajectory,trial):
        raw_data_path = "{}/logs/log.csv"
        raw_data_path = raw_data_path.format(self.root_dir)
        destination_folder_path = "{}/data/raw".format(self.root_dir)
        new_filename = "raw_{}_{}_{}.csv".format(benchmark,trajectory,trial)
        tools.copy_file_and_rename(raw_data_path, destination_folder_path, new_filename)

    def load_raw_SLAM_data(self, benchmark, trajectory, trial,
                           raw_data_path = "{}/datasets/Raw_SLAM_Data/{}/raw/raw_{}_{}_{}.csv"):
        data_path = raw_data_path.format(self.root_dir, benchmark, benchmark, trajectory, trial)
        destination_folder_path = "{}/logs".format(self.root_dir)
        new_filename = "log.csv"
        tools.copy_file_and_rename(data_path, destination_folder_path, new_filename)

    def load_n_save_estimated_traj(self):
        # Step 0: Read the CSV file
        df = pd.read_csv("{}/logs/log.csv".format(self.root_dir))
        
        for index, value in enumerate(df['TimeStamp'][1:-1], start=1):
            if(value == df.at[index+1, 'TimeStamp'] or value == 0 or value < df.at[index-1, 'TimeStamp']):
                df.at[index, 'TimeStamp'] = 0.5*(df.at[index-1, 'TimeStamp'] + df.at[index+1, 'TimeStamp'])
        #assert(df["TimeStamp"].is_monotonic_increasing)
        #assert(df["TimeStamp"].is_unique)

        # Step 2: Drop the rows of initialization
        target = 2
        initial_rows = (df["TrackMode"] == target).idxmax()
        initial_rows += 2
        df = df.iloc[initial_rows:].reset_index(drop=True)

        # Step 3: Select several columns by their names
        df_traj = copy.deepcopy(df)
        selected_columns = ["TimeStamp","PX","PY","PZ","QX","QY","QZ","QW"]  # Replace with your desired column names
        df_traj = df_traj[selected_columns]

        # Step 4: Save the DataFrame as a text (TXT) file
        df_traj.to_csv("{}/logs/trajectory.txt".format(self.root_dir), \
                       sep=' ', index=False, header=False)  
        # You can change the separator as needed
        return df, df_traj, initial_rows


class Apartment:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.benchmark = "Apartment"
        self.scriptTemplate = "{}/Examples/Monocular/mono_tum_vi {}/Vocabulary/ORBvoc.txt {}/datasets/Apartment/Unity_ST.yaml {}/datasets/Apartment/InputData/{}/mav0/cam0/data {}/datasets/Apartment/InputData/{}/{}_timestamp.txt"
        self.trajectories = ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", \
                             "B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7"]
        self.scriptList = []
        self.timestamp_interger_num = 0

    def generate_script(self):
        self.scriptList = []
        for i, trajectory in enumerate(self.trajectories):
            scriptDict = {}
            scriptDict["benchmark"] = self.benchmark
            scriptDict["trajectory"] = trajectory
            scriptTemp = copy.deepcopy(self.scriptTemplate)
            scriptTemp = scriptTemp.format(self.root_dir, self.root_dir, self.root_dir, \
                                           self.root_dir, trajectory, \
                                           self.root_dir, trajectory, trajectory)
            scriptDict["script"] = scriptTemp
            self.scriptList.append(scriptDict)

    def get_script(self):
        return self.scriptList
    
    def copy_ground_truth_traj(self, trajectory):
        gt_csv_path = "{}/datasets/SenseTime/{}/groundtruth/data.csv"
        gt_csv_path = gt_csv_path.format(self.root_dir, trajectory)
        df = pd.read_csv(gt_csv_path)
        columns = ["#t[s:double]","p.x[m:double]", "p.y[m:double]", "p.z[m:double]", \
                "q.x[double]", "q.y[double]", "q.z[double]", "q.w[double]"]
        df = df[columns]

        df.to_csv("{}/logs/ground_truth.txt".format(self.root_dir), sep=' ', index=False, header=False)

    def copy_raw_SLAM_data(self, benchmark,trajectory,trial):
        raw_data_path = "{}/logs/log.csv"
        raw_data_path = raw_data_path.format(self.root_dir)
        destination_folder_path = "{}/data/raw".format(self.root_dir)
        new_filename = "raw_{}_{}_{}.csv".format(benchmark,trajectory,trial)
        tools.copy_file_and_rename(raw_data_path, destination_folder_path, new_filename)

    def load_raw_SLAM_data(self, benchmark, trajectory, trial,
                           raw_data_path = "{}/datasets/Raw_SLAM_Data/{}/raw/raw_{}_{}_{}.csv"):
        data_path = raw_data_path.format(self.root_dir, benchmark, benchmark, trajectory, trial)
        destination_folder_path = "{}/logs".format(self.root_dir)
        new_filename = "log.csv"
        tools.copy_file_and_rename(data_path, destination_folder_path, new_filename)

    def load_n_save_estimated_traj(self):
        # Step 0: Read the CSV file
        df = pd.read_csv("{}/logs/log.csv".format(self.root_dir))

        for index, value in enumerate(df['TimeStamp'][1:-1], start=1):
            if(value == df.at[index+1, 'TimeStamp'] or value == 0 or value < df.at[index-1, 'TimeStamp']):
                df.at[index, 'TimeStamp'] = 0.5*(df.at[index-1, 'TimeStamp'] + df.at[index+1, 'TimeStamp'])
        #assert(df["TimeStamp"].is_monotonic_increasing)
        #assert(df["TimeStamp"].is_unique)

        # Step 2: Drop the rows of initialization
        target = 2
        initial_rows = (df["TrackMode"] == target).idxmax()
        initial_rows += 2
        df = df.iloc[initial_rows:].reset_index(drop=True)

        # Step 3: Select several columns by their names
        df_traj = copy.deepcopy(df)
        selected_columns = ["TimeStamp","PX","PY","PZ","QX","QY","QZ","QW"]  # Replace with your desired column names
        df_traj = df_traj[selected_columns]

        # Step 4: Check timestamp unit whether the same as groundtruth
        integer_part, decimal_part = str(df_traj['TimeStamp'].values[0]).split('.')
        assert(len(decimal_part) == 9, "Please run the generate_timestamp.ipynb to correct the format")

        # Step 5: Save the DataFrame as a text (TXT) file
        df_traj.to_csv("{}/logs/trajectory.txt".format(self.root_dir), \
                       sep=' ', index=False, header=False)  
        df.to_csv("{}/logs/log.csv".format(self.root_dir), index=False)
        # You can change the separator as needed
        return df, df_traj, initial_rows

class AbandonedFactory:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.benchmark = "AbandonedFactory"
        self.scriptTemplate = "{}/Examples/Monocular/mono_tum_vi {}/Vocabulary/ORBvoc.txt {}/datasets/AbandonedFactory/Unity_ST.yaml {}/datasets/AbandonedFactory/InputData/{}/mav0/cam0/data {}/datasets/AbandonedFactory/InputData/{}/{}_timestamp.txt"
        self.trajectories = ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", \
                             "B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7"]
        self.scriptList = []
        self.timestamp_interger_num = 0

    def generate_script(self):
        self.scriptList = []
        for i, trajectory in enumerate(self.trajectories):
            scriptDict = {}
            scriptDict["benchmark"] = self.benchmark
            scriptDict["trajectory"] = trajectory
            scriptTemp = copy.deepcopy(self.scriptTemplate)
            scriptTemp = scriptTemp.format(self.root_dir, self.root_dir, self.root_dir, \
                                           self.root_dir, trajectory, \
                                           self.root_dir, trajectory, trajectory)
            scriptDict["script"] = scriptTemp
            self.scriptList.append(scriptDict)

    def get_script(self):
        return self.scriptList
    
    def copy_ground_truth_traj(self, trajectory):
        gt_csv_path = "{}/datasets/SenseTime/{}/groundtruth/data.csv"
        gt_csv_path = gt_csv_path.format(self.root_dir, trajectory)
        df = pd.read_csv(gt_csv_path)
        columns = ["#t[s:double]","p.x[m:double]", "p.y[m:double]", "p.z[m:double]", \
                "q.x[double]", "q.y[double]", "q.z[double]", "q.w[double]"]
        df = df[columns]

        df.to_csv("{}/logs/ground_truth.txt".format(self.root_dir), sep=' ', index=False, header=False)

    def copy_raw_SLAM_data(self, benchmark,trajectory,trial):
        raw_data_path = "{}/logs/log.csv"
        raw_data_path = raw_data_path.format(self.root_dir)
        destination_folder_path = "{}/data/raw".format(self.root_dir)
        new_filename = "raw_{}_{}_{}.csv".format(benchmark,trajectory,trial)
        tools.copy_file_and_rename(raw_data_path, destination_folder_path, new_filename)

    def load_raw_SLAM_data(self, benchmark, trajectory, trial,
                           raw_data_path = "{}/datasets/Raw_SLAM_Data/{}/raw/raw_{}_{}_{}.csv"):
        data_path = raw_data_path.format(self.root_dir, benchmark, benchmark, trajectory, trial)
        destination_folder_path = "{}/logs".format(self.root_dir)
        new_filename = "log.csv"
        tools.copy_file_and_rename(data_path, destination_folder_path, new_filename)

    def load_n_save_estimated_traj(self):
        # Step 0: Read the CSV file
        df = pd.read_csv("{}/logs/log.csv".format(self.root_dir))

        for index, value in enumerate(df['TimeStamp'][1:-1], start=1):
            if(value == df.at[index+1, 'TimeStamp'] or value == 0 or value < df.at[index-1, 'TimeStamp']):
                df.at[index, 'TimeStamp'] = 0.5*(df.at[index-1, 'TimeStamp'] + df.at[index+1, 'TimeStamp'])
        #assert(df["TimeStamp"].is_monotonic_increasing)
        #assert(df["TimeStamp"].is_unique)

        # Step 2: Drop the rows of initialization
        target = 2
        initial_rows = (df["TrackMode"] == target).idxmax()
        initial_rows += 2
        df = df.iloc[initial_rows:].reset_index(drop=True)

        # Step 3: Select several columns by their names
        df_traj = copy.deepcopy(df)
        selected_columns = ["TimeStamp","PX","PY","PZ","QX","QY","QZ","QW"]  # Replace with your desired column names
        df_traj = df_traj[selected_columns]

        # Step 4: Check timestamp unit whether the same as groundtruth
        integer_part, decimal_part = str(df_traj['TimeStamp'].values[0]).split('.')
        assert(len(decimal_part) == 9, "Please run the generate_timestamp.ipynb to correct the format")

        # Step 5: Save the DataFrame as a text (TXT) file
        df_traj.to_csv("{}/logs/trajectory.txt".format(self.root_dir), \
                       sep=' ', index=False, header=False)  
        df.to_csv("{}/logs/log.csv".format(self.root_dir), index=False)
        # You can change the separator as needed
        return df, df_traj, initial_rows

class FireStationGarage:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.benchmark = "FireStationGarage"
        self.scriptTemplate = "{}/Examples/Monocular/mono_tum_vi {}/Vocabulary/ORBvoc.txt {}/datasets/FireStationGarage/Unity_ST.yaml {}/datasets/FireStationGarage/InputData/{}/mav0/cam0/data {}/datasets/FireStationGarage/InputData/{}/{}_timestamp.txt"
        self.trajectories = ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", \
                             "B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7"]
        self.scriptList = []
        self.timestamp_interger_num = 0

    def generate_script(self):
        self.scriptList = []
        for i, trajectory in enumerate(self.trajectories):
            scriptDict = {}
            scriptDict["benchmark"] = self.benchmark
            scriptDict["trajectory"] = trajectory
            scriptTemp = copy.deepcopy(self.scriptTemplate)
            scriptTemp = scriptTemp.format(self.root_dir, self.root_dir, self.root_dir, \
                                           self.root_dir, trajectory, \
                                           self.root_dir, trajectory, trajectory)
            scriptDict["script"] = scriptTemp
            self.scriptList.append(scriptDict)

    def get_script(self):
        return self.scriptList
    
    def copy_ground_truth_traj(self, trajectory):
        gt_csv_path = "{}/datasets/SenseTime/{}/groundtruth/data.csv"
        gt_csv_path = gt_csv_path.format(self.root_dir, trajectory)
        df = pd.read_csv(gt_csv_path)
        columns = ["#t[s:double]","p.x[m:double]", "p.y[m:double]", "p.z[m:double]", \
                "q.x[double]", "q.y[double]", "q.z[double]", "q.w[double]"]
        df = df[columns]

        df.to_csv("{}/logs/ground_truth.txt".format(self.root_dir), sep=' ', index=False, header=False)

    def copy_raw_SLAM_data(self, benchmark,trajectory,trial):
        raw_data_path = "{}/logs/log.csv"
        raw_data_path = raw_data_path.format(self.root_dir)
        destination_folder_path = "{}/data/raw".format(self.root_dir)
        new_filename = "raw_{}_{}_{}.csv".format(benchmark,trajectory,trial)
        tools.copy_file_and_rename(raw_data_path, destination_folder_path, new_filename)

    def load_raw_SLAM_data(self, benchmark, trajectory, trial,
                           raw_data_path = "{}/datasets/Raw_SLAM_Data/{}/raw/raw_{}_{}_{}.csv"):
        data_path = raw_data_path.format(self.root_dir, benchmark, benchmark, trajectory, trial)
        destination_folder_path = "{}/logs".format(self.root_dir)
        new_filename = "log.csv"
        tools.copy_file_and_rename(data_path, destination_folder_path, new_filename)

    def load_n_save_estimated_traj(self):
        # Step 0: Read the CSV file
        df = pd.read_csv("{}/logs/log.csv".format(self.root_dir))

        for index, value in enumerate(df['TimeStamp'][1:-1], start=1):
            if(value == df.at[index+1, 'TimeStamp'] or value == 0 or value < df.at[index-1, 'TimeStamp']):
                df.at[index, 'TimeStamp'] = 0.5*(df.at[index-1, 'TimeStamp'] + df.at[index+1, 'TimeStamp'])
        #assert(df["TimeStamp"].is_monotonic_increasing)
        #assert(df["TimeStamp"].is_unique)

        # Step 2: Drop the rows of initialization
        target = 2
        initial_rows = (df["TrackMode"] == target).idxmax()
        initial_rows += 2
        df = df.iloc[initial_rows:].reset_index(drop=True)

        # Step 3: Select several columns by their names
        df_traj = copy.deepcopy(df)
        selected_columns = ["TimeStamp","PX","PY","PZ","QX","QY","QZ","QW"]  # Replace with your desired column names
        df_traj = df_traj[selected_columns]

        # Step 4: Check timestamp unit whether the same as groundtruth
        integer_part, decimal_part = str(df_traj['TimeStamp'].values[0]).split('.')
        assert(len(decimal_part) == 9, "Please run the generate_timestamp.ipynb to correct the format")

        # Step 5: Save the DataFrame as a text (TXT) file
        df_traj.to_csv("{}/logs/trajectory.txt".format(self.root_dir), \
                       sep=' ', index=False, header=False)  
        df.to_csv("{}/logs/log.csv".format(self.root_dir), index=False)
        # You can change the separator as needed
        return df, df_traj, initial_rows


class FireStationKitchen:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.benchmark = "FireStationKitchen"
        self.scriptTemplate = "{}/Examples/Monocular/mono_tum_vi {}/Vocabulary/ORBvoc.txt {}/datasets/FireStationKitchen/Unity_ST.yaml {}/datasets/FireStationKitchen/InputData/{}/mav0/cam0/data {}/datasets/FireStationKitchen/InputData/{}/{}_timestamp.txt"
        self.trajectories = ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", \
                             "B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7"]
        self.scriptList = []
        self.timestamp_interger_num = 0

    def generate_script(self):
        self.scriptList = []
        for i, trajectory in enumerate(self.trajectories):
            scriptDict = {}
            scriptDict["benchmark"] = self.benchmark
            scriptDict["trajectory"] = trajectory
            scriptTemp = copy.deepcopy(self.scriptTemplate)
            scriptTemp = scriptTemp.format(self.root_dir, self.root_dir, self.root_dir, \
                                           self.root_dir, trajectory, \
                                           self.root_dir, trajectory, trajectory)
            scriptDict["script"] = scriptTemp
            self.scriptList.append(scriptDict)

    def get_script(self):
        return self.scriptList
    
    def copy_ground_truth_traj(self, trajectory):
        gt_csv_path = "{}/datasets/SenseTime/{}/groundtruth/data.csv"
        gt_csv_path = gt_csv_path.format(self.root_dir, trajectory)
        df = pd.read_csv(gt_csv_path)
        columns = ["#t[s:double]","p.x[m:double]", "p.y[m:double]", "p.z[m:double]", \
                "q.x[double]", "q.y[double]", "q.z[double]", "q.w[double]"]
        df = df[columns]

        df.to_csv("{}/logs/ground_truth.txt".format(self.root_dir), sep=' ', index=False, header=False)

    def copy_raw_SLAM_data(self, benchmark,trajectory,trial):
        raw_data_path = "{}/logs/log.csv"
        raw_data_path = raw_data_path.format(self.root_dir)
        destination_folder_path = "{}/data/raw".format(self.root_dir)
        new_filename = "raw_{}_{}_{}.csv".format(benchmark,trajectory,trial)
        tools.copy_file_and_rename(raw_data_path, destination_folder_path, new_filename)

    def load_raw_SLAM_data(self, benchmark, trajectory, trial,
                           raw_data_path = "{}/datasets/Raw_SLAM_Data/{}/raw/raw_{}_{}_{}.csv"):
        data_path = raw_data_path.format(self.root_dir, benchmark, benchmark, trajectory, trial)
        destination_folder_path = "{}/logs".format(self.root_dir)
        new_filename = "log.csv"
        tools.copy_file_and_rename(data_path, destination_folder_path, new_filename)

    def load_n_save_estimated_traj(self):
        # Step 0: Read the CSV file
        df = pd.read_csv("{}/logs/log.csv".format(self.root_dir))

        for index, value in enumerate(df['TimeStamp'][1:-1], start=1):
            if(value == df.at[index+1, 'TimeStamp'] or value == 0 or value < df.at[index-1, 'TimeStamp']):
                df.at[index, 'TimeStamp'] = 0.5*(df.at[index-1, 'TimeStamp'] + df.at[index+1, 'TimeStamp'])
        #assert(df["TimeStamp"].is_monotonic_increasing)
        #assert(df["TimeStamp"].is_unique)

        # Step 2: Drop the rows of initialization
        target = 2
        initial_rows = (df["TrackMode"] == target).idxmax()
        initial_rows += 2
        df = df.iloc[initial_rows:].reset_index(drop=True)

        # Step 3: Select several columns by their names
        df_traj = copy.deepcopy(df)
        selected_columns = ["TimeStamp","PX","PY","PZ","QX","QY","QZ","QW"]  # Replace with your desired column names
        df_traj = df_traj[selected_columns]

        # Step 4: Check timestamp unit whether the same as groundtruth
        integer_part, decimal_part = str(df_traj['TimeStamp'].values[0]).split('.')
        assert(len(decimal_part) == 9, "Please run the generate_timestamp.ipynb to correct the format")

        # Step 5: Save the DataFrame as a text (TXT) file
        df_traj.to_csv("{}/logs/trajectory.txt".format(self.root_dir), \
                       sep=' ', index=False, header=False)  
        df.to_csv("{}/logs/log.csv".format(self.root_dir), index=False)
        # You can change the separator as needed
        return df, df_traj, initial_rows


class FireStationOffice:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.benchmark = "FireStationOffice"
        self.scriptTemplate = "{}/Examples/Monocular/mono_tum_vi {}/Vocabulary/ORBvoc.txt {}/datasets/FireStationOffice/Unity_ST.yaml {}/datasets/FireStationOffice/InputData/{}/mav0/cam0/data {}/datasets/FireStationOffice/InputData/{}/{}_timestamp.txt"
        self.trajectories = ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", \
                             "B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7"]
        self.scriptList = []
        self.timestamp_interger_num = 0

    def generate_script(self):
        self.scriptList = []
        for i, trajectory in enumerate(self.trajectories):
            scriptDict = {}
            scriptDict["benchmark"] = self.benchmark
            scriptDict["trajectory"] = trajectory
            scriptTemp = copy.deepcopy(self.scriptTemplate)
            scriptTemp = scriptTemp.format(self.root_dir, self.root_dir, self.root_dir, \
                                           self.root_dir, trajectory, \
                                           self.root_dir, trajectory, trajectory)
            scriptDict["script"] = scriptTemp
            self.scriptList.append(scriptDict)

    def get_script(self):
        return self.scriptList
    
    def copy_ground_truth_traj(self, trajectory):
        gt_csv_path = "{}/datasets/SenseTime/{}/groundtruth/data.csv"
        gt_csv_path = gt_csv_path.format(self.root_dir, trajectory)
        df = pd.read_csv(gt_csv_path)
        columns = ["#t[s:double]","p.x[m:double]", "p.y[m:double]", "p.z[m:double]", \
                "q.x[double]", "q.y[double]", "q.z[double]", "q.w[double]"]
        df = df[columns]

        df.to_csv("{}/logs/ground_truth.txt".format(self.root_dir), sep=' ', index=False, header=False)

    def copy_raw_SLAM_data(self, benchmark,trajectory,trial):
        raw_data_path = "{}/logs/log.csv"
        raw_data_path = raw_data_path.format(self.root_dir)
        destination_folder_path = "{}/data/raw".format(self.root_dir)
        new_filename = "raw_{}_{}_{}.csv".format(benchmark,trajectory,trial)
        tools.copy_file_and_rename(raw_data_path, destination_folder_path, new_filename)

    def load_raw_SLAM_data(self, benchmark, trajectory, trial,
                           raw_data_path = "{}/datasets/Raw_SLAM_Data/{}/raw/raw_{}_{}_{}.csv"):
        data_path = raw_data_path.format(self.root_dir, benchmark, benchmark, trajectory, trial)
        destination_folder_path = "{}/logs".format(self.root_dir)
        new_filename = "log.csv"
        tools.copy_file_and_rename(data_path, destination_folder_path, new_filename)

    def load_n_save_estimated_traj(self):
        # Step 0: Read the CSV file
        df = pd.read_csv("{}/logs/log.csv".format(self.root_dir))

        for index, value in enumerate(df['TimeStamp'][1:-1], start=1):
            if(value == df.at[index+1, 'TimeStamp'] or value == 0 or value < df.at[index-1, 'TimeStamp']):
                df.at[index, 'TimeStamp'] = 0.5*(df.at[index-1, 'TimeStamp'] + df.at[index+1, 'TimeStamp'])
        #assert(df["TimeStamp"].is_monotonic_increasing)
        #assert(df["TimeStamp"].is_unique)

        # Step 2: Drop the rows of initialization
        target = 2
        initial_rows = (df["TrackMode"] == target).idxmax()
        initial_rows += 2
        df = df.iloc[initial_rows:].reset_index(drop=True)

        # Step 3: Select several columns by their names
        df_traj = copy.deepcopy(df)
        selected_columns = ["TimeStamp","PX","PY","PZ","QX","QY","QZ","QW"]  # Replace with your desired column names
        df_traj = df_traj[selected_columns]

        # Step 4: Check timestamp unit whether the same as groundtruth
        integer_part, decimal_part = str(df_traj['TimeStamp'].values[0]).split('.')
        assert(len(decimal_part) == 9, "Please run the generate_timestamp.ipynb to correct the format")

        # Step 5: Save the DataFrame as a text (TXT) file
        df_traj.to_csv("{}/logs/trajectory.txt".format(self.root_dir), \
                       sep=' ', index=False, header=False)  
        df.to_csv("{}/logs/log.csv".format(self.root_dir), index=False)
        # You can change the separator as needed
        return df, df_traj, initial_rows


benchmark_factory = {
    "SenseTime": SenseTime,
    "LivingRoom": LivingRoom,
    "Hall": Hall,
    "Lab": Lab,
    "Lab2": Lab2,
    "StorageRoom": StorageRoom,
    "Apartment": Apartment,
    "AbandonedFactory": AbandonedFactory,
    "FireStationGarage": FireStationGarage,
    "FireStationKitchen": FireStationKitchen,
    "FireStationOffice": FireStationOffice
}