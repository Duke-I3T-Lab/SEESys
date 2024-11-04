from typing import Optional
import os
from multiprocessing import Pool, cpu_count
import glob
import re
import logging
from itertools import repeat, chain

import numpy as np
import pandas as pd
from tqdm import tqdm

from functools import partial

logger = logging.getLogger('__main__')


class Normalizer(object):
    """
    Normalizes dataframe across ALL contained rows (time steps). Different from per-sample normalization.
    """

    def __init__(self, norm_type, mean=None, std=None, min_val=None, max_val=None):
        """
        Args:
            norm_type: choose from:
                "standardization", "minmax": normalizes dataframe across ALL contained rows (time steps)
                "per_sample_std", "per_sample_minmax": normalizes each sample separately (i.e. across only its own rows)
            mean, std, min_val, max_val: optional (num_feat,) Series of pre-computed values
        """

        self.norm_type = norm_type
        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val

    def normalize(self, df):
        """
        Args:
            df: input dataframe
        Returns:
            df: normalized dataframe
        """
        if self.norm_type == "standardization":
            if self.mean is None:
                self.mean = df.mean()
                self.std = df.std()
            return (df - self.mean) / (self.std + np.finfo(float).eps)

        elif self.norm_type == "minmax":
            if self.max_val is None:
                self.max_val = df.max()
                self.min_val = df.min()
            return (df - self.min_val) / (self.max_val - self.min_val + np.finfo(float).eps)

        elif self.norm_type == "per_sample_std":
            grouped = df.groupby(by=df.index)
            return (df - grouped.transform('mean')) / grouped.transform('std')

        elif self.norm_type == "per_sample_minmax":
            grouped = df.groupby(by=df.index)
            min_vals = grouped.transform('min')
            return (df - min_vals) / (grouped.transform('max') - min_vals + np.finfo(float).eps)

        else:
            raise (NameError(f'Normalize method "{self.norm_type}" not implemented'))


def interpolate_missing(y):
    """
    Replaces NaN values in pd.Series `y` using linear interpolation
    """
    if y.isna().any():
        y = y.interpolate(method='linear', limit_direction='both')
    return y


def subsample(y, limit=256, factor=2):
    """
    If a given Series is longer than `limit`, returns subsampled sequence by the specified integer factor
    """
    if len(y) > limit:
        return y[::factor].reset_index(drop=True)
    return y


class BaseData(object):

    def set_num_processes(self, n_proc):

        if (n_proc is None) or (n_proc <= 0):
            self.n_proc = cpu_count()  # max(1, cpu_count() - 1)
        else:
            self.n_proc = min(n_proc, cpu_count())


class DeepSEEData(BaseData):
    """
    Dataset class for Machine dataset.
    Attributes:
        all_df: dataframe indexed by ID, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(self, root_dir, dataset_list=None, n_proc=1, config=None):
        self.root_dir = root_dir
        self.dataset_list = dataset_list
        self.set_num_processes(n_proc=n_proc)
        self.window_length = config.window_length
        self.window_step = config.window_step
        self.time_series_paths = {}
        
        self.config = config

        self.dataset_trajectory = {
            "SenseTime": [
                "A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", \
                "B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7", \
                "C8", "C9", "C10", "C11", "D8", "D9", "D10"
            ],
            "LivingRoom": [
                "A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7"
            ],
            "Hall": [
                "A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7"
            ],
            "Lab": [
                "A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7"
            ],
            "Lab2": [
                "A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7"
            ],
            
            "Apartment": [
                "A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", \
                "B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7"
            ],
            "FireStationKitchen": [
                "A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", \
                "B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7"
            ],
            "FireStationOffice": [
                "A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", \
                "B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7"
            ],
            "FireStationGarage": [
                "A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", \
                "B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7"
            ],
            "AbandonedFactory": [
                "A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", \
                "B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7"
            ],
            "TUMRGBD1": [
                'rgbd_dataset_freiburg1_360',
                'rgbd_dataset_freiburg1_desk',
                'rgbd_dataset_freiburg1_desk2',
                'rgbd_dataset_freiburg1_floor',
                'rgbd_dataset_freiburg1_plant',
                'rgbd_dataset_freiburg1_room',
                'rgbd_dataset_freiburg1_rpy',
                'rgbd_dataset_freiburg1_teddy',
                'rgbd_dataset_freiburg1_xyz'
            ],
            "TUMRGBD2": [
                'rgbd_dataset_freiburg2_360_hemisphere',
                'rgbd_dataset_freiburg2_360_kidnap',
                'rgbd_dataset_freiburg2_coke',
                'rgbd_dataset_freiburg2_desk',
                'rgbd_dataset_freiburg2_desk_with_person',
                'rgbd_dataset_freiburg2_dishes',
                'rgbd_dataset_freiburg2_flowerbouquet',
                'rgbd_dataset_freiburg2_flowerbouquet_brownbackground',
                'rgbd_dataset_freiburg2_large_no_loop',
                'rgbd_dataset_freiburg2_large_with_loop',
                'rgbd_dataset_freiburg2_metallic_sphere',
                'rgbd_dataset_freiburg2_metallic_sphere2',
                'rgbd_dataset_freiburg2_pioneer_360',
                'rgbd_dataset_freiburg2_pioneer_slam',
                'rgbd_dataset_freiburg2_pioneer_slam2',
                'rgbd_dataset_freiburg2_pioneer_slam3',
                'rgbd_dataset_freiburg2_rpy',
                'rgbd_dataset_freiburg2_xyz'
            ],
            "TUMRGBD3": [
                'rgbd_dataset_freiburg3_cabinet',
                'rgbd_dataset_freiburg3_large_cabinet',
                'rgbd_dataset_freiburg3_long_office_household',
                'rgbd_dataset_freiburg3_nostructure_notexture_far',
                'rgbd_dataset_freiburg3_nostructure_notexture_near_withloop',
                'rgbd_dataset_freiburg3_nostructure_texture_far',
                'rgbd_dataset_freiburg3_nostructure_texture_near_withloop',
                'rgbd_dataset_freiburg3_sitting_halfsphere',
                'rgbd_dataset_freiburg3_sitting_rpy',
                'rgbd_dataset_freiburg3_sitting_static',
                'rgbd_dataset_freiburg3_sitting_xyz',
                'rgbd_dataset_freiburg3_structure_notexture_far',
                'rgbd_dataset_freiburg3_structure_notexture_near',
                'rgbd_dataset_freiburg3_structure_texture_far',
                'rgbd_dataset_freiburg3_structure_texture_near',
                'rgbd_dataset_freiburg3_teddy',
                'rgbd_dataset_freiburg3_walking_halfsphere',
                'rgbd_dataset_freiburg3_walking_rpy',
                'rgbd_dataset_freiburg3_walking_static',
                'rgbd_dataset_freiburg3_walking_xyz'
            ]
        }

        # Initialize data dictionary
        self.data = {}

        all_results = self.load_all(root_dir, dataset_list=dataset_list)

        for i, result in enumerate(all_results):
            #feature_df_list, label_df_list, sequence_list = zip(*result)
            data_dict_list, datasets_list, sequences_list, iters_list = zip(*result)

            for j, data_dict in enumerate(data_dict_list):
                dataset = datasets_list[j]
                sequence = sequences_list[j]
                iter = iters_list[j]

                if dataset not in list(self.data.keys()):
                    self.data[dataset] = {}
                if sequence not in list(self.data[dataset].keys()):
                    self.data[dataset][sequence] = {}
                
                self.data[dataset][sequence][iter] = data_dict


    def load_all(self, root_dir, dataset_list):
        all_results = []
        for idx, dataset in enumerate(dataset_list):
            print("Loading DeepSEE-{} dataset".format(dataset))
            # Check dataset 
            assert dataset in list(self.dataset_trajectory.keys()), "Dataset is not defined in dataset_trajectory!"
            # Create dataset and trajectory dict
            self.data[dataset] = {}
            for traj in self.dataset_trajectory[dataset]:
                self.data[dataset][traj] = {}
                # Delete the traj if it is still None after load_all

            # dataset_path: './data/SenseTime/', './data/TUMRGBD1/', ...
            dataset_path = os.path.join(root_dir, dataset)
        
            time_series_paths = glob.glob(os.path.join(dataset_path, 'timeSeries/*'))

            time_series_paths = [p for p in time_series_paths if os.path.isfile(p) and p.endswith('.csv')]

            self.time_series_paths[dataset] = time_series_paths

            if len(time_series_paths) == 0:
                raise Exception("No .csv files found under dataset: '{}'".format(dataset))
            
            if self.n_proc > 1:
                # Load in parallel
                _n_proc = min(self.n_proc, len(time_series_paths))  # no more than file_names needed here
                logger.info("Loading {} datasets files using {} parallel processes ...".format(dataset, _n_proc))
                with Pool(processes=_n_proc) as pool:
                    result = pool.map(partial(DeepSEEData.load_single,
                                              dataset=dataset,
                                              window_length=self.window_length,
                                              window_step=self.window_step,
                                              target_transform=self.config.target_transform),
                                              time_series_paths)
            else:  # read 1 file at a time
                result = [DeepSEEData.load_single(dataset=dataset, 
                                                  file_path=path, 
                                                  window_length=self.window_length, 
                                                  window_step=self.window_step,
                                                  target_transform=self.config.target_transform) for path in time_series_paths]

            # merge two dictionary
            all_results.append(result)
        return all_results
    
    @staticmethod
    def select_columns(df, target_transform=False):
        keep_cols = ['RelativeError','TimeStamp','TrackMode','Brightness','Contrast','Entropy','Laplacian', 
                    'AvgMPDepth','VarMPDepth','PrePOKeyMapLoss',\
                    'PostPOOutlier', 'MatchedInlier',\
                    'DX','DY','DZ','Yaw','Pitch','Roll',\
                    'local_visual_BA_Err']
        df = df[keep_cols].astype(float)

        # Set abnormal values to NaN
        threshold = 0.05 # 5 cm in 0.33 second
        df.loc[:, 'RelativeError'] = df['RelativeError'].apply(lambda x: np.nan if x > threshold else x)

        # Record Nan value indices
        nan_indices = df[df['RelativeError'].isna()].index

        # Remove waving initialization indices
        init_index = np.min(df[df['TrackMode']==2].index)
        remove_indices = pd.Index(range(init_index, init_index+240)) # remove the following 8 seconds
        nan_indices = nan_indices.append(remove_indices).drop_duplicates().sort_values()
        
        #df.loc[:, 'RelativeError'] = df['RelativeError'].fillna(0)
        if target_transform:
            df.loc[:, 'RelativeError'] = df['RelativeError'].clip(lower=0.001, upper=0.02)
            df.loc[:, 'RelativeError'] = np.log1p(10000*df['RelativeError'])
        
        else:
            df.loc[:, 'RelativeError'] = df['RelativeError'].clip(lower=0.001, upper=0.02)
        
        ## uniform smoothing
        df.loc[:, 'RelativeError'] = df['RelativeError'].rolling(window=15, min_periods=1).apply(np.nanmean, raw=True)

        # Recover Nan value
        df.loc[nan_indices, 'RelativeError'] = np.nan

        df.loc[:, 'Yaw'] = np.sin(df['Yaw']) #df['Yaw']/np.pi
        df.loc[:, 'Pitch'] = np.sin(df['Pitch']) #df['Pitch']/np.pi
        df.loc[:, 'Roll'] = np.sin(df['Roll']) #df['Roll']/np.pi

        df.loc[:, 'Brightness'] = df['Brightness']/160.0
        df.loc[:, 'Contrast'] = df['Contrast']/70.0
        df.loc[:, 'Entropy'] = df['Entropy']/8.0
        df.loc[:, 'Laplacian'] = df['Laplacian']/90.0
        df.loc[:, 'AvgMPDepth'] = df['AvgMPDepth']*1.2
        df.loc[:, 'VarMPDepth'] = df['VarMPDepth']*4.0
        df.loc[:, 'PrePOKeyMapLoss'] = df['PrePOKeyMapLoss']/600.0
        df.loc[:, 'PostPOOutlier'] = df['PostPOOutlier']/100.0
        df.loc[:, 'MatchedInlier'] = df['MatchedInlier']/400.0
        df.loc[:, 'DX'] = df['DX']*100.0
        df.loc[:, 'DY'] = df['DY']*100.0
        df.loc[:, 'DZ'] = df['DZ']*100.0

        df.loc[:, 'local_visual_BA_Err'] = np.log1p(df['local_visual_BA_Err'])/10.0
        return df
    

    @staticmethod
    def read_frame_from_file(file):
        # Read the metadata
        rows = np.fromfile(file, dtype=np.int32, count=1)[0]
        cols = np.fromfile(file, dtype=np.int32, count=1)[0]
        type_ = np.fromfile(file, dtype=np.int32, count=1)[0]
        # Calculate the total size of image data
        total_size = rows * cols * 3  # Assuming the image type is CV_8UC3
        # Read the raw image data
        frame_data = np.fromfile(file, dtype=np.uint8, count=total_size)
        # Reshape the data into an image
        frame = np.reshape(frame_data, (rows, cols, 3)).transpose(2,0,1) # shape (3, 96, 128)
        frame = np.expand_dims(frame, axis=0) # shape (1, 3, 96, 128)
        return frame
    
    @staticmethod
    def read_point_dist(ts_file_path, dataset, sequence, iter, time_stamp):
        dataset_path = os.path.dirname(os.path.dirname(ts_file_path))
        pointDist_dir = os.path.join(dataset_path, 
                                     "pointDist/dist_{}_{}_{}".format(dataset, 
                                                                       sequence, 
                                                                       iter))
        pointDist_path = os.path.join(pointDist_dir, "{}.bin".format(time_stamp))
        with open(pointDist_path, "rb") as file:
            frames = []
            while True:
                try:
                    frame = DeepSEEData.read_frame_from_file(file)
                    frames.append(frame)
                except IndexError:
                    break
        
        # frames: [(1, 3, 96, 128),(1, 3, 96, 128),(1, 3, 96, 128),(1, 3, 96, 128)]
        return np.concatenate(frames, axis=0) # shape (4, 3, 96, 128)


    @staticmethod
    def load_single(file_path, window_length, window_step, target_transform, dataset):
        df = DeepSEEData.read_data(file_path)

        # file_path e.g., ./data/SenseTime/timeSeries/data_SenseTime_A0_0.csv
        file_name = file_path.split('/')[-1].split('.')[-2]
        # file_name e.g., data_SenseTime_A0_0
        
        sequence = None
        iter = None
        if(dataset in ["SenseTime", "LivingRoom", "Hall", "Lab", "Lab2", "Apartment",\
                       "FireStationKitchen", "FireStationOffice",\
                        "FireStationGarage", "AbandonedFactory"]):
            sequence = file_name.split('_')[-2]
            iter = file_name.split('_')[-1]
        ## TODO: verify the following code when new datasets are available
        elif(dataset == "TUMRGBD1"):
            sequence = "_".join(file_name.split("_")[1:-1])
            iter = file_name.split('_')[-1]
        elif(dataset == "TUMRGBD2"):
            sequence = "_".join(file_name.split("_")[1:-1])
            iter = file_name.split('_')[-1]
        elif(dataset == "TUMRGBD3"):
            sequence = "_".join(file_name.split("_")[1:-1])
            iter = file_name.split('_')[-1]

        assert sequence is not None, "Cannot find sequence: {} in {}".format(sequence, dataset)
        try:
            iter = int(iter)
        except:
            raise Exception("iter should be an integer!")

        df = DeepSEEData.select_columns(df, target_transform)

        # Identify invalid rows
        # Step 1: need to have RE
        df['isValid'] = df['RelativeError'].notna()

        df['RelativeError'] = df['RelativeError'].fillna(2.5)
        # Step 2: RE is not reliable after lost tracking + window_length
        for index, row in df.iterrows():
            if row['TrackMode'] != 2:
                if index < len(df) - window_length - 10:
                    df.loc[index:index+window_length+10, 'isValid'] = False
                else:
                    df.loc[index:, 'isValid'] = False

        data_dict = {}

        next_index = 0
        for index, row in df.iterrows():
            if index < next_index:
                continue

            isValid = row['isValid']
            if isValid:
                next_index += window_step

                # ['RelativeError','TimeStamp','TrackMode','Brightness','Contrast','Entropy','Laplacian', 
                # 'AvgMPDepth','VarMPDepth','PrePOKeyMapLoss',\
                # 'PostPOOutlier', 'MatchedInlier','NumberKeyPoints',\
                # 'DX','DY','DZ','Yaw','Pitch','Roll',\
                # 'local_visual_BA_Err',\
                # 'isValid']
                time_stamp = "{:.6f}".format(row.iloc[1])
                data_dict[time_stamp] = {}

                data_dict[time_stamp]['relativeError'] = row.iloc[0]
                data_dict[time_stamp]['timeSeries'] = df.iloc[index-30:index, 
                                                              3:-1].fillna(0)#.values
                data_dict[time_stamp]['pointDist'] = DeepSEEData.read_point_dist(file_path, 
                                                                                 dataset,                                                                                                                                     sequence, 
                                                                                 iter, 
                                                                                 time_stamp)
            else:
                next_index += 1 

        return data_dict, dataset, sequence, iter

    @staticmethod
    def read_data(file_path):
        """Reads a single .csv, which typically contains a day of datasets of various machine sessions.
        """
        df = pd.read_csv(file_path, delimiter=',')
        return df

data_factory = {'SLAMData': DeepSEEData}
