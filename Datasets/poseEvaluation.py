from evo.tools import log
log.configure_logging(verbose=True, debug=True, silent=False)

import pprint
import numpy as np

from evo.tools import plot
import matplotlib.pyplot as plt
# %matplotlib inline
# %matplotlib notebook

# temporarily override some package settings
from evo.tools.settings import SETTINGS
SETTINGS.plot_usetex = False

from evo.tools import file_interface
from evo.core import sync
from evo.core import metrics

# Import for plotting trajectory with error magnitude
from evo.core.metrics import PoseRelation, Unit

import copy
import pandas as pd

# Import for regional alignment
from evo import core
from evo.core.trajectory import PosePath3D, PoseTrajectory3D

class PoseErrorEvaluator:
    def __init__(self, root_dir, delta=60, max_diff=0.05, max_null_length=10):
        self.root_dir = root_dir
        self.delta = delta
        self.max_diff = max_diff
        self.max_null_length = max_null_length
        self.traj_ref = None
        self.traj_est = None
        self.merged_df = None

    @staticmethod
    def find_align_regions(traj_est, speed_threshold=1, rescale_threshold=200): 
        # 1. Find out regions that is lost tracking
        zero_indices = np.where(traj_est.speeds == 0.0)[0]
        lost_regions = []
        if len(zero_indices) > 0:
            start = zero_indices[0]
            end = zero_indices[0]

            for num in zero_indices[1:]:
                if num == end + 1 :
                    end = num
                else:
                    if(end - start > 1):
                        lost_regions.append([start,end])
                    end = num
                    start = num
        # 2. Find out moments that has map merge/relocalization
        shift_checkpoints = np.where(traj_est.speeds>speed_threshold)[0]
        # 3. Merge the first two regions
        checkpoints = []
        checkpoints.append(0)
        checkpoints.append(traj_est.num_poses-1)
        lost_checkpoints = []
        for lost_region in lost_regions:
            lost_checkpoints.append(lost_region[0])
            lost_checkpoints.append(lost_region[1])
        checkpoints = checkpoints + lost_checkpoints
        checkpoints = checkpoints + list(shift_checkpoints)
        checkpoints = list(set(checkpoints))
        checkpoints = sorted(checkpoints)
        # 4. Add auxilary checkpoints for scale problem
        aux_checkpoints = []
        aux_checkpoint = checkpoints[0]
        for checkpoint in checkpoints[1:]:
            while(checkpoint - aux_checkpoint > rescale_threshold):
                aux_checkpoint += rescale_threshold
                aux_checkpoints.append(aux_checkpoint)
            aux_checkpoint = checkpoint
        checkpoints = checkpoints + aux_checkpoints
        checkpoints = list(set(checkpoints))
        checkpoints = sorted(checkpoints)
        # 5. Generate align regions
        align_regions = []
        start = checkpoints[0]
        for idx, checkpoint in enumerate(checkpoints[1:]):
            if(checkpoint - start <= 1):
                ## when there are consecutive checkpoints
                # When start in lost and end in shift, 
                # they need to be in seperate region
                if (start in lost_checkpoints and checkpoint in shift_checkpoints):
                    start = checkpoint
                # When start in shift and end in shift, 
                # they need to be in the same region
                if (start in shift_checkpoints and checkpoint in shift_checkpoints):
                    pass
                
            else:
                align_regions.append([start, checkpoint])
                start = checkpoint
        align_regions_dict = {}
        align_regions_dict['align_regions'] = align_regions
        align_regions_dict['lost_regions'] = lost_regions
        align_regions_dict['shift_checkpoints'] = shift_checkpoints
        align_regions_dict['aux_checkpoints'] = aux_checkpoints
        return align_regions_dict



    def load_trajectory(self):
        ref_file = "{}/logs/ground_truth.txt".format(self.root_dir)
        est_file = "{}/logs/trajectory.txt".format(self.root_dir)

        traj_est = file_interface.read_tum_trajectory_file(est_file)
        traj_ref = file_interface.read_tum_trajectory_file(ref_file)

        # Just in case that
        # the first timestamp is not initialized yet
        traj_est.timestamps[0] = traj_est.timestamps[1] - 0.03

        align_regions_dict = PoseErrorEvaluator.find_align_regions(traj_est)
        print("Potential subtrajectories for alignment: {}".format(align_regions_dict['align_regions']))
        # split trajectory to subtrajectories
        # xyz = traj_est._positions_xyz
        # quat = traj_est._orientations_quat_wxyz

        # # Process invalid regions
        # for idx, region in enumerate(align_regions):
        #     if region[1]-region[0] == 1:
        #         # Need to adjust the trajectory
        #         if region[0] == 0:
        #             # If it is the first row in traj, 
        #             # overwrite the first&second row by the third row
        #             traj_est._positions_xyz[0, :] = xyz[2, :]
        #             traj_est._positions_xyz[1, :] = xyz[2, :]
        #             traj_est._orientations_quat_wxyz[0, :] = quat[2, :]
        #             traj_est._orientations_quat_wxyz[1, :] = quat[2, :]
        #         elif region[1] == traj_est.num_poses-1:
        #             # If it is the last row in traj,
        #             # overwrite the last 2 row by the last third row
        #             traj_est._positions_xyz[-1, :] = xyz[-3, :]
        #             traj_est._positions_xyz[-2, :] = xyz[-3, :]
        #             traj_est._orientations_quat_wxyz[-1, :] = quat[-3, :]
        #             traj_est._orientations_quat_wxyz[-2, :] = quat[-3, :]
        #         else:
        #             # Then it happens in the middle of trajectory
        #             traj_est._positions_xyz[region[0], :] = xyz[region[0]-1, :]
        #             traj_est._positions_xyz[region[1], :] = xyz[region[1]+1, :]
        #             traj_est._orientations_quat_wxyz[region[0], :] = quat[region[0]-1, :]
        #             traj_est._orientations_quat_wxyz[region[1], :] = quat[region[1]+1,:]

        # # Now redo the find_align_regions
        # align_regions = PoseErrorEvaluator.find_align_regions(traj_est)
        # # Force to align subtrajectories for correcting scales
        # align_regions = PoseErrorEvaluator.force_align_regions(traj_est, align_regions)
        if len(align_regions_dict['align_regions']) > 0:
            # print("Actual subtrajectories for alignment: {}".format(align_regions))
            xyz = traj_est._positions_xyz
            quat = traj_est._orientations_quat_wxyz
            time = traj_est.timestamps
            # split trajectory to subtrajectories
            subtrajectories = []
            for idx, region in enumerate(align_regions_dict['align_regions']):
                xyz_sub = xyz[region[0]:region[1], :]
                xyz_sub[0,:] = xyz_sub[1,:]
                xyz_sub[-1,:] = xyz_sub[-2,:]
                
                quat_sub = quat[region[0]:region[1], :]
                quat_sub[0,:] = quat_sub[1,:]
                quat_sub[-1,:] = quat_sub[-2,:]

                # Perform subtrajectory alignment
                time_sub = time[region[0]:region[1]]
                traj_sub = PoseTrajectory3D(xyz_sub, quat_sub, time_sub)
                print("Original estimated subtrajectory length: {}".format(traj_sub.num_poses))
                try:
                    traj_ref_copy = file_interface.read_tum_trajectory_file(ref_file) #copy.deepcopy(traj_ref)
                    traj_ref_copy, traj_sub = sync.associate_trajectories(traj_ref_copy, traj_sub, max_diff=0.05)
                    
                    #n = int(traj_sub.timestamps.shape[0]/2)
                    #traj_sub.align(traj_ref_copy, correct_scale=True, correct_only_scale=False, n=n)
                    traj_sub.align(traj_ref_copy, correct_scale=True, correct_only_scale=False)
                    
                    subtrajectories.append(traj_sub)
                except Exception as e:
                    print("subtrajectory alignment failed: {}".format(e))
                    traj_ref_copy = file_interface.read_tum_trajectory_file(ref_file) #copy.deepcopy(traj_ref)
                    traj_ref_sub, traj_sub = sync.associate_trajectories(traj_ref_copy, traj_sub, self.max_diff)
                    subtrajectories.append(traj_ref_sub)

                #self.plot_trajectory(subtrajectories[-1], traj_ref, trial=idx)

            traj_est_aligned = core.trajectory.merge(subtrajectories)
            traj_ref, traj_est_aligned = sync.associate_trajectories(traj_ref, traj_est_aligned, self.max_diff)
            #n = int(traj_est_aligned.timestamps.shape[0]/2)
            #print("aligned length: {}".format(n))
            #traj_est_aligned.align(traj_ref, correct_scale=True, correct_only_scale=False, n=n)
        else:
            # The redo find_align_region cannot find any subtrajectories to align
            # then directly synchronize the two trajectories
            traj_ref, traj_est_aligned = sync.associate_trajectories(traj_ref, traj_est, self.max_diff)

        self.traj_ref = traj_ref
        #self.traj_est = traj_est
        self.traj_est = traj_est_aligned
        print("Loaded trajectory ({} poses)  with ground truth {} poses".format(traj_est_aligned.num_poses, traj_ref.num_poses))
        print("="*50)

    def calculate_RE(self, pose_relation = metrics.PoseRelation.translation_part):
        # error metric settings
        #pose_relation = metrics.PoseRelation.translation_part
        #pose_relation = metrics.PoseRelation.point_distance
        #pose_relation = metrics.PoseRelation.point_distance_error_ratio
        
        delta_unit = metrics.Unit.frames
        all_pairs = True
        # form the (reference, estimation) pair
        data = (self.traj_ref, self.traj_est)
        # load error metric setting
        rpe_metric = metrics.RPE(pose_relation=pose_relation, delta=self.delta,
                                delta_unit=delta_unit, all_pairs=all_pairs)
        # calculate the error
        rpe_metric.process_data(data)
        # devided by the subjectory length --> invariant to the length
        error = np.array(rpe_metric.error)#/float(self.delta) 

        # assign time stamps to the error
        timeStamps = self.traj_ref.timestamps[self.delta:]
        assert(len(timeStamps==len(error)))
        self.error_df = pd.DataFrame({'TimeStamp': timeStamps, 'RelativeError': error})
        print("Loaded trajectory with ground truth")
        print("="*50)
        return rpe_metric

    # Interpolate consecutive null values up to max_null_length
    @staticmethod
    def interpolate_consecutive_nulls(column_df, max_null_length=10):
        column = copy.deepcopy(column_df)
        is_null = column.isnull()
        consecutive_nulls = 0
        for i in range(len(column)):
            if is_null[i]:
                consecutive_nulls += 1
            else:
                # igonre long null sequence
                if consecutive_nulls >= max_null_length:
                    consecutive_nulls = 0
                elif consecutive_nulls > 0:
                    # Interpolate using linear method for consecutive null values
                    column[i - consecutive_nulls:i] = np.linspace(column[i - consecutive_nulls - 1], column[i], consecutive_nulls + 2)[1:-1]
                consecutive_nulls = 0
        return column


    def merge_feature_with_label(self):
        # load feature
        feature_df = pd.read_csv("{}/logs/log.csv".format(self.root_dir))

        # Step 1: Check duplicated/zero time stamp
        for index, value in enumerate(feature_df['TimeStamp'][1:-1], start=1):
            if(value == feature_df.at[index+1, 'TimeStamp'] or value == 0 or value < feature_df.at[index-1, 'TimeStamp']):
                feature_df.at[index, 'TimeStamp'] = 0.5*(feature_df.at[index-1, 'TimeStamp'] + feature_df.at[index+1, 'TimeStamp'])
        #assert(feature_df["TimeStamp"].is_monotonic_increasing)
        #assert(feature_df["TimeStamp"].is_unique)

        error_df = self.error_df
        # merge feature
        merged_df = pd.merge_asof(feature_df, error_df , on='TimeStamp', tolerance=self.max_diff)
        # move label to the front
        # Column to move to the first position
        column_to_move = 'RelativeError'
        # Reorder the columns
        new_columns = [column_to_move] + [col for col in merged_df.columns if col != column_to_move]
        merged_df = merged_df[new_columns]
        merged_df["RelativeError"] = PoseErrorEvaluator.interpolate_consecutive_nulls(merged_df["RelativeError"], 
                                                                                      self.max_null_length)
        self.merged_df = merged_df

    def get_traj_w_gt(self):
        #return self.traj_est_aligned, self.traj_ref
        return self.traj_est, self.traj_ref

    def get_feature_w_label(self):
        return self.merged_df
    
    def get_error_df(self):
        return self.error_df

    @staticmethod
    def attach_label_2_features(delta, error, df):
        new_column_name = 'RelativeError'
        df_label = pd.DataFrame({new_column_name: error})
        df = df.iloc[delta:].reset_index(drop=True)
        if(len(df) != len(df_label)):
            print(len(df))
            print(len(df_label))
            diff = len(df) - len(df_label)
            df = df.iloc[diff:]
            assert(diff <= 30)
        data_df = pd.concat([df_label, df], ignore_index=True, axis=1)
        return data_df

    @staticmethod
    def plot_aligned_trajectory(traj_est, traj_ref, benchmark=None, trajectory=None, trial=None):
        fig = plt.figure(figsize=[10,10])
        
        traj_est_aligned = copy.deepcopy(traj_est)
        n = int(traj_est_aligned.timestamps.shape[0]/2)
        print("aligned length: {}".format(n))
        traj_est_aligned.align(traj_ref, correct_scale=True, correct_only_scale=False, n=n)
        
        traj_by_label = {
            #"estimate (not aligned)": traj_est,
            "estimate (aligned)": traj_est_aligned,
            "reference": traj_ref
        }
        
        plot.trajectories(fig, traj_by_label, plot.PlotMode.xyz)
        fig.savefig('./figures/{}-{}-{}-trajectory.png'.format(benchmark, trajectory, trial))

    @staticmethod
    def plot_trajectory(traj_est, traj_ref, benchmark=None, trajectory=None, trial=None):
        fig = plt.figure(figsize=[10,10])

        traj_est_aligned = copy.deepcopy(traj_est)
        
        traj_by_label = {
            #"estimate (not aligned)": traj_est,
            "estimate (aligned)": traj_est_aligned,
            "reference": traj_ref
        }

        plot.trajectories(fig, traj_by_label, plot.PlotMode.xyz)
        fig.savefig('./figures/{}-{}-{}-trajectory.png'.format(benchmark, trajectory, trial))
    
    
    @staticmethod
    def plot_error(error_df, delta, benchmark=None, trajectory=None, trial=None, plot_error_min = -0.05, plot_error_max=0.75):
        fig = plt.figure(figsize=[10,3])
        # Plot the line
        #plt.plot(error_df['TimeStamp'], error_df['RelativeError'])
        plt.plot(error_df['RelativeError'])
        plt.ylim(plot_error_min, plot_error_max)
        # Add labels and title
        plt.xlabel('Time Step')
        plt.ylabel('Relative Error')
        plt.title('Relative Error on {}-{}-{} with sub-trajectory={} frames'.format(benchmark, 
                                                                                        trajectory, 
                                                                                        trial, 
                                                                                        delta))

        fig.savefig('./figures/{}-{}-{}-error.png'.format(benchmark, trajectory, trial))

    @staticmethod
    def plot_trajectory_with_error(rpe_metric, traj_ref, traj_est_aligned, benchmark, trajectory, trial, plot_colormap_max = 0.1):
        result = rpe_metric.get_result()
        
        plot_mode = plot.PlotMode(plot.PlotMode.xy)
        # Plot the values color-mapped onto the trajectory.
        fig = plt.figure(figsize=(10,10))
        ax = plot.prepare_axis(
            fig, plot_mode,
            length_unit=Unit(SETTINGS.plot_trajectory_length_unit))
        plot.traj(ax, plot_mode, traj_ref,
              style=SETTINGS.plot_reference_linestyle,
              color=SETTINGS.plot_reference_color, label='reference',
              alpha=SETTINGS.plot_reference_alpha,
              plot_start_end_markers=SETTINGS.plot_start_end_markers)
        plot.draw_coordinate_axes(ax, traj_ref, plot_mode,
                              SETTINGS.plot_reference_axis_marker_scale)
    
        # plot_colormap_min = result.stats["min"]
        # plot_colormap_max = result.stats["max"]
        # plot_colormap_max = np.percentile(
        #     result.np_arrays["error_array"], 100)     
        # plot_colormap_max = np.percentile(
        #     result.np_arrays["error_array"], 75)
        plot_colormap_min = 0      

        plot.traj_colormap(ax, traj_est_aligned, result.np_arrays["error_array"],
                       plot_mode, min_map=plot_colormap_min,
                       max_map=plot_colormap_max,
                       title=result.info["title"],
                       plot_start_end_markers=SETTINGS.plot_start_end_markers)

        plot.draw_coordinate_axes(ax, traj_est_aligned, plot_mode,
                                SETTINGS.plot_axis_marker_scale)
        fig.savefig('./figures/{}-{}-{}-trajectory-error.png'.format(benchmark, trajectory, trial))

