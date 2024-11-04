import os
import shutil

import copy
import time

import subprocess

def copy_file_and_rename(source_file_path, destination_folder_path, new_filename):
    try:
        # Check if the source file exists
        if not os.path.exists(source_file_path):
            raise FileNotFoundError(f"The source file '{source_file_path}' does not exist.")

        # Create the destination folder if it doesn't exist
        if not os.path.exists(destination_folder_path):
            os.makedirs(destination_folder_path)

        # Create the destination file path by joining the destination folder and new filename
        destination_file_path = os.path.join(destination_folder_path, new_filename)

        # Copy the file from the source path to the destination path
        shutil.copy2(source_file_path, destination_file_path)

        print(f"File copied successfully and renamed to '{new_filename}' in '{destination_folder_path}'.")
    except Exception as e:
        print(f"Error: {e}")


def delete_files_in_folder(folder_path, b_delete_folder=False):
    try:
        # List all files in the folder
        files = os.listdir(folder_path)

        # Loop through the files and delete them one by one
        for file in files:
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
                #print(f"Deleted: {file_path}")
            else:
                if b_delete_folder:
                    delete_folder(file_path)
                else:
                    print(f"Skipping: {file_path} (It is not a file)")


        print(f"All files in {folder_path} deleted successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")


def delete_folder(folder_path):
    # Check if the folder exists
    if os.path.exists(folder_path):
        try:
            # Use shutil.rmtree to delete the folder and its contents
            shutil.rmtree(folder_path)
            print(f"Successfully deleted {folder_path}")
        except Exception as e:
            print(f"Error deleting {folder_path}: {str(e)}")
    else:
        print(f"The folder {folder_path} does not exist.")

def copy_folder(source, target):
    os.rename(source, target)


def attach_label_2_features(delta, error, df):
    new_column_name = 'RelativeError'
    df_label = pd.DataFrame({new_column_name: error})
    df = df.iloc[delta:].reset_index(drop=True)
    assert(len(df) == len(df_label))
    data_df = pd.concat([df_label, df], axis=1)
    return data_df


def runCommand(consoleCommand, consoleOutputEncoding="utf-8", timeout=240):
    """get command output from terminal
    Args:
        consoleCommand (str): console/terminal command string
        consoleOutputEncoding (str): console output encoding, default is utf-8
        timeout (int): wait max timeout for run console command
    Returns:
        console output (str)
    Raises:
    """
    # print("getCommandOutput: consoleCommand=%s" % consoleCommand)
    isRunCmdOk = False
    consoleOutput = ""
    try:
        # consoleOutputByte = subprocess.check_output(consoleCommand)
        consoleOutputByte = subprocess.check_output(consoleCommand, shell=True, timeout=timeout)

        consoleOutput = consoleOutputByte.decode(consoleOutputEncoding) # '640x360\n'
        consoleOutput = consoleOutput.strip() # '640x360'
        isRunCmdOk = True
    except subprocess.CalledProcessError as callProcessErr:
        cmdErrStr = str(callProcessErr)
        print("Error %s for run command %s" % (cmdErrStr, consoleCommand))

    # print("isRunCmdOk=%s, consoleOutput=%s" % (isRunCmdOk, consoleOutput))
    return isRunCmdOk, consoleOutput