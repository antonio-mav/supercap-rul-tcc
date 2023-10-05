from pathlib import Path
import multiprocessing
import math
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from tqdm import tqdm


def main():
    raw_dir = Path.cwd() / "data" / "raw"
    raw_file_paths = dict_to_list_of_tuples(get_file_paths(raw_dir, "xls"))

    processed_dir = Path.cwd() / "data" / "processed"
    with multiprocessing.Pool(multiprocessing.cpu_count()-1) as pool:
        for _ in tqdm(pool.imap(load_data_unpack, raw_file_paths), total=len(raw_file_paths)):
            pass

    generate_dataset_info(processed_dir)


def get_batch_idx(file_path):
    return int(str(file_path.parents[0])[-1])


def get_cap_idx(file_path):
    stem = str(file_path.stem)
    if "__" not in stem:
        return int(stem)
    else:
        return int(stem.split("__", maxsplit=1)[0])


def dict_to_list_of_tuples(dictionary):
    list_of_tuples = []
    for batch_idx, batch_caps in dictionary.items():
        for cap_idx, cap_paths in batch_caps.items():
            list_of_tuples.append((batch_idx, cap_idx, cap_paths))

    return list_of_tuples


def get_file_paths(root_dir, file_type):
    file_paths_dict = {}

    for file_path in sorted(root_dir.rglob("*." + file_type)):
        # Get relevant indexes
        batch_idx = get_batch_idx(file_path)
        cap_idx = get_cap_idx(file_path)

        # Create keys if they don't exist
        if batch_idx not in file_paths_dict:
            file_paths_dict[batch_idx] = {}
        if cap_idx not in file_paths_dict[batch_idx]:
            file_paths_dict[batch_idx][cap_idx] = []

        # Append file path to list
        file_paths_dict[batch_idx][cap_idx].append(file_path)

    return file_paths_dict


def time_str_to_seconds(time_str):
    hours, minutes, seconds = [float(value) for value in time_str.split(":")]
    return hours*3600 + minutes*60 + seconds


def replace_inf(array):
    idxs = np.where(np.isinf(array))[0]
    next_values = np.roll(array, -1)
    array[idxs] = next_values[idxs]
    return array


def get_start_cycle(batch_idx):
    if batch_idx == 1:
        start_cycle = 3
    else:
        start_cycle = 2
    return start_cycle


def calculate_capacitances(cap_df):
    discharge_cycle = cap_df[cap_df["status"] == 0].groupby("cycle")
    discharge_current = discharge_cycle["current(mA)"].mean() * -1e-3
    voltage_variation = discharge_cycle["voltage(V)"].first() - discharge_cycle["voltage(V)"].last()
    discharge_time = discharge_cycle["record_time(s)"].last()
    capacitances = discharge_current*discharge_time / voltage_variation

    if np.any(np.isinf(capacitances)):
        capacitances = replace_inf(capacitances.to_numpy())

    return capacitances


def calculate_ruls(capacitances, batch_idx, boundary):
    ruls = []
    crossing_idx = np.where(capacitances <= boundary)[0][0] + 1

    start_cycle = get_start_cycle(batch_idx) - 1
    end_cycle = len(capacitances) + 1

    for curr_cycle in range(start_cycle, end_cycle):
        rul = crossing_idx - curr_cycle
        if rul >= 0:
            ruls.append(rul)

    start_idx = get_start_cycle(batch_idx) - 1

    return np.array(ruls)[start_idx:]


def split_reseting_array(array, ref_array, reset_value=0):
    assert len(array) == len(ref_array)
    reset_idxs = np.where(ref_array == reset_value)[0]
    arrays = np.split(array, reset_idxs[1:])

    return arrays


def cumulative_merge_arrays(arrays):
    cum_arrays = []
    
    cum_sum = 0
    for array in arrays:
        cum_arrays.append(array + cum_sum)
        cum_sum += array[-1]

    return np.concatenate(cum_arrays)


def linear_interpolation(x_in, y_in, lower_bound, upper_bound, num):
    # Flip input sequences if they are in descending order
    if x_in[0] > x_in[-1]:
        x_in = np.flip(x_in)
        y_in = np.flip(y_in)
 
    # Interpolate an ascending sequence using a set number of points
    interp_x_in = np.linspace(lower_bound, upper_bound, num=num)
    interp_y_in = np.interp(interp_x_in, x_in, y_in)
    
    return interp_y_in


def preprocess_capacities(cap_df, batch_idx, useful_life, pts_per_step=32):
    capacities = []

    start_cycle = get_start_cycle(batch_idx)
    end_cycle = useful_life + start_cycle
    useful_cap_df = cap_df[(cap_df["cycle"] >= start_cycle) & (cap_df["cycle"] <= end_cycle)]

    for _, cycle in useful_cap_df.groupby("cycle"):
        charge_capacities = None
        discharge_capacities = None
        for _, data in cycle.groupby("status"):
            # True if current data group corresponds to a charge cycle
            charging = np.all(data["status"] == 1)

            if batch_idx == 2 and charging:
                # Merge reseting capacities from batch 2 to a single one
                charging_steps = split_reseting_array(data["capacity(mAh)"].to_numpy(), data["record_time(s)"].to_numpy())
                step_capacities = cumulative_merge_arrays(charging_steps)
            else:
                # Get single capacity curve otherwise
                step_capacities = data["capacity(mAh)"].to_numpy()

            # Interpolate capacity curves
            voltages = data["voltage(V)"].to_numpy()
            if charging:
                charge_capacities = linear_interpolation(voltages, step_capacities, 1, 2.7, pts_per_step)
            else:
                discharge_capacities = linear_interpolation(voltages, step_capacities, 1, 2.7, pts_per_step)

        # Join capacities
        interp_capacities = [charge_capacities, discharge_capacities]
        capacities.extend(interp_capacities)

    return np.concatenate(capacities).reshape((-1, pts_per_step*2))

def preprocessing_pipeline_intermediate(batch_idx, cap_idx, cap_paths):
    cap_dfs = []
    for path in cap_paths:
        sec_sheets = pd.ExcelFile(path)
        sheet_names = [name for name in sec_sheets.sheet_names if "record" in name]
        for sheet_name in sheet_names:
            # Load relevant sheets and columns of the .xls files as dataframes
            sheet = sec_sheets.parse(sheet_name)
            sheet["record_time(s)"] = sheet["record_time(h:min:s.ms)"].transform(time_str_to_seconds)
            cap_dfs.append(sheet.drop(columns=["step", "record_number", "record_time(h:min:s.ms)"]))

    cap_dataframe = pd.concat(cap_dfs, ignore_index=True)
    cap_dataframe.to_pickle(path=f"./data/intermediate/b{batch_idx}c{cap_idx}.tar.gz")


def preprocessing_pipeline(batch_idx, cap_idx, cap_paths):
    cap_dfs = []
    for path in cap_paths:
        sec_sheets = pd.ExcelFile(path)
        sheet_names = [name for name in sec_sheets.sheet_names if "record" in name]
        for sheet_name in sheet_names:
            # Load relevant sheets and columns of the .xls files as dataframes
            sheet = sec_sheets.parse(sheet_name)
            sheet["record_time(s)"] = sheet["record_time(h:min:s.ms)"].transform(time_str_to_seconds)
            cap_dfs.append(sheet.drop(columns=["step", "record_number", "record_time(h:min:s.ms)"]))

    raw_cap_df = pd.concat(cap_dfs, ignore_index=True)

    # Preprocess variables that will be used in final model
    capacitances = savgol_filter(calculate_capacitances(raw_cap_df), 25, 1)
    ruls = calculate_ruls(capacitances, batch_idx, 0.9033)
    capacities = preprocess_capacities(raw_cap_df, batch_idx, ruls[0], pts_per_step=16)
    start_idx = get_start_cycle(batch_idx) - 1
    end_idx = start_idx + len(ruls)

    # Save data as a compressed DataFrame
    processed_cap_data = {"capacity": capacities.tolist(), "rul": ruls, "capacitance": capacitances[start_idx:end_idx], "curr_cycle": np.arange(start_idx, end_idx)}
    processed_cap_df = pd.DataFrame(data=processed_cap_data)
    processed_cap_df.to_pickle(path=f"./data/processed/b{batch_idx}c{cap_idx}.tar.gz")


def load_data_unpack(args):
    preprocessing_pipeline(*args)


def generate_dataset_info(root_dir):
    dataset_info = {"file_name":[], "useful_life": []}

    files = list(root_dir.rglob("*.tar.gz"))
    for file in files:
        dataset_info["file_name"].append(file.name)
        dataset_info["useful_life"].append(pd.read_pickle(file)["rul"][0])

    pd.DataFrame(dataset_info).to_csv(root_dir / "dataset_info.csv", index=False)


if __name__ == "__main__":
    main()
