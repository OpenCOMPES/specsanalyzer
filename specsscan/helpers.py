"""This script contains helper functions used by the specscan class"""
import datetime as dt
import json
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Sequence
from typing import Tuple
from typing import Union
from urllib.error import HTTPError
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np
import pandas as pd
from specsanalyzer.settings import insert_default_config  # name can be generalized
from tqdm.auto import tqdm


def load_images(
    scan_path: Path,
    df_lut: Union[pd.DataFrame, None] = None,
    iterations: Union[
        np.ndarray,
        slice,
        Sequence[int],
        Sequence[slice],
    ] = None,
    delays: Union[
        np.ndarray,
        slice,
        int,
        Sequence[int],
        Sequence[slice],
    ] = None,
    tqdm_enable_nested: bool = False,
) -> np.ndarray:
    """Loads a 2D/3D numpy array of images for the given
        scan path with an optional averaging
        over the given iterations/delays. The function provides
        functionality to both load_scan and check_scan methods of
        the SpecsScan class. When iterations/delays is provided,
        average is performed over the iterations/delays for all
        delays/iterations.
    Args:
        scan_path: object of class Path pointing
                to the scan folder
        df_lut: Pandas dataframe containing the contents of LUT.txt
                as obtained from parse_lut_to_df()
        iterations: A 1-D array of the indices of iterations over
                which the images are to be averaged. The array
                can be a list, numpy array or a Tuple consisting of
                slice objects and integers. For ex.,
                np.s_[1:10, 15, -1] would be a valid input.
        delays: A 1-D array of the indices of delays over
                which the images are to be averaged. The array can
                be a list, numpy array or a Tuple consisting of
                slice objects and integers. For ex.,
                np.s_[1:10, 15, -1] would be a valid input.
    Returns:
        data: A 2-D or 3-D (concatenated) numpy array consisting
            of raw data
    """

    scan_list = sorted(
        file.stem
        for file in scan_path.joinpath("AVG").iterdir()
        if file.suffix == ".tsv"
    )

    data = []

    if iterations is not None or delays is not None:

        avg_dim = "iterations" if iterations is not None else "delays"

        if df_lut is not None:
            raw_array = df_lut["filename"].to_numpy()
        else:
            raw_gen = scan_path.joinpath("RAW").glob("*.tsv")
            raw_array = np.array(
                [file.stem + ".tsv" for file in raw_gen],
            )

        raw_2d = get_raw2d(
            scan_list,
            raw_array,
        )

        # Slicing along the given iterations or delays
        try:
            if avg_dim == "delays":
                raw_2d_sliced = raw_2d[:, np.r_[delays]]
            else:  # iterations is not None
                if delays is not None:
                    raise ValueError(
                        "Invalid input. One of either iterations or"
                        "delays is expected, both were provided.",
                    )
                raw_2d_sliced = raw_2d[np.r_[iterations]].T

        except IndexError as exc:
            raise IndexError(
                f"Invalid {avg_dim} for "
                "the chosen data. In case of a single scan, "
                f"try without passing iterations inside the "
                "load_scan method.",
            ) from exc

        print(f"Averaging over {avg_dim}...")
        for dim in tqdm(raw_2d_sliced):
            avg_list = []
            for image in tqdm(dim, leave=False, disable=not tqdm_enable_nested):
                if image != "nan":

                    with open(
                        scan_path.joinpath(f"RAW/{image}"),
                        encoding="utf-8",
                    ) as file:
                        new_im = np.loadtxt(file, delimiter="\t")
                        avg_list.append(new_im)

            data.append(
                np.average(
                    np.array(avg_list),
                    axis=0,
                ),
            )
    else:
        for image in tqdm(scan_list):
            with open(
                scan_path.joinpath(
                    f"AVG/{image}.tsv",
                ),
                encoding="utf-8",
            ) as file:

                new_im = np.loadtxt(file, delimiter="\t")
                data.append(new_im)

    return np.array(data)


def get_raw2d(
    scan_list: List[str],
    raw_array: np.ndarray,
) -> np.ndarray:
    """Converts a 1-D array of raw scan names
        into 2-D based on the number of iterations
    Args:
        scan_list: A list of AVG scan names.
        raw_list: 1-D array of RAW scan names.
    Returns:
        raw_2d: 2-D numpy array of size for ex.,
            (total_iterations, delays) for a delay scan.
    """

    total_iterations = len(
        [im for im in raw_array if f"{scan_list[0]}_" in im],
    )

    delays = len(scan_list)
    diff = delays * (total_iterations) - len(raw_array)

    if diff:  # Ongoing or aborted scan
        diff = delays - diff  # Number of scans in the last iteration
        raw_2d = raw_array[:-diff].reshape(
            total_iterations - 1,
            delays,
        )

        last_iter_array = np.full(
            (1, delays),
            fill_value="nan",
            dtype="object",
        )

        last_iter_array[0, :diff] = raw_array[-diff:]
        raw_2d = np.concatenate(
            (raw_2d, last_iter_array),
        )
    else:  # Complete scan
        raw_2d = raw_array.reshape(total_iterations, delays)

    return raw_2d


def parse_lut_to_df(scan_path: Path) -> Union[pd.DataFrame, None]:
    """Loads the contents of LUT.txt file into a pandas
        data frame to be used as metadata.
    Args:
        scan_path: Path object for the scan path
    Returns: A pandas DataFrame
    """
    try:
        df_lut = pd.read_csv(scan_path.joinpath("RAW/LUT.txt"), sep="\t")
        df_lut.reset_index(inplace=True)

        new_cols = df_lut.columns.to_list()[1:]
        new_cols[new_cols.index("delaystage")] = "Delay"
        new_cols.insert(3, "delay (fs)")  # Create label to drop the column later

        df_lut.columns = new_cols
        df_lut.drop(columns="delay (fs)", inplace=True)

    except FileNotFoundError:
        print(
            "LUT.txt not found. Storing metadata from info.txt",
        )
        return None

    return df_lut


def get_coords(
    scan_path: Path,
    scan_type: str,
    scan_info: Dict[Any, Any],
    df_lut: Union[pd.DataFrame, None] = None,
) -> Tuple[np.ndarray, str]:
    """Reads the contents of scanvector.txt file
        into a numpy array.
    Args:
        scan_path: Path object for the scan path
        scan_type: Type of scan (delay, mirror etc.)
        scan_info: scan_info class dict
        df_lut: df_lut: Pandas dataframe containing
                the contents of LUT.txt as obtained
                from parse_lut_to_df()
    Raises:
        FileNotFoundError
    Returns:
        coords: 1-D numpy array containing coordinates
                of the scanned axis.
        dim: string containing the name of the coordinate
    """
    try:
        with open(
            scan_path.joinpath("scanvector.txt"),
            encoding="utf-8",
        ) as file:
            data = np.loadtxt(file, ndmin=2)

        coords, index = compare_coords(data)
        if scan_type == "mirror":
            dim = ["mirrorX", "mirrorY"][index]
        elif scan_type == "manipulator":
            dim = [
                "X",
                "Y",
                "Z",
                "polar",
                "tilt",
                "azimuth",
            ][index]
        else:
            dim = scan_type

    except FileNotFoundError as exc:
        if scan_type == "single":
            return (np.array([]), "")

        if df_lut is not None:
            print(
                "scanvector.txt not found. Obtaining coordinates from LUT",
            )

            df_new: pd.DataFrame = df_lut.loc[:, df_lut.columns[2:]]

            coords, index = compare_coords(df_new.to_numpy())
            dim = df_new.columns[index]

        else:
            raise FileNotFoundError(
                "scanvector.txt file not found!",
            ) from exc

    if scan_type == "delay":
        t_0 = scan_info["TimeZero"]
        coords -= t_0
        coords *= 2 / (3 * 10**11) * 10**15

    return coords, dim


def compare_coords(
    axis_data: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """To check the most changing column in a given
        2-D numpy array.
    Args:
        axis_data: 2-D numpy array containing LUT data
    Returns:
        coords: Maximum changing column as a coordinate
        index: Index of the coords in the axis_data array
    """

    diff_list = [abs(col[-1] - col[0]) for col in axis_data.T]

    index = diff_list.index(max(diff_list))

    if max(diff_list) == 0:
        raise ValueError("Coordinates not found in LUT.")

    coords = axis_data[:, index]
    return coords, index


def parse_info_to_dict(path: Path) -> Dict:
    """Parses the contents of info.txt file
        into a dictionary
    Args:
        path: Path object pointing to the scan folder
    Returns:
        info_dict: Parsed dictionary
    """
    info_dict: Dict[Any, Any] = {}
    try:
        with open(path.joinpath("info.txt"), encoding="utf-8") as info_file:

            for line in info_file.readlines():

                if "=" in line:  # older scans
                    line_list = line.rstrip("\nV").split("=")

                elif ":" in line:
                    line_list = line.rstrip("\nV").split(":")

                else:
                    continue

                key, value = line_list[0], line_list[1]

                try:
                    info_dict[key] = float(value)
                except ValueError:
                    info_dict[key] = value

    except FileNotFoundError as exc:
        raise FileNotFoundError("info.txt file not found.") from exc

    return info_dict


def handle_meta(  # pylint:disable=too-many-branches
    df_lut: pd.DataFrame,
    scan_info: dict,
    config: dict,
) -> dict:
    """Helper function for the handling metadata from different files
    Args:
        df_lut: Pandas dataframe containing
                the contents of LUT.txt as obtained
                from parse_lut_to_df()
        scan_info: scan_info class dict containing
                containing the contents of info.txt file
        config: config dictionary containing the contents
                of config.yaml file
    Returns:
        metadata_dict: metadata dictionary containing additional metadata
                from the EPICS archive.
    """
    metadata_dict = {}
    # get metadata from LUT dataframe
    lut_meta = {}
    if df_lut is not None:
        for col in df_lut.columns:
            col_array = df_lut[f'{col}'].to_numpy()
            if len(set(col_array)) == 1:
                lut_meta[col] = col_array[0]
            else:
                lut_meta[col] = col_array

    scan_meta = insert_default_config(lut_meta, scan_info)  # merging two dictionaries

    replace_dict = config["epics_channels"]
    for key in list(scan_meta):
        if key.lower() in replace_dict:
            scan_meta[replace_dict[key.lower()]] = scan_meta[key]
            scan_meta.pop(key)

    metadata_dict["scan_info"] = scan_meta
    # Get metadata from Epics archive, if not present already
    print("Collecting data from the EPICS archive...")
    if "time" in scan_meta:
        time_list = [scan_meta["time"][0], scan_meta["time"][-1]]
    elif "StartTime" in scan_meta:
        time_list = [scan_meta["StartTime"]]
    else:
        raise ValueError("Could not find timestamps in scan info.")

    dt_list_iso = [time.replace('.', '-').replace(' ', 'T') for time in time_list]
    datetime_list = [dt.datetime.fromisoformat(dt_iso) for dt_iso in dt_list_iso]
    ts_from = dt.datetime.timestamp(datetime_list[0])  # POSIX timestamp
    ts_to = dt.datetime.timestamp(datetime_list[-1])  # POSIX timestamp
    metadata_dict['timing'] = {
        'acquisition_start': dt.datetime.utcfromtimestamp(ts_from).replace(
            tzinfo=dt.timezone.utc,
        ).isoformat(),
        'acquisition_stop': dt.datetime.utcfromtimestamp(ts_to).replace(
            tzinfo=dt.timezone.utc,
        ).isoformat(),
        'acquisition_duration': int(ts_to - ts_from),
        'collection_time': float(ts_to - ts_from),
    }
    filestart = dt.datetime.utcfromtimestamp(ts_from).isoformat()  # Epics time in UTC?
    fileend = dt.datetime.utcfromtimestamp(ts_to).isoformat()
    epics_channels = replace_dict.values()

    channels_missing = set(epics_channels) - set(metadata_dict['scan_info'].keys())
    for channel in channels_missing:
        try:
            req_str = "http://aa0.fhi-berlin.mpg.de:17668/retrieval/" + \
                "data/getData.json?pv=" + \
                channel + "&from=" + filestart + "Z&to=" + fileend + "Z"
            with urlopen(req_str) as req:
                data = json.load(req)
                vals = [x['val'] for x in data[0]['data']]
                metadata_dict["scan_info"][f'{channel}'] = sum(vals) / len(vals)
        except (IndexError, ZeroDivisionError):
            metadata_dict["scan_info"][f'{channel}'] = np.nan
            print(f"Data for channel {channel} doesn't exist for time {filestart}")
        except HTTPError as error:
            print(
                f"Incorrect URL for the archive channel {channel}. "
                "Make sure that the channel name, file start "
                "and end times are correct.",
            )
            print("Error code: ", error)
        except URLError as error:
            print(
                f"Cannot access the archive URL for channel {channel}. "
                f"Make sure that you are within the FHI network."
                f"Skipping over channels {channels_missing}.",
            )
            print("Error code: ", error)
            break
    print("Done!")

    return metadata_dict


def find_scan(path: Path, scan: int) -> List[Path]:
    """Search function to locate the scan folder
    Args:
        path: Path object for data from the default config file
        scan: Scan number of the scan of interest
    Returns:
        scan_path: Path object pointing to the scan folder
    """
    print("Scan path not provided, searching directories...")
    for file in path.iterdir():

        if file.is_dir():

            try:
                base = int(file.stem)

            except ValueError:  # not numeric
                continue

            if base >= 2019:  # only look at folders 2019 onwards

                scan_path = sorted(
                    file.glob(f"*/*/Raw Data/{scan}"),
                )
                if scan_path:
                    print("Scan found at path:", scan_path[0])
                    break
    else:
        scan_path = []
    return scan_path


def find_scan_type(  # pylint:disable=too-many-nested-blocks
    path: Path,
    scan_type: str,
) -> None:
    """Rudimentary function to print scan paths given the scan_type
    Args:
        path: Path object pointing to the year, for ex.,
            Path("//nap32/topfloor/trARPES/PESData/2020")
        scan_type: string containing the scan_type from the list
            ["delay","temperature","manipulator","mirror","single"]
    Returns:
        None
    """

    if scan_type not in [
        "delay",
        "temperature",
        "manipulator",
        "mirror",
        "single",
    ]:
        print("Invalid scan type!")
        return None

    for month in path.iterdir():
        if month.is_dir():
            for day in month.iterdir():
                if day.is_dir():
                    try:
                        for scan_path in day.joinpath("Raw Data").iterdir():

                            stype = parse_info_to_dict(scan_path)["ScanType"]
                            if stype == scan_type:
                                print(scan_path)
                    except (FileNotFoundError, NotADirectoryError):
                        pass
    return None
