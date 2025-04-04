"""This script contains helper functions used by the specsscan class"""
from __future__ import annotations

import datetime as dt
import importlib
import logging
from pathlib import Path
from typing import Any
from typing import Sequence

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from specsanalyzer.config import complete_dictionary
from specsscan.metadata import MetadataRetriever

# Configure logging
logger = logging.getLogger("specsanalyzer.specsscan")


def get_scan_path(path: Path | str, scan: int, basepath: Path | str) -> Path:
    """Returns the path to the given scan.

    Args:
        path (Path | str): Path under which to search. If empty, the basepath will be queried
        scan (int): Scan number
        basepath (Path | str): Default base path to search for scans under

    Raises:
        FileNotFoundError: Raised if the path or scan cannot be found.

    Returns:
        Path: Path object to the given scan directory
    """
    if path:
        path = Path(path).joinpath(str(scan).zfill(4))
        if not path.is_dir():
            raise FileNotFoundError(
                f"The provided path {path} was not found.",
            )
    else:
        # search for the given scan using the default path
        path = Path(basepath)
        path_scan_list = find_scan(path, scan)
        if not path_scan_list:
            raise FileNotFoundError(
                f"Scan number {scan} not found",
            )
        path = path_scan_list[0]

    return path


def load_images(
    scan_path: Path,
    df_lut: pd.DataFrame = None,
    iterations: np.ndarray | slice | Sequence[int] | Sequence[slice] = None,
    delays: np.ndarray | slice | int | Sequence[int] | Sequence[slice] = None,
    tqdm_enable_nested: bool = False,
) -> list[np.ndarray]:
    """Loads a 2D/3D numpy array of images for the given scan path with an optional averaging
    over the given iterations/delays. The function provides functionality to both load_scan
    and check_scan methods of the SpecsScan class. When iterations/delays is provided,
    average is performed over the iterations/delays for all delays/iterations.

    Args:
        scan_path (Path): object of class Path pointing to the scan folder
        df_lut (pd.DataFrame, optional): Pandas dataframe containing the contents of LUT.txt as
            obtained from parse_lut_to_df(). Defaults to None.
        iterations (np.ndarray | slice | Sequence[int] | Sequence[slice], optional): A 1-D
            array of the indices of iterations over which the images are to be averaged. The array
            can be a list, numpy array or a Tuple consisting of slice objects and integers. For
            ex., ``np.s_[1:10, 15, -1]`` would be a valid input. Defaults to None.
        delays (np.ndarray | slice | int | Sequence[int] | Sequence[slice], optional): A 1-D
            array of the indices of delays over which the images are to be averaged. The array can
            be a list, numpy array or a Tuple consisting of slice objects and integers. For ex.,
            ``np.s_[1:10, 15, -1]`` would be a valid input. Defaults to None.
        tqdm_enable_nested (bool, optional): Option to enable a nested progress bar.
            Defaults to False.

    Raises:
        ValueError: Raised if both iterations and delays is provided.
        IndexError: Raised if no valid dimension for averaging is found.

    Returns:
        list[np.ndarray]: A list of 2-D numpy arrays of raw data
    """
    scan_list = sorted(
        file.stem for file in scan_path.joinpath("AVG").iterdir() if file.suffix == ".tsv"
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

        logger.info(f"Averaging over {avg_dim}...")
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

    return data


def get_raw2d(scan_list: list[str], raw_array: np.ndarray) -> np.ndarray:
    """Converts a 1-D array of raw scan names into 2-D based on the number of iterations

    Args:
        scan_list (list[str]): A list of AVG scan names.
        raw_list (np.ndarray): 1-D array of RAW scan names.

    Returns:
        np.ndarray: 2-D numpy array of size for ex., (total_iterations, delays) for a delay scan.
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


def parse_lut_to_df(scan_path: Path) -> pd.DataFrame:
    """Loads the contents of LUT.txt file into a pandas data frame to be used as metadata.

    Args:
        scan_path (Path): Path object for the scan path

    Returns:
        pd.DataFrame: A pandas DataFrame
    """
    try:
        df_lut = pd.read_csv(scan_path.joinpath("RAW/LUT.txt"), sep="\t")
        df_lut.reset_index(inplace=True)

        new_cols = df_lut.columns.to_list()[1:]
        new_cols[new_cols.index("delaystage")] = "DelayStage"
        new_cols.insert(3, "delay (fs)")  # Create label to drop the column later

        df_lut = df_lut.set_axis(new_cols, axis="columns")
        df_lut.drop(columns="delay (fs)", inplace=True)

    except FileNotFoundError:
        logger.info(
            "LUT.txt not found. Storing metadata from info.txt",
        )
        return None

    return df_lut


def get_coords(
    scan_path: Path,
    scan_type: str,
    scan_info: dict[Any, Any],
    df_lut: pd.DataFrame = None,
) -> tuple[np.ndarray, str]:
    """Reads the contents of scanvector.txt file into a numpy array.

    Args:
        scan_path (Path): Path object for the scan path
        scan_type (str): Type of scan (delay, mirror etc.)
        scan_info (dict[Any, Any]): scan_info class dict
        df_lut (pd.DataFrame, optional): Pandas dataframe containing the contents of LUT.txt as
            obtained from parse_lut_to_df(). Defaults to None.

    Raises:
        FileNotFoundError: Raised in neither scanvector.txt nor LUT.txt are found.

    Returns:
        tuple[np.ndarray, str]:
            - coords: 1-D numpy array containing coordinates of the scanned axis.
            - dim: string containing the name of the coordinate
    """
    try:
        with open(scan_path.joinpath("scanvector.txt"), encoding="utf-8") as file:
            data = np.loadtxt(file, ndmin=2)

        coords, index = compare_coords(data)
        if scan_type == "mirror":
            dim = ["mirrorX", "mirrorY"][index]
        elif scan_type == "manipulator":
            dim = ["X", "Y", "Z", "polar", "tilt", "azimuth"][index]
        else:
            dim = scan_type

    except FileNotFoundError as exc:
        if scan_type == "single":
            return (np.array([]), "")

        if df_lut is not None:
            logger.info("scanvector.txt not found. Obtaining coordinates from LUT")

            df_new: pd.DataFrame = df_lut.loc[:, df_lut.columns[2:]]

            coords, index = compare_coords(df_new.to_numpy())
            dim = df_new.columns[index]

        else:
            raise FileNotFoundError("scanvector.txt file not found!") from exc

    if scan_type == "delay":
        t0 = scan_info["TimeZero"]
        coords = mm_to_fs(coords, t0)

    return coords, dim


def mm_to_fs(delaystage, t0):
    delay = delaystage - t0
    delay *= 2 / 2.99792458e11 * 1e15
    return delay


def compare_coords(axis_data: np.ndarray) -> tuple[np.ndarray, int]:
    """Identifies the most changing column in a given 2-D numpy array.

    Args:
        axis_data (np.ndarray): 2-D numpy array containing LUT data

    Returns:
        tuple[np.ndarray, int]:
            - coords: Maximum changing column as a coordinate
            - index: Index of the coords in the axis_data array
    """

    diff_list = [abs(col[-1] - col[0]) for col in axis_data.T]

    index = diff_list.index(max(diff_list))

    if max(diff_list) == 0:
        raise ValueError("Coordinates not found in LUT.")

    coords = axis_data[:, index]
    return coords, index


def parse_info_to_dict(path: Path) -> dict:
    """Parses the contents of info.txt file into a dictionary

    Args:
        path (Path): Path object pointing to the scan folder

    Returns:
        dict: Parsed info_dict dictionary
    """
    info_dict: dict[Any, Any] = {}
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


def handle_meta(
    df_lut: pd.DataFrame,
    scan_info: dict,
    config: dict,
    scan: int,
    fast_axes: list[str],
    slow_axes: list[str],
    projection: str,
    metadata: dict = None,
    collect_metadata: bool = False,
    token: str = None,
) -> dict:
    """Helper function for the handling metadata from different files

    Args:
        df_lut (pd.DataFrame): Pandas dataframe containing the contents of LUT.txt as obtained
            from ``parse_lut_to_df()``
        scan_info (dict): scan_info class dict containing containing the contents of info.txt file
        config (dict): config dictionary containing the contents of config.yaml file
        scan (int): Scan number
        fast_axes (list[str]): The fast-axis dimensions of the scan
        slow_axes (list[str]): The slow-axis dimensions of the scan
        metadata (dict, optional): Metadata dictionary with additional metadata for the scan.
            Defaults to empty dictionary.
        collect_metadata (bool, optional): Option to collect further metadata e.g. from EPICS
            archiver needed for NeXus conversion. Defaults to False.
        token (str, optional):: The elabFTW api token to use for fetching metadata

    Returns:
        dict: metadata dictionary containing additional metadata from the EPICS
        archive and elabFTW.
    """

    if metadata is None:
        metadata = {}

    logger.info("Gathering metadata from different locations")
    # get metadata from LUT dataframe
    lut_meta = {}
    energy_scan_mode = "snapshot"
    if df_lut is not None:
        for col in df_lut.columns:
            col_array = df_lut[f"{col}"].to_numpy()
            if len(set(col_array)) == 1:
                lut_meta[col] = col_array[0]
            else:
                lut_meta[col] = col_array

        kinetic_energy = df_lut["KineticEnergy"].to_numpy()
        if len(set(kinetic_energy)) > 1 and scan_info["ScanType"] == "voltage":
            energy_scan_mode = "fixed_analyser_transmission"  # spell-checker: word: analyser

    metadata["scan_info"] = complete_dictionary(
        metadata.get("scan_info", {}),
        complete_dictionary(scan_info, lut_meta),
    )  # merging dictionaries

    # Store delay info
    if "DelayStage" in metadata["scan_info"] and "TimeZero" in metadata["scan_info"]:
        metadata["scan_info"]["delay"] = mm_to_fs(
            metadata["scan_info"]["DelayStage"],
            metadata["scan_info"]["TimeZero"],
        )

    # store program version
    metadata["scan_info"]["program_name"] = "specsanalyzer"
    metadata["scan_info"]["program_version"] = importlib.metadata.version("specsanalyzer")

    # timing
    logger.info("Collecting time stamps...")
    if "time" in metadata["scan_info"]:
        time_list = [metadata["scan_info"]["time"][0], metadata["scan_info"]["time"][-1]]
    elif "StartTime" in metadata["scan_info"]:
        time_list = [metadata["scan_info"]["StartTime"]]
    else:
        raise ValueError("Could not find timestamps in scan info.")

    dt_list_iso = [time.replace(".", "-").replace(" ", "T") for time in time_list]
    datetime_list = [dt.datetime.fromisoformat(dt_iso) for dt_iso in dt_list_iso]
    ts_from = dt.datetime.timestamp(min(datetime_list))  # POSIX timestamp
    ts_to = dt.datetime.timestamp(max(datetime_list))  # POSIX timestamp
    if ts_from == ts_to:
        try:
            ts_to = (
                ts_from
                + metadata["scan_info"]["Exposure"] / 1000 * metadata["scan_info"]["Averages"]
            )
        except KeyError:
            pass
    metadata["timing"] = {
        "acquisition_start": dt.datetime.fromtimestamp(ts_from, dt.timezone.utc).isoformat(),
        "acquisition_stop": dt.datetime.fromtimestamp(ts_to, dt.timezone.utc).isoformat(),
        "acquisition_duration": int(ts_to - ts_from),
        "collection_time": float(ts_to - ts_from),
    }

    if collect_metadata:
        metadata_retriever = MetadataRetriever(config, token)

        metadata = metadata_retriever.fetch_epics_metadata(
            ts_from=ts_from,
            ts_to=ts_to,
            metadata=metadata,
        )

        metadata = metadata_retriever.fetch_elab_metadata(
            scan=scan,
            metadata=metadata,
        )

    metadata["scan_info"]["energy_scan_mode"] = energy_scan_mode

    metadata["scan_info"]["projection"] = projection
    metadata["scan_info"]["scheme"] = (
        "angular dispersive" if projection == "reciprocal" else "spatial dispersive"
    )

    metadata["scan_info"]["slow_axes"] = slow_axes if slow_axes else ""
    metadata["scan_info"]["fast_axes"] = fast_axes

    return metadata


def find_scan(path: Path, scan: int) -> list[Path]:
    """Search function to locate the scan folder

    Args:
        path (Path): Path object for data from the default config file
        scan (int): Scan number of the scan of interest

    Returns:
        List[Path]: scan_path: Path object pointing to the scan folder
    """
    logger.info("Scan path not provided, searching directories...")
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
                    logger.info(f"Scan found at path: {scan_path[0]}")
                    break
    else:
        scan_path = []
    return scan_path


def find_scan_type(
    path: Path,
    scan_type: str,
):
    """Rudimentary function to print scan paths given the scan_type

    Args:
        path (Path): Path object pointing to the year, for ex.,
            Path("//nap32/topfloor/trARPES/PESData/2020")
        scan_type (str): string containing the scan_type from the list
            ["delay","temperature","manipulator","mirror","single"]
    """
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
