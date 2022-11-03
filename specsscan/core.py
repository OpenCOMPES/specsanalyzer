"""This is the SpecsScan core class

"""
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
import xarray as xr
from specsanalyzer import SpecsAnalyzer

from specsscan.helpers import find_scan
from specsscan.helpers import get_coords
from specsscan.helpers import load_images
from specsscan.helpers import parse_info_to_dict
from specsscan.helpers import parse_lut_to_df
from specsscan.metadata import MetaHandler
from specsscan.settings import parse_config


class SpecsScan:
    """[summary]"""

    def __init__(  # pylint: disable=dangerous-default-value
        self,
        metadata: dict = {},
        config: Union[dict, str] = {},
    ):

        self._config = parse_config(config)

        self._attributes = MetaHandler(meta=metadata)

        self._scan_info: Dict[Any, Any] = {}

        try:
            self.spa = SpecsAnalyzer(config=self._config["spa_params"])
        except KeyError:
            self.spa = SpecsAnalyzer()

    def __repr__(self):
        if self._config is None:
            pretty_str = "No configuration available"
        else:
            pretty_str = ""
            for k in self._config:
                pretty_str += f"{k} = {self._config[k]}\n"
        # TODO Proper report with scan number, dimensions, configuration etc.
        return pretty_str if pretty_str is not None else ""

    @property
    def config(self):
        """Get config"""
        return self._config

    @config.setter
    def config(self, config: Union[dict, str]):
        """Set config"""
        self._config = parse_config(config)
        try:
            self.spa = SpecsAnalyzer(config=self._config["spa_params"])
        except KeyError:
            self.spa = SpecsAnalyzer()

    def load_scan(  # pylint:disable=too-many-locals
        self,
        scan: int,
        path: Union[str, Path] = "",
        iterations: Union[
            list,
            np.ndarray,
            Tuple[int, slice],
        ] = None,
    ) -> xr.DataArray:
        """Load scan with given scan number.

        Args:
            scan: The run number of interest
            path: Either a string of the path to the folder
                containing the scan or a Path object
            iterations: A 1-D array of the number of iterations over
                which the images are to be averaged. The array
                can be a list, numpy array or a Tuple consisting of
                slice objects and integers. For ex.,
                np.s_[1:10, 15, -1] would be a valid input for
                iterations.

        Raises:
            FileNotFoundError

        Returns:
            xres: xarray DataArray object with kinetic energy, angle/position
                and optionally delay as coordinates.
        """
        if path:
            path = Path(path).joinpath(str(scan))
            if not path.is_dir():
                raise FileNotFoundError(
                    f"The provided path {path} was not found.",
                )
        else:
            # search for the given scan using the default path
            path = Path(self._config["data_path"])
            # path_scan = sorted(path.glob(f"20[1,2][9,0-9]/*/*/Raw Data/{scan}"))
            path_scan_list = find_scan(path, scan)
            if not path_scan_list:
                raise FileNotFoundError(
                    f"Scan number {scan} not found",
                )
            path = path_scan_list[0]

        df_lut = parse_lut_to_df(path)  # TODO: storing metadata from df_lut

        data = load_images(
            path,
            df_lut=df_lut,
            iterations=iterations,
        )

        try:
            self._scan_info = parse_info_to_dict(path)

        except FileNotFoundError:
            print("info.txt file not found.")
            raise

        (scan_type, lens_mode, kin_energy, pass_energy, work_function) = (
            self._scan_info["ScanType"],
            self._scan_info["LensMode"],
            self._scan_info["KineticEnergy"],
            self._scan_info["PassEnergy"],
            self._scan_info["WorkFunction"],
        )

        xr_list = []
        for image in data:
            xr_list.append(
                self.spa.convert_image(
                    image,
                    lens_mode,
                    kin_energy,
                    pass_energy,
                    work_function,
                ),
            )

        coords = get_coords(  # df_lut can be used instead
            path,
            scan_type,
            self._scan_info,
        )
        # Handle as per scantype
        if scan_type == "single":
            res_xarray = xr_list[0]
        elif scan_type == "delay":
            res_xarray = xr.concat(
                xr_list,
                dim=xr.DataArray(coords, dims="Delay", name="Delay"),
            )
        elif scan_type == "mirror":
            res_xarray = xr.concat(
                xr_list,
                dim="Mirror",
                # dim=xr.DataArray(coords, dims=["MirrorX"], name="Mirror"),
            )
        elif scan_type == "temperature":
            res_xarray = xr.concat(xr_list, dim="Temperature")

        return res_xarray


# def load_images(
#     scan_path: Path,
#     df_lut: Union[pd.DataFrame, None] = None,
#     iterations: Union[
#         List[int],
#         np.ndarray,
#         Tuple[int, slice],
#     ] = None,
# ) -> np.ndarray:
#     """Loads a 2D/3D numpy array of images for the given
#         scan path with an optional averaging
#         over the given iterations
#     Args:
#         scan_path: object of class Path pointing
#                 to the scan folder
#         raw_array: 1-D numpy array containing the scan names from
#                 the RAW folder.
#         iterations: A 1-D array of the number of iterations over
#                 which the images are to be averaged. The array
#                 can be a list, numpy array or a Tuple consisting of
#                 slice objects and integers. For ex.,
#                 np.s_[1:10, 15, -1] would be a valid input for
#                 iterations.
#     Returns:
#         data: A 2-D or 3-D (concatenated) numpy array consisting
#             of raw data
#     """

#     scan_list = sorted(
#         [
#             file.stem for file in scan_path.joinpath("AVG").iterdir()
#             if file.suffix == ".tsv"
#         ]
#     )

#     data = []
#     if iterations is not None:

#         if df_lut is not None:
#             raw_array = df_lut["filename"].to_numpy()
#         else:
#             raise Exception(
#                 "RAW folder empty. "
#                 "Try without passing iterations in case of a single scan.",
#             )

#         raw_2d = get_raw2d(scan_list, raw_array)
#         # Slicing along the given iterations
#         raw_2d_iter = raw_2d[np.r_[iterations]].T

#         print("Averaging over iterations...")
#         for delay in tqdm(raw_2d_iter):
#             avg_list = []
#             for image in tqdm(delay, leave=False):
#                 if image != "nan":

#                     with open(
#                         scan_path.joinpath(f"RAW/{image}"),
#                         encoding="utf-8",
#                     ) as file:
#                         new_im = np.loadtxt(file, delimiter="\t")
#                         avg_list.append(new_im)

#             data.append(
#                 np.average(
#                     np.array(avg_list),
#                     axis=0,
#                 ),
#             )
#     else:
#         for image in tqdm(scan_list):
#             with open(
#                 scan_path.joinpath(
#                     f"AVG/{image}.tsv"
#                 ),
#                 encoding="utf-8",
#             ) as file:

#                 new_im = np.loadtxt(file, delimiter="\t")
#                 data.append(new_im)

#     return np.array(data)


# def get_raw2d(
#     scan_list: List[str],
#     raw_array: np.ndarray,
# ) -> np.ndarray:
#     """Converts a 1-D array of raw scan names
#         into 2-D based on the number of iterations
#     Args:
#         scan_list: A list of AVG scan names.
#         raw_list: 1-D array of RAW scan names.
#     Returns:
#         raw_2d: 2-D numpy array of size for ex.,
#             (total_iterations, delays) for a delay scan.
#     """

#     total_iterations = len(
#         [
#             im for im in raw_array if f"{scan_list[0]}_" in im
#         ],
#     )

#     delays = len(scan_list)
#     diff = delays * (total_iterations) - len(raw_array)

#     if diff:  # Ongoing or aborted scan
#         diff = delays - diff  # Number of scans in the last iteration
#         raw_2d = raw_array[:-diff].reshape(
#             total_iterations - 1,
#             delays,
#         )

#         last_iter_array = np.full(
#             (1, delays),
#             fill_value="nan",
#             dtype="object",
#         )

#         last_iter_array[0, :diff] = raw_array[-diff:]
#         raw_2d = np.concatenate(
#             (raw_2d, last_iter_array),
#         )
#     else:  # Complete scan
#         raw_2d = raw_array.reshape(total_iterations, delays)

#     return raw_2d


# def parse_lut_to_df(scan_path: Path) -> Union[pd.DataFrame, None]:
#     """Loads the contents of LUT.txt file into a pandas
#         data frame to be used as metadata.
#     Args:
#         scan_path: Path object for the scan path
#     Returns: A pandas DataFrame
#     """
#     try:
#         df_lut = pd.read_csv(scan_path.joinpath("RAW/LUT.txt"), sep="\t")
#         df_lut.reset_index(inplace=True)
#         new_cols = df_lut.columns.to_list()[1:]
#         new_cols.insert(3, "delay (fs)")  # Correct the column names
#         df_lut.columns = new_cols

#     except FileNotFoundError:
#         print(
#             "LUT.txt not found. "
#             "Storing metadata from info.txt"
#         )
#         return None

#     return df_lut


# def get_coords(
#     scan_path: Path,
#     scan_type: str,
#     scan_info: Dict[Any, Any],
# ) -> np.ndarray:
#     """Reads the contents of scanvector.txt file
#         into a numpy array.
#     Args:
#         scan_path: Path object for the scan path
#         scan_type: Type of scan (delay, mirror etc.)
#         scan_info: scan_info class dict
#     Raises:
#         FileNotFoundError
#     Returns:
#         coords: 1-D/2-D numpy array containing coordinates
#                 of the scanned axis.
#     """
#     try:
#         with open(
#             scan_path.joinpath("scanvector.txt"),
#             encoding="utf-8",
#         ) as file:
#             coords = np.loadtxt(file)

#     except FileNotFoundError as exc:
#         if scan_type == "single":
#             return None

#         raise FileNotFoundError(
#             "scanvector.txt file not found!"
#         ) from exc

#     if scan_type == "delay":
#         t_0 = scan_info["TimeZero"]
#         coords -= t_0
#         coords *= 2 / (3 * 10**11) * 10**15

#     return coords


# def parse_info_to_dict(path: Path) -> Dict:
#     """Parses the contents of info.txt file
#         into a dictionary
#     Args:
#         path: Path object pointing to the scan folder
#     Returns:
#         info_dict: Parsed dictionary
#     """
#     info_dict: Dict[Any, Any] = {}
#     with open(path.joinpath("info.txt"), encoding="utf-8") as info_file:

#         for line in info_file.readlines():

#             if "=" in line:  # older scans
#                 line_list = line.rstrip("\nV").split("=")

#             elif ":" in line:
#                 line_list = line.rstrip("\nV").split(":")

#             else:
#                 continue

#             key, value = line_list[0], line_list[1]

#             try:
#                 info_dict[key] = float(value)
#             except ValueError:
#                 info_dict[key] = value

#     return info_dict


# def find_scan(path: Path, scan: int) -> List[Path]:
#     """Search function to locate the scan folder
#     Args:
#         path: Path object for data from the default config file
#         scan: Scan number of the scan of interest
#     Returns:
#         scan_path: Path object pointing to the scan folder
#     """
#     print("Scan path not provided, searching directories...")
#     for file in path.iterdir():

#         if file.is_dir():

#             try:
#                 base = int(file.stem)

#             except ValueError:  # not numeric
#                 continue

#             if base >= 2019:  # only look at folders 2019 onwards

#                 scan_path = sorted(
#                     file.glob(f"*/*/Raw Data/{scan}"),
#                 )
#                 if scan_path:
#                     print("Scan found at path:", scan_path[0])
#                     break
#     return scan_path
