"""This module contains file input/output functions for the specsanalyzer module"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import Sequence

import h5py
import numpy as np
import tifffile
import xarray as xr
from pynxtools.dataconverter.convert import convert

_IMAGEJ_DIMS_ORDER = "TZCYXS"
_IMAGEJ_DIMS_ALIAS = {
    "T": [
        "delayStage",
        "pumpProbeTime",
        "time",
        "delay",
        "T",
    ],
    "Z": [
        "dldTime",
        "t",
        "energy",
        "e",
        "E",
        "binding_energy",
        "energies",
        "binding_energies",
    ],
    "C": ["C"],
    "Y": ["dldPosY", "ky", "y", "ypos", "Y"],
    "X": ["dldPosX", "kx", "x", "xpos", "X"],
    "S": ["S"],
}


def recursive_write_metadata(h5group: h5py.Group, node: dict):
    """Recurses through a python dictionary and writes it into an hdf5 file.

    Args:
        h5group (h5py.Group): hdf5 group element where to store the current dict node to.
        node (dict): dictionary node to store

    Raises:
        Warning: warns if elements have been converted into strings for saving.
        ValueError: Rises when elements cannot be saved even as strings.
    """
    for key, item in node.items():
        if isinstance(
            item,
            (
                np.ndarray,
                np.int64,
                np.float64,
                str,
                bytes,
                int,
                float,
                list,
            ),
        ):
            try:
                h5group.create_dataset(key, data=item)
            except TypeError:
                h5group.create_dataset(key, data=str(item))
                print(f"Saved {key} as string.")
        elif isinstance(item, dict):
            group = h5group.create_group(key)
            recursive_write_metadata(group, item)
        else:
            try:
                h5group.create_dataset(key, data=str(item))
                print(f"Saved {key} as string.")
            except BaseException as exc:
                raise ValueError(
                    f"Unknown error occured, cannot save {item} of type {type(item)}.",
                ) from exc


def recursive_parse_metadata(
    node: h5py.Group | h5py.Dataset,
) -> dict:
    """Recurses through an hdf5 file, and parse it into a dictionary.

    Args:
        node (h5py.Group | h5py.Dataset): hdf5 group or dataset to parse into dictionary.

    Returns:
        dict: Dictionary of elements in the hdf5 path contained in node
    """
    if isinstance(node, h5py.Group):
        dictionary = {}
        for key, value in node.items():
            dictionary[key] = recursive_parse_metadata(value)

    else:
        entry = node[...]
        try:
            dictionary = entry.item()
            if isinstance(dictionary, (bytes, bytearray)):
                dictionary = dictionary.decode()
        except ValueError:
            dictionary = entry

    return dictionary


def to_h5(data: xr.DataArray, faddr: str, mode: str = "w"):
    """Save xarray formatted data to hdf5

    Args:
        data (xr.DataArray): input data
        faddr (str): complete file name (including path)
        mode (str): hdf5 read/write mode

    Raises:
        Warning: subfunction warns if elements have been converted into strings for
            saving.
    """
    with h5py.File(faddr, mode) as h5_file:
        print(f"saving data to {faddr}")

        # Saving data, make a single dataset
        dataset = h5_file.create_dataset("binned/BinnedData", data=data.data)
        try:
            dataset.attrs["units"] = data.attrs["units"]
            dataset.attrs["long_name"] = data.attrs["long_name"]
        except KeyError:
            pass

        # Saving axes
        axes_group = h5_file.create_group("axes")
        axes_number = 0
        for bin_name in data.dims:
            axis = axes_group.create_dataset(
                f"ax{axes_number}",
                data=data.coords[bin_name],
            )
            axis.attrs["name"] = bin_name
            try:
                axis.attrs["unit"] = data.coords[bin_name].attrs["unit"]
            except KeyError:
                pass
            axes_number += 1

        if "metadata" in data.attrs and isinstance(
            data.attrs["metadata"],
            dict,
        ):
            meta_group = h5_file.create_group("metadata")

            recursive_write_metadata(meta_group, data.attrs["metadata"])

    print("Saving complete!")


def load_h5(faddr: str, mode: str = "r") -> xr.DataArray:
    """Read xarray data from formatted hdf5 file

    Args:
        faddr (str): complete file name (including path)
        mode (str, optional): hdf5 read/write mode. Defaults to "r"

    Returns:
        xr.DataArray: output xarra data
    """
    with h5py.File(faddr, mode) as h5_file:
        # Reading data array
        try:
            data = np.asarray(h5_file["binned"]["BinnedData"])
        except KeyError as exc:
            raise ValueError(
                f"Wrong Data Format, the BinnedData were not found. The error was{exc}.",
            ) from exc

        # Reading the axes
        bin_axes = []
        bin_names = []

        try:
            for axis in h5_file["axes"]:
                bin_axes.append(h5_file["axes"][axis])
                bin_names.append(h5_file["axes"][axis].attrs["name"])
        except KeyError as exc:
            raise ValueError(
                f"Wrong Data Format, the axes were not found. The error was {exc}",
            ) from exc

        # load metadata
        metadata = None
        if "metadata" in h5_file:
            metadata = recursive_parse_metadata(h5_file["metadata"])

        coords = {}
        for name, vals in zip(bin_names, bin_axes):
            coords[name] = vals

        xarray = xr.DataArray(data, dims=bin_names, coords=coords)

        try:
            for axis in range(len(bin_axes)):
                xarray[bin_names[axis]].attrs["unit"] = h5_file["axes"][f"ax{axis}"].attrs["unit"]
            xarray.attrs["units"] = h5_file["binned"]["BinnedData"].attrs["units"]
            xarray.attrs["long_name"] = h5_file["binned"]["BinnedData"].attrs["long_name"]
        except (KeyError, TypeError):
            pass

        if metadata is not None:
            xarray.attrs["metadata"] = metadata

        return xarray


def to_tiff(
    data: xr.DataArray | np.ndarray,
    faddr: Path | str,
    alias_dict: dict = None,
):
    """Save an array as a .tiff stack compatible with ImageJ

    Args:
        data (xr.DataArray | np.ndarray): data to be saved. If a np.ndarray, the order is retained.
            If it is an xarray.DataArray, the order is inferred from axis_dict instead.
            ImageJ likes tiff files with axis order as TZCYXS. Therefore, best axis order in input
            should be: Time, Energy, posY, posX. The channels 'C' and 'S' are automatically added
            and can be ignored.
        faddr (Path | str): full path and name of file to save.
        alias_dict (dict, optional): name pairs for correct axis ordering. Keys should be any of
            T,Z,C,Y,X,S. The Corresponding value should be a dimension of the xarray or
            the dimension number if a numpy array. This is used to sort the data in the
            correct order for imagej standards. If None it tries to guess the order
            from the name of the axes or assumes T,Z,C,Y,X,S order for numpy arrays.
            Defaults to None.

    Raise:
        AttributeError: if more than one axis corresponds to a single dimension
        NotImplementedError: if data is not 2,3 or 4 dimensional
        TypeError: if data is not a np.ndarray or an xarray.DataArray
    """

    out: np.ndarray | xr.DataArray = None
    if isinstance(data, np.ndarray):
        # TODO: add sorting by dictionary keys
        dim_expansions = {2: [0, 1, 2, 5], 3: [0, 2, 5], 4: [2, 5]}
        dims = {
            2: ["x", "y"],
            3: ["x", "y", "energy"],
            4: ["x", "y", "energy", "delay"],
        }
        try:
            out = np.expand_dims(data, dim_expansions[data.ndim])
        except KeyError:
            raise NotImplementedError(  # pylint: disable=W0707
                f"Only 2-3-4D arrays supported when data is a {type(data)}",
            )

        dims_order = dims[data.ndim]

    elif isinstance(data, xr.DataArray):
        dims_order = _fill_missing_dims(list(data.dims), alias_dict=alias_dict)
        out = data.expand_dims(
            {dim: 1 for dim in dims_order if dim not in data.dims},
        )
        out = out.transpose(*dims_order)
    else:
        raise TypeError(f"Cannot handle data of type {data.type}")

    faddr = Path(faddr).with_suffix(".tiff")

    tifffile.imwrite(faddr, out.astype(np.float32), imagej=True)

    print(f"Successfully saved {faddr}\n Axes order: {dims_order}")
    # return dims_order


def _sort_dims_for_imagej(dims: list, alias_dict: dict = None) -> list:
    """Guess the order of the dimensions from the alias dictionary

    Args:
        dims (list): the list of dimensions to sort
        alias_dict (dict, optional): name pairs for correct axis ordering. Keys should be any of
            T,Z,C,Y,X,S. The Corresponding value should be a dimension of the xarray or
            the dimension number if a numpy array. This is used to sort the data in the
            correct order for imagej standards. If None it tries to guess the order
            from the name of the axes or assumes T,Z,C,Y,X,S order for numpy arrays.
            Defaults to None.

    Raises:
        ValueError: for duplicate entries for a single imagej dimension
        NameError: when a dimension cannot be found in the alias dictionary

    Returns:
        list: List of sorted dimensions
    """
    order = _fill_missing_dims(dims=dims, alias_dict=alias_dict)
    return [d for d in order if d in dims]


def _fill_missing_dims(dims: list, alias_dict: dict = None) -> list:
    """Guess the order of the dimensions from the alias dictionary

    Args:
        dims (list): the list of dimensions to sort
        alias_dict (dict, optional): name pairs for correct axis ordering. Keys should be any of
            T,Z,C,Y,X,S. The Corresponding value should be a dimension of the xarray or
            the dimension number if a numpy array. This is used to sort the data in the
            correct order for imagej standards. If None it tries to guess the order
            from the name of the axes or assumes T,Z,C,Y,X,S order for numpy arrays.
            Defaults to None.

    Raises:
        ValueError: for duplicate entries for a single imagej dimension
        NameError: when a dimension cannot be found in the alias dictionary

    Returns:
        list: List of extended dimensions
    """
    order: list = []
    # overwrite the default values with the provided dict
    if alias_dict is None:
        alias_dict = {}
    else:
        for k, v in alias_dict.items():
            assert k in _IMAGEJ_DIMS_ORDER, "keys must all be one of " f"{_IMAGEJ_DIMS_ALIAS}"
            if not isinstance(v, (list, tuple)):
                alias_dict[k] = [v]

    alias_dict = {**_IMAGEJ_DIMS_ALIAS, **alias_dict}
    added_dims = 0
    for imgj_dim in _IMAGEJ_DIMS_ORDER:
        found_one = False
        for dim in dims:
            if dim in alias_dict[imgj_dim]:
                if found_one:
                    raise ValueError(
                        f"Duplicate entries for {imgj_dim}: {dim} and {order[-1]} ",
                    )
                order.append(dim)
                found_one = True
        if not found_one:
            order.append(imgj_dim)
            added_dims += 1
    if len(order) != len(dims) + added_dims:
        raise NameError(
            f"Could not interpret dimensions {[d for d in dims if d not in order]}",
        )
    return order


def load_tiff(
    faddr: str | Path,
    coords: dict = None,
    dims: Sequence = None,
    attrs: dict = None,
) -> xr.DataArray:
    """Loads a tiff stack to an xarray.

    The .tiff format does not retain information on the axes, so these need to
    be manually added with the axes argument. Otherwise, this returns the data
    only as np.ndarray

    Args:
        faddr (str | Path): Path to file to load.
        coords (dict, optional): The axes describing the data, following the tiff stack order:
        dims (Sequence, optional): the order of the coordinates provided, considering the data is
            ordered as TZCYXS. If None (default) it infers the order from the order
            of the coords dictionary.
        attrs (dict, optional): dictionary to add as attributes to the xarray.DataArray

    Returns:
        xr.DataArray: an xarray representing the data loaded from the .tiff file
    """
    data = tifffile.imread(faddr)

    if coords is None:
        coords = {
            k.replace("_", ""): np.linspace(0, n, n)
            for k, n in zip(
                _IMAGEJ_DIMS_ORDER,
                data.shape,
            )
            if n > 1
        }

    data = data.squeeze()

    if dims is None:
        dims = list(coords.keys())

    assert data.ndim == len(dims), (
        f"Data dimension {data.ndim} must coincide number of coordinates "
        f"{len(coords)} and dimensions {len(dims)} provided,"
    )
    return xr.DataArray(data=data, coords=coords, dims=dims, attrs=attrs)


def to_nexus(
    data: xr.DataArray,
    faddr: str,
    reader: str,
    definition: str,
    input_files: str | Sequence[str],
    **kwds,
):
    """Saves the x-array provided to a NeXus file at faddr, using the provided reader,
    NeXus definition and configuration file.

    Args:
        data (xr.DataArray): The data to save, containing metadata definitions in
            data._attrs["metadata"].
        faddr (str): The file path to save to.
        reader (str): The name of the NeXus reader to use.
        definition (str): The NeXus definiton to use.
        input_files (str | Sequence[str]): The file path to the configuration file to use.
        **kwds: Keyword arguments for ``nexusutils.dataconverter.convert``.
    """

    if isinstance(input_files, str):
        input_files = tuple([input_files])
    else:
        input_files = tuple(input_files)

    convert(
        input_file=input_files,
        objects=(data),
        reader=reader,
        nxdl=definition,
        output=faddr,
        **kwds,
    )


def get_pair_from_list(list_line: list) -> list:
    """Returns key value pair for the read function
    where a line in the file contains '=' character.

    Args:
        list_line (list): list of splitted line from the file.

    Returns:
        list: List of a tuple containing key value pair.
    """
    k, v = list_line[0], list_line[1]
    k = k.strip()
    if "#" in v:
        v = v[: v.index("#")].strip()

    if len(v.split()) > 1:
        try:
            v = [float(i) for i in v.split()]
        except ValueError:  # to handle one edge case
            return [(k, float(v.strip('"m')))]

    else:
        try:
            v = float(v)
        except ValueError:
            v = v.strip(' " ')

    return [(k, v)]


def read_calib2d(filepath: str) -> list:
    """Reads the calib2d file into a convenient list for the parser
    function containing useful and cleaned data.

    Args:
        filepath (str): Path to file to load.

    Returns:
        list: List containing dictionary, string and float objects.
    """
    with open(filepath, encoding="utf-8") as file:
        lines = file.readlines()

    listf: list[Any] = []
    for line in lines:
        if "# !!!!! Place a valid calib2D file from your Specslab Installation here!" in line:
            print(
                "No valid calib2 file found. Please provide the path to the calib2d file",
                "\r\ncorresponding to your detector in the config, or copy into the config",
                "\r\nfolder of the package! Without valid calib2d file, calibration parameters",
                "\r\nhave to be provided explicitly.",
            )

        if line[0] == "\n" or line[0] == "#":
            continue
        line_list = line.strip("[]\n").split("=")
        if len(line_list) > 1:
            listf.append(dict(get_pair_from_list(line_list)))
        else:
            line_str = line_list[0]
            if "defaults" in line_str:
                listf.append(line_str.split()[0])
            elif "@" in line_str:
                listf.append(float(line_str.split("@")[1]))
            else:
                pass

    return listf


def parse_calib2d_to_dict(filepath: str) -> dict:
    """Parses the given calib2d file into a nested dictionary structure
    to provide parameters for image conversion.

    Args:
        filepath (str): Path to file to load.

    Returns:
        dict: Populated nested dictionary parsed from the provided calib2d file.
    """
    listf = read_calib2d(filepath)

    calib_dict: dict[Any, Any] = {}
    mode = None
    retardation_ratio = None
    for elem in listf:
        if isinstance(elem, str):  # Initialize mode dict
            mode = elem
            calib_dict[mode] = {"rr": {}, "default": {}}
            retardation_ratio = None
        elif isinstance(elem, (int, float)):  # Initialize rr nested dict
            retardation_ratio = elem
            calib_dict[mode]["rr"][retardation_ratio] = {}
        else:  # populate the dict
            if retardation_ratio:
                calib_dict[mode]["rr"][retardation_ratio].update(elem)
            elif mode:
                calib_dict[mode]["default"].update(elem)
            else:
                calib_dict.update(elem)

        # add the supported lens modes
        (
            calib_dict["supported_angle_modes"],
            calib_dict["supported_space_modes"],
        ) = get_modes_from_calib_dict(calib_dict)

    return calib_dict


def get_modes_from_calib_dict(calib_dict: dict) -> tuple[list, list]:
    """create a list of supported modes, divided in spatial and angular modes

    Args:
        calib_dict (dict): the calibration dictionary, created with the io
        parse_calib2d_to_dict

    Returns:
        tuple[list, list]: lists of supported angular and spatial lens modes
    """
    key_list = list(calib_dict.keys())
    supported_angle_modes = []
    supported_space_modes = []
    for elem in key_list:
        if "AngleMode" in elem or "AngularDispersion" in elem:
            # this is an angular mode
            supported_angle_modes.append(elem)
        if "Area" in elem or "Magnification" in elem:
            # this is an spatial mode
            supported_space_modes.append(elem)
    return supported_angle_modes, supported_space_modes
