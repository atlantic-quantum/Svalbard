#!/usr/bin/env python3
"""
Helper functions for managing strings.
"""

import os

import numpy as np


def convert_version_str(version_str: str = "1.0.0") -> int:
    """Convert version string to a number for easy comparison, 1.2.3 => 123

    Currently only supports single-digit version numbers.

    Parameters
    ----------
    version_str : str, optional
        Version string to convert, by default '1.0.0'

    Returns
    -------
    int
        Version as number.
    """
    if not isinstance(version_str, str):
        version_str = str(version_str)
    version_list = version_str.split(".")
    # avoid errors if input is empty
    if len(version_list) == 1 and len(version_list[0]) == 0:
        version_list = []
    version_number = 0
    for n, value in enumerate([100, 10, 1]):
        # make sure code handles trailing zeros, ie 3.1 = 3.1.0
        if n < len(version_list):
            version_number += value * int(version_list[n][0])
    return version_number


def append_counter(name: str, sep: str = "_", remove_extension: bool = False) -> str:
    """Append counter to the input string, using format s_2, s_3 etc.

    Parameters
    ----------
    name : str
        Input string.
    sep : str, optional
        Seperator before counter, by default '_'.
    remove_extension : bool, optional
        Remove file extension (like .txt), by default False.

    Returns
    -------
    str
        Output string with counter applied
    """
    # remove extension
    if remove_extension:
        (name, ext) = os.path.splitext(name)
    else:
        ext = ""
    # split from right
    parts = name.rsplit(sep, 1)
    n = 1
    if len(parts) > 1:
        if parts[1].isdigit():
            n = int(parts[1])
        else:
            parts[0] = name
    new_name = "%s%s%d%s" % (parts[0], sep, n + 1, ext)
    return new_name


def get_si_string(
    value: float, unit: str = "", decimals: int = 3, exp_if_no_unit: bool = False
) -> str:
    """Generate a string with unit and SI prefix for given input value.

    The function converts the input value to exponential format, groups the
    exponents in groups covering 3 orders of magnitude, and returns a string
    with the correct prefix and unit.
    For example, for V units:  0.0122 => 12.2 mV

    Parameters
    ----------
    value : float
        Input value
    unit : str, optional
        Unit, by default ''
    decimals : int, optional
        Digit of precision, by default 3.  Note that because of the grouping
        of 3 orders of magnitude, the function will always use at least 3 digits
        of precision.
    exp_if_no_unit : bool, optional
        If True and no unit is given, the funcion will return a string with
        the value expressed in exponential format.  Default is False.

    Returns
    -------
    str
        String with value, prefix and unit
    """
    # call conversion function and return correct string
    d = convert_to_si(value, unit, decimals=decimals, exp_if_no_unit=exp_if_no_unit)
    return d["full_str"]


def get_engineering_str(value: float, decimals: int = 3) -> str:
    """Generate a string in engineering format given input value.

    The function converts the input value to exponential format, groups the
    exponents in groups covering 3 orders of magnitude, and returns a string
    with the value in exponentinal format.
    For example, for V units:  0.0122 => 12.2E-3

    Parameters
    ----------
    value : float
        Input value
    decimals : int, optional
        Digit of precision, by default 3.  Note that because of the grouping
        of 3 orders of magnitude, the function will always use at least 3 digits
        of precision.

    Returns
    -------
    str
        String with value in engineering format
    """
    # call conversion function and return correct string
    d = convert_to_si(value, unit="", decimals=decimals)
    return d["exp_str"]


def convert_to_si(
    value: float,
    unit: str = "",
    decimals: int = 3,
    exp_if_no_unit: bool = False,
    exponent: int = None,
) -> dict:
    """Scale a value and generate string with unit and proper SI formatting.

    The function converts the input value to exponential format, groups the
    exponents in groups covering 3 orders of magnitude, and returns a dict with
    the scaled value, the scaling factor, the correct prefix, a string with the
    value with prefix+unit, and a string with the value in exponential format.

    Parameters
    ----------
    value : float
        Input value
    unit : str, optional
        Unit, by default ''
    decimals : int, optional
        Digit of precision, by default 3.  Note that because of the grouping
        of 3 orders of magnitude, the function will always use at least 3 digits
        of precision.
    exp_if_no_unit : bool, optional
        If True and no unit is given, the funcion will return a string with
        the value expressed in exponential format.  Default is False.
    exponent : int, optional
        If specified, the user can define what exponent to use.  For example,
        if exponent=3 and unit is Hz, the value will alwasy be returned as kHz,
        regardless of how large the value is.  Default is None.

    Returns
    -------
    dict
        Dicionary with the following fields:
        - full_str:  String with value, prefix and unit
        - prefix:  SI prefix in use
        - scaled_value:  Value scaled to match SI prefix
        - scale:  Scaling factor to take input value to scaled_value
        - exp_str:  String with value in exponential format
    """
    # create output dict
    d = dict(full_str="", prefix="", scaled_value=value, scale=1.0, exp_str="")
    # check for string, inf, -inf, nan
    if isinstance(value, str):  # pragma: no cover
        d["full_str"] = value
        d["exp_str"] = value
        return d
    if np.isnan(value):
        d["full_str"] = "NaN"
        d["exp_str"] = "NaN"
        return d
    if value == float("inf"):
        d["full_str"] = "Inf"
        d["exp_str"] = "Inf"
        return d
    if value == float("-inf"):
        d["full_str"] = "-Inf"
        d["exp_str"] = "-Inf"
        return d
    # make sure we have at least three digits
    decimals = max(decimals, 3)
    # check if exponent is given by user
    if exponent is None:
        # avoid error if value is zero
        if value != 0:
            # base conversion on built-in conversion to engineering string
            temp = ("%%.%de" % (decimals - 1)) % value
            n = temp.upper().find("E")
            if n < 0:  # pragma: no cover
                exponent = 0
            else:
                exponent = int(temp[n + 1 :])
        else:
            exponent = 0

    # exponenets are always in steps of 3
    exponent = 3 * (exponent // 3)

    # index to available prefix strings
    prefices = ["y", "z", "a", "f", "p", "n", "u", "m", "", "k", "M", "G", "T", "P"]
    indx = exponent // 3 + 8
    if 0 <= indx < len(prefices):
        d["prefix"] = prefices[indx]
    else:
        # no prefix available, use exponential notation
        d["prefix"] = "E%d" % exponent
    # find new value
    d["scale"] = 1.0 / (10.0**exponent)
    d["scaled_value"] = value * d["scale"]
    # create complete output strings
    value = f"{d['scaled_value']:.{decimals}g}"
    d["full_str"] = f"{value}{'' if unit == '' else ' '}{d['prefix']}{unit}"
    d["exp_str"] = f"{value}{'' if exponent == 0 else f'E{exponent}'}"
    if exp_if_no_unit and unit == "":
        d["full_str"] = d["exp_str"]
    return d


def get_value_from_si_string(si_str: str) -> float:
    """Returns a number with the value encoded in a SI-type string.

    The function assumes the string to contain either a number or a number
    followed by an SI-prefix.  The function returns None for invalid strings.


    Parameters
    ----------
    si_str : str,
        String with input value, in SI format (value + prefix, no unit)

    Returns
    -------
    float
        Value contained in input string, scaled according to prefix in use.
    """
    # strip whitespaces
    si_str = si_str.strip()
    # check length
    if len(si_str) == 0:
        return 0.0
    try:
        # look for SI prefix
        prefices = ["y", "z", "a", "f", "p", "n", "u", "m", "", "k", "M", "G", "T", "P"]
        if si_str[-1] in prefices:
            # look for prefix and return scaled value
            exponent = 3 * (prefices.index(si_str[-1]) - 8)
            return float(si_str[:-1]) * (10**exponent)
        # check if last character is numeric, if so convert directly
        if si_str[-1].isdigit() or (si_str[-1] == "."):
            return float(si_str)
        else:
            # otherwise return none
            return None
    except ValueError:  # pragma: no cover
        return None
