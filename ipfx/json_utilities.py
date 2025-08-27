import logging
import numpy as np
import simplejson as json

ju_logger = logging.getLogger(__name__)


def read(file_name):
    """Shortcut reading JSON from a file."""
    with open(file_name, "rb") as f:
        json_string = f.read().decode("utf-8")
        if len(json_string) == 0:  # If empty file
            # Create a string that will give an empty JSON object instead of an
            # error
            json_string = "{}"
        json_obj = json.loads(json_string)

    return json_obj


def write(file_name, obj):
    """Shortcut for writing JSON to a file.  This also takes care of
    serializing numpy and data types."""
    with open(file_name, "wb") as f:
        try:
            f.write(write_string(obj))
        except TypeError:
            f.write(bytes(write_string(obj), "utf-8"))


def write_string(obj):
    """Shortcut for writing JSON to a string.  This also takes care of
    serializing numpy and data types."""
    return json.dumps(
        obj,
        indent=2,
        ignore_nan=True,
        default=json_handler,
        iterable_as_array=True,
    )


def json_handler(obj):
    """Used by write_string convert a few non-standard types to things that the
    json package can handle."""
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, bool) or isinstance(obj, np.bool_):
        return bool(obj)
    elif hasattr(obj, "isoformat"):
        return obj.isoformat()
    else:
        raise TypeError(
            "Object of type %s with value of %s is not JSON serializable"
            % (type(obj), repr(obj))
        )
