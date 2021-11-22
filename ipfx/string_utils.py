from typing import Union

def to_str(s: Union[str, bytes]
)-> str:
    """Convert bytes to string if not string


    Parameters
    ----------
        s (Union[str,bytes]): variable to convert

    Returns
    -------
        str
    """
    if isinstance(s, bytes):
        return s.decode('utf-8')
    else:
        return s

