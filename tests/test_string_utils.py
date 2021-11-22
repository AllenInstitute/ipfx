from ipfx.string_utils import to_str
import pytest




@pytest.mark.parametrize(
    "input, expected",
    [
        # Case 0: supplied string
        (
            "a_string", "a_string"
        ),
        # Case 1: supplied bytes
        (
            b"a_string", "a_string"
        ),
    ]
)
def test_to_string(input, expected):
    obtained = to_str(input)
    assert obtained == expected