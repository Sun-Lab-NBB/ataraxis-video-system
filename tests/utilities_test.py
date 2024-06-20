"""Tests service functions available through the utilities module."""

import re

import pytest
from ataraxis_time.utilities import format_message


def test_format_message():
    """Verifies correct functioning of the format-message function"""

    # Verifies that a long message is formatted appropriately.
    long_message = (
        "This is a very long message that needs to be formatted properly. It should be wrapped at 120 characters "
        "without breaking long words or splitting on hyphens. The formatting should be applied correctly to ensure "
        "readability and consistency across the library."
    )
    # DO NOT REFORMAT. This will break the test.
    # noinspection LongLine
    expected_long_message = (
        "This is a very long message that needs to be formatted properly. It should be wrapped at 120 characters without breaking\n"
        "long words or splitting on hyphens. The formatting should be applied correctly to ensure readability and consistency\n"
        "across the library."
    )
    assert format_message(long_message) == expected_long_message

    # Verifies that a short message remains unaffected.
    short_message = "This is a short message."
    assert format_message(short_message) == short_message


def test_format_message_error_handling():
    """Verifies format-message error handling behavior."""

    # Ensures inputting a non-string results in TypeError
    invalid_type = 123
    custom_error_message = (
        f"Invalid 'message' argument time encountered when formatting text message according to library display "
        f"standards. Expected a {type(str)} type, but encountered {invalid_type} of type {type(invalid_type)}."
    )
    with pytest.raises(TypeError, match=re.escape(format_message(custom_error_message))):
        format_message(invalid_type)
