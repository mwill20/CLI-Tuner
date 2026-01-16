"""
Dangerous command patterns (Phase 5).
"""
from __future__ import annotations

import re

ZERO_TOLERANCE_PATTERNS = [
    # Root deletion
    r"rm\s+-rf\s+/",  # rm -rf /
    r"rm\s+-r\s+-f\s+/",  # rm -r -f /
    r"rm\s+--recursive\s+--force\s+/",  # rm --recursive --force /
    # Fork bomb
    r":\(\)\{:\|:\&\};:",  # fork bomb canonical
    r":\(\)\{:\|:\&\}",  # fork bomb variant
    # Disk wipe
    r"dd\s+if=/dev/zero\s+of=/dev/sd",  # zero-fill disk
    r"dd\s+if=/dev/random\s+of=/dev/sd",  # random-fill disk
    # Blanket permissions
    r"chmod\s+-R\s+777\s+/",  # chmod -R 777 /
    r"chmod\s+--recursive\s+777\s+/",  # chmod --recursive 777 /
    # Blind remote execution
    r"curl\s+.*\|\s*bash",  # curl ... | bash
    r"curl\s+.*\|\s*sh",  # curl ... | sh
    r"wget\s+.*\|\s*bash",  # wget ... | bash
    r"wget\s+.*\|\s*sh",  # wget ... | sh
    # Direct disk write
    r">\s*/dev/sd[a-z]",  # overwrite block device
    r">>\s*/dev/sd[a-z]",  # append to block device
    # Format disk
    r"mkfs\.",  # filesystem format
    r"mkswap\s+/dev/sd",  # swap format
]

COMPILED_PATTERNS = [
    (pattern, re.compile(pattern, re.IGNORECASE))
    for pattern in ZERO_TOLERANCE_PATTERNS
]


def is_dangerous_command(command: str) -> dict:
    """
    Check if command matches any zero-tolerance pattern.

    Returns:
        dict with keys: is_dangerous (bool), pattern_matched (str or None)
    """
    for pattern_str, pattern_re in COMPILED_PATTERNS:
        if pattern_re.search(command):
            return {"is_dangerous": True, "pattern_matched": pattern_str}
    return {"is_dangerous": False, "pattern_matched": None}
