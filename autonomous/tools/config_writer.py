"""Write measured values into config.py, keeping everything else as it is.

Calibration and test tools call write_values() when the user presses APPLY
(or passes --apply). Only the named assignments change: comments, layout
and every other value are left alone. A one-line assignment keeps its
trailing comment. The old file is kept as config.py.bak, the new text is
checked to parse before it replaces the old file, and the write is atomic.

Every name must already exist in config.py (a typo must not quietly add a
new setting); a missing name raises KeyError and nothing is written.
"""
import ast
import os
import shutil

CONFIG_FILE = "config.py"
BACKUP_SUFFIX = ".bak"
COMMENT_COLUMN = 33                  # config.py lines up its comments here


def config_path():
    # autonomous/config.py, wherever this is run from.
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), CONFIG_FILE)


def format_value(name, value):
    # Python source for one assignment. A list of tuples (FRONT_BANDS) or
    # rows (matrices) gets one item per line, like the hand-written config.
    if isinstance(value, list) and value and isinstance(value[0], (list, tuple)):
        items = "".join(f"    {_plain(item)},\n" for item in value)
        return f"{name} = [\n{items}]"
    return f"{name} = {_plain(value)}"


def _plain(value):
    # repr with NumPy scalars turned into plain Python numbers.
    if isinstance(value, dict):
        return "{" + ", ".join(f"{_plain(k)}: {_plain(v)}" for k, v in value.items()) + "}"
    if isinstance(value, (list, tuple)):
        inner = ", ".join(_plain(v) for v in value)
        if isinstance(value, tuple):
            return f"({inner},)" if len(value) == 1 else f"({inner})"
        return f"[{inner}]"
    if hasattr(value, "item"):
        value = value.item()
    return repr(value)


def assignment_spans(source):
    # {name: (first_line, last_line)} (1-based, inclusive) for top-level
    # `NAME = ...` assignments.
    spans = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            spans[node.targets[0].id] = (node.lineno, node.end_lineno)
    return spans


def replace_values(source, values):
    # The new config text with each named assignment replaced.
    spans = assignment_spans(source)
    missing = [name for name in values if name not in spans]
    if missing:
        raise KeyError(f"not in config.py: {', '.join(missing)}")
    lines = source.splitlines(keepends=True)
    for name in sorted(values, key=lambda n: spans[n][0], reverse=True):
        first, last = spans[name]
        new = format_value(name, values[name])
        comment = _trailing_comment(lines[first - 1]) if first == last and "\n" not in new else ""
        if comment:
            pad = COMMENT_COLUMN - 1 - len(new)
            new = new + " " * max(1, pad) + comment
        lines[first - 1:last] = [new + "\n"]
    text = "".join(lines)
    ast.parse(text)                      # never write a config that doesn't parse
    return text


def _trailing_comment(line):
    # "# comment" at the end of a one-line assignment, or "".
    _, hash_, comment = line.rstrip("\n").partition("#")
    return "# " + comment.strip() if hash_ else ""


def write_values(path, values):
    # Rewrites path with the values; returns the backup's path.
    with open(path) as f:
        source = f.read()
    text = replace_values(source, values)
    backup = path + BACKUP_SUFFIX
    shutil.copyfile(path, backup)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)
    return backup
