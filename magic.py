import contextlib as c
import hashlib as h
import os as o
import tempfile as t

_m = None

def magic():
    global _m
    if _m is None:
        with open("./magic", "rb") as f:
            _m = f.read()
            if h.md5(_m[16:]).digest() != _m[:16]:
                exit(0)
    return _m

@c.contextmanager
def magic3():
    with t.TemporaryDirectory() as _t:
        with open(o.path.join(_t, "m.png"), "wb") as f:
            f.write(magic()[16:])
        yield ('-i', o.path.join(_t, "m.png"), '-filter_complex')
