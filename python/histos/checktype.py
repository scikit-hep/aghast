#!/usr/bin/env python

import sys

def string(context, obj):
    if not ((sys.version_info[0] >= 3 and isinstance(obj, str)) or (sys.version_info[0] < 3 and isinstance(obj, basestring))):
        raise TypeError("{0} must be a string, cannot pass {1}".format(context, repr(obj)))
    return obj

# def string(context, obj, required=False):
#     if obj is None and required:
#         raise TypeError("{0} is required, cannot pass {1}".format(context, repr(obj)))
#     if obj is not None and not ((sys.version_info[0] >= 3 and isinstance(obj, str)) or (sys.version_info[0] < 3 and isinstance(obj, basestring))):
#         raise TypeError("{0} must be a string, cannot pass {1}".format(context, repr(obj)))
#     return obj

