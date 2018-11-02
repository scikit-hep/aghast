#!/usr/bin/env python

import sys
import numbers

import numpy

def required(context, obj):
    if obj is None:
        raise TypeError("{0} is required, cannot pass {1}".format(context, repr(obj)))
    return obj

def enum(context, obj, choices):
    if obj is None:
        return None
    elif obj not in choices:
        raise TypeError("{0} must be one of {1}, cannot pass {2}".format(context, choices, repr(obj)))
    else:
        return choices[choices.index(obj)]

def string(context, obj):
    if obj is None:
        return obj
    elif not ((sys.version_info[0] >= 3 and isinstance(obj, str)) or (sys.version_info[0] < 3 and isinstance(obj, basestring))):
        raise TypeError("{0} must be a string, cannot pass {1}".format(context, repr(obj)))
    else:
        return obj

def number(context, obj, min=float("-inf"), max=float("inf"), min_inclusive=True, max_inclusive=True):
    if obj is None:
        return obj
    elif not isinstance(obj, (numbers.Real, numpy.floating, numpy.integer)):
        raise TypeError("{0} must be a number, cannot pass {1}".format(context, repr(obj)))
    elif min_inclusive and not min <= obj:
        raise TypeError("{0} must not be below {1} (inclusive), cannot pass {2}".format(context, min, repr(obj)))
    elif not min_inclusive and not min < obj:
        raise TypeError("{0} must not be below {1} (exclusive), cannot pass {2}".format(context, min, repr(obj)))
    elif max_inclusive and not obj <= max:
        raise TypeError("{0} must not be above {1} (inclusive), cannot pass {2}".format(context, max, repr(obj)))
    elif not max_inclusive and not obj < max:
        raise TypeError("{0} must not be above {1} (exclusive), cannot pass {2}".format(context, max, repr(obj)))
    else:
        return float(obj)
