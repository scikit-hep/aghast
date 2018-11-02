#!/usr/bin/env python

import sys
import numbers

import numpy

class Check(object):
    def __init__(self, classname, param, required):
        self.classname = classname
        self.param = param
        self.required = required

    def __repr__(self):
        return "<{0} {1}.{2} at 0x{3:012x}>".format(type(self).__name__, self.classname, self.param, id(self))

    def __call__(self, obj):
        if obj is None and self.required:
            raise TypeError("{0}.{1} is required, cannot pass {1}".format(self.classname, self.param, repr(obj)))
        else:
            return obj

class CheckEnum(Check):
    def __init__(self, classname, param, required, choices):
        super(CheckEnum, self).__init__(classname, param, required)
        self.choices = choices

    def __call__(self, obj):
        super(CheckEnum, self).__call__(obj)
        if obj not in self.choices:
            raise TypeError("{0}.{1} must be one of {2}, cannot pass {3}".format(self.classname, self.param, self.choices, repr(obj)))
        else:
            return self.choices[self.choices.index(obj)]

class CheckString(Check):
    def __call__(self, obj):
        super(CheckString, self).__call__(obj)
        if not ((sys.version_info[0] >= 3 and isinstance(obj, str)) or (sys.version_info[0] < 3 and isinstance(obj, basestring))):
            raise TypeError("{0}.{1} must be a string, cannot pass {2}".format(self.classname, self.param, repr(obj)))
        else:
            return obj

class CheckNumber(Check):
    def __init__(self, classname, param, required, min=float("-inf"), max=float("inf"), min_inclusive=True, max_inclusive=True):
        super(CheckNumber, self).__init__(classname, param, required)
        self.min = min
        self.max = max
        self.min_included = min_included
        self.max_included = max_included

    def __call__(self, obj):
        super(CheckNumber, self).__call__(obj)
        if not isinstance(obj, (numbers.Real, numpy.floating, numpy.integer)):
            raise TypeError("{0}.{1} must be a number, cannot pass {2}".format(self.classname, self.param, repr(obj)))
        elif self.min_inclusive and not self.min <= obj:
            raise TypeError("{0}.{1} must not be below {2} (inclusive), cannot pass {3}".format(self.classname, self.param, self.min, repr(obj)))
        elif not self.min_inclusive and not self.min < obj:
            raise TypeError("{0}.{1} must not be below {2} (exclusive), cannot pass {3}".format(self.classname, self.param, self.min, repr(obj)))
        elif self.max_inclusive and not obj <= self.max:
            raise TypeError("{0}.{1} must not be above {2} (inclusive), cannot pass {3}".format(self.classname, self.param, self.max, repr(obj)))
        elif not self.max_inclusive and not obj < self.max:
            raise TypeError("{0}.{1} must not be above {2} (exclusive), cannot pass {3}".format(self.classname, self.param, self.max, repr(obj)))
        else:
            return float(obj)

class CheckInteger(Check):
    def __init__(self, classname, param, required, min=float("-inf"), max=float("inf")):
        super(CheckInteger, self).__init__(classname, param, required)
        self.min = min
        self.max = max

    def __call__(self, obj):
        super(CheckInteger, self).__call__(obj)
        if not isinstance(obj, (numbers.Integral, numpy.integer)):
            raise TypeError("{0}.{1} must be an integer, cannot pass {2}".format(self.classname, self.param, repr(obj)))
        elif self.min <= obj:
            raise TypeError("{0}.{1} must not be below {2} (inclusive), cannot pass {3}".format(self.classname, self.param, self.min, repr(obj)))
        elif obj <= self.max:
            raise TypeError("{0}.{1} must not be above {2} (inclusive), cannot pass {3}".format(self.classname, self.param, self.max, repr(obj)))
        else:
            return float(obj)



# def required(classname, obj):
#     if obj is None:
#         raise TypeError("{0} is required, cannot pass {1}".format(classname, repr(obj)))
#     return obj

# def enum(classname, obj, choices):
#     if obj is None:
#         return None
#     elif obj not in choices:
#         raise TypeError("{0} must be one of {1}, cannot pass {2}".format(classname, choices, repr(obj)))
#     else:
#         return choices[choices.index(obj)]

# def string(classname, obj):
#     if obj is None:
#         return obj
#     elif not ((sys.version_info[0] >= 3 and isinstance(obj, str)) or (sys.version_info[0] < 3 and isinstance(obj, basestring))):
#         raise TypeError("{0} must be a string, cannot pass {1}".format(classname, repr(obj)))
#     else:
#         return obj

# def number(classname, obj, min=float("-inf"), max=float("inf"), min_inclusive=True, max_inclusive=True):
#     if obj is None:
#         return obj
#     elif not isinstance(obj, (numbers.Real, numpy.floating, numpy.integer)):
#         raise TypeError("{0} must be a number, cannot pass {1}".format(classname, repr(obj)))
#     elif min_inclusive and not min <= obj:
#         raise TypeError("{0} must not be below {1} (inclusive), cannot pass {2}".format(classname, min, repr(obj)))
#     elif not min_inclusive and not min < obj:
#         raise TypeError("{0} must not be below {1} (exclusive), cannot pass {2}".format(classname, min, repr(obj)))
#     elif max_inclusive and not obj <= max:
#         raise TypeError("{0} must not be above {1} (inclusive), cannot pass {2}".format(classname, max, repr(obj)))
#     elif not max_inclusive and not obj < max:
#         raise TypeError("{0} must not be above {1} (exclusive), cannot pass {2}".format(classname, max, repr(obj)))
#     else:
#         return float(obj)
