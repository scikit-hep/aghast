#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/aghast/blob/master/LICENSE

import sys
import numbers
import collections

try:
    from collections.abc import Iterable
    from collections.abc import Sequence
    from collections.abc import Mapping
except ImportError:
    from collections import Iterable
    from collections import Sequence
    from collections import Mapping

import numpy


def setparent(parent, value):
    import aghast.interface

    if isinstance(value, aghast.interface.Ghast):
        if getattr(value, "_parent", parent) is not parent:
            raise ValueError(
                "already attached to another hierarchy: {0}".format(repr(value))
            )
        else:
            value._parent = parent

    elif (sys.version_info[0] >= 3 and isinstance(value, (str, bytes))) or (
        sys.version_info[0] < 3 and isinstance(value, basestring)
    ):
        pass

    elif isinstance(value, Vector):
        for x in value:
            setparent(parent, x)

    elif isinstance(value, Lookup):
        for x in value.values():
            setparent(parent, x)


def _checkitem(check):
    if check.type is str:
        return CheckString(check.classname, check.paramname, required=check.required)
    elif check.type is float:
        return CheckNumber(check.classname, check.paramname, required=check.required)
    elif check.type is int:
        return CheckInteger(check.classname, check.paramname, required=check.required)
    elif isinstance(check.type, list):
        return CheckEnum(
            check.classname,
            check.paramname,
            required=check.required,
            choices=check.type,
        )
    else:
        return CheckClass(
            check.classname, check.paramname, required=check.required, type=check.type
        )


class Vector(Sequence):
    def __init__(self, data):
        if data is None:
            self._data = ()
        else:
            self._data = tuple(data)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, where):
        return self._data[where]

    def _detached(self, top):
        import aghast.interface

        return Vector(
            [
                x if not isinstance(x, aghast.interface.Ghast) else x._detached(False)
                for x in self
            ]
        )

    def __repr__(self):
        tmp = [repr(x) for x in self]
        if sum(len(x) for x in tmp) < 100:
            return "[" + ", ".join(tmp) + "]"
        elif len(tmp) < 100:
            return "[" + ",\n ".join(tmp) + "]"
        else:
            return (
                "[" + ",\n ".join(tmp[:50]) + ",\n ...\n " + ",\n ".join(tmp[:50]) + "]"
            )

    def __eq__(self, other):
        if not isinstance(other, (Vector, Iterable)):
            return False
        if len(self) != len(other):
            return False
        for i in range(len(self)):
            if self[i] != other[i]:
                return False
        else:
            return True

    def __ne__(self, other):
        return not self.__eq__(other)


class FBVector(Vector):
    def __init__(self, length, get, check, parent):
        self._got = [None] * length
        self._get = get
        assert isinstance(check, CheckVector), repr(type(check))
        if not check.minlen <= length <= check.maxlen:
            raise TypeError(
                "{0}.{1} length must be between {2} and {3} (inclusive)".format(
                    check.classname, check.paramname, check.minlen, check.maxlen
                )
            )
        self._check = _checkitem(check)
        self._parent = parent

    def __len__(self):
        return len(self._got)

    def __getitem__(self, where):
        if self._got[where] is None:
            self._got[where] = self._check.fromflatbuffers(self._get(where))
            setparent(self._parent, self._got[where])
        return self._got[where]


class Lookup(Mapping):
    def __init__(self, data):
        import aghast.interface

        if data is None:
            self._data = collections.OrderedDict()
        else:
            self._data = collections.OrderedDict(data)
        for n, x in self._data.items():
            if isinstance(x, aghast.interface.Ghast) and not hasattr(x, "_parent"):
                x._identifier = n

    def __len__(self):
        return len(self._data)

    def __getitem__(self, where):
        return self._data[where]

    def __iter__(self):
        return iter(self._data)

    def _detached(self, top):
        import aghast.interface

        return Lookup(
            {
                n: x
                if not isinstance(x, aghast.interface.Ghast)
                else x._detached(False)
                for n, x in self.items()
            }
        )

    def __repr__(self):
        tmp = [repr(n) + ": " + repr(x) for n, x in self.items()]
        if sum(len(x) for x in tmp) < 100:
            return "{" + ", ".join(tmp) + "}"
        elif len(tmp) < 100:
            return "{" + ",\n ".join(tmp) + "}"
        else:
            return (
                "{" + ",\n ".join(tmp[:50]) + ",\n ...\n " + ",\n ".join(tmp[:50]) + "}"
            )

    def __eq__(self, other):
        if not isinstance(other, (Lookup, Mapping)):
            return False
        if set(self) != set(other):
            return False
        for n in self:
            if self[n] != other[n]:
                return False
        else:
            return True

    def __ne__(self, other):
        return not self.__eq__(other)


class FBLookup(Lookup):
    def __init__(self, length, lookup, get, check, parent):
        self._lookup = {lookup(i).decode("utf-8"): i for i in range(length)}
        self._got = {}
        self._get = get
        assert isinstance(check, CheckLookup), repr(type(check))
        if not check.minlen <= length <= check.maxlen:
            raise TypeError(
                "{0}.{1} length must be between {2} and {3} (inclusive)".format(
                    check.classname, check.paramname, check.minlen, check.maxlen
                )
            )
        self._check = _checkitem(check)
        self._parent = parent

    def __len__(self):
        return len(self._lookup)

    def __iter__(self):
        return iter(self._lookup)

    def __getitem__(self, where):
        import aghast.interface

        item = self._got.get(where, None)
        if item is None:
            item = self._check.fromflatbuffers(self._get(self._lookup[where]))
            self._got[where] = item
            if isinstance(item, aghast.interface.Ghast):
                item._identifier = where
            setparent(self._parent, item)
        return item


class Check(object):
    def __init__(self, classname, paramname, required):
        self.classname = classname
        self.paramname = paramname
        self.required = required

    def __repr__(self):
        return "<{0} {1}.{2} at 0x{3:012x}>".format(
            type(self).__name__, self.classname, self.paramname, id(self)
        )

    def __call__(self, obj):
        if obj is None and self.required:
            raise TypeError(
                "{0}.{1} is required, cannot pass {2} (type {3})".format(
                    self.classname, self.paramname, repr(obj), type(obj)
                )
            )
        else:
            return obj

    def fromflatbuffers(self, obj):
        return obj


class CheckBool(Check):
    def __call__(self, obj):
        super(CheckBool, self).__call__(obj)
        if obj is None:
            return obj
        elif not isinstance(obj, (bool, numpy.bool_, numpy.bool)):
            raise TypeError(
                "{0}.{1} must be boolean, cannot pass {2} (type {3})".format(
                    self.classname, self.paramname, repr(obj), type(obj)
                )
            )
        return bool(obj)


class CheckString(Check):
    def __call__(self, obj):
        super(CheckString, self).__call__(obj)
        if obj is None:
            return obj
        elif not (
            (sys.version_info[0] >= 3 and isinstance(obj, str))
            or (sys.version_info[0] < 3 and isinstance(obj, basestring))
        ):
            raise TypeError(
                "{0}.{1} must be a string, cannot pass {2} (type {3})".format(
                    self.classname, self.paramname, repr(obj), type(obj)
                )
            )
        else:
            return obj

    def fromflatbuffers(self, obj):
        if obj is None:
            return obj
        else:
            return obj.decode("utf-8")


class CheckNumber(Check):
    def __init__(
        self,
        classname,
        paramname,
        required,
        min=float("-inf"),
        max=float("inf"),
        min_inclusive=True,
        max_inclusive=True,
    ):
        super(CheckNumber, self).__init__(classname, paramname, required)
        self.min = min
        self.max = max
        self.min_inclusive = min_inclusive
        self.max_inclusive = max_inclusive

    def __call__(self, obj):
        super(CheckNumber, self).__call__(obj)
        if obj is None:
            return obj
        elif not isinstance(
            obj, (numbers.Real, numpy.floating, numpy.integer)
        ) or numpy.isnan(obj):
            raise TypeError(
                "{0}.{1} must be a number, cannot pass {2} (type {3})".format(
                    self.classname, self.paramname, repr(obj), type(obj)
                )
            )
        elif self.min_inclusive and not self.min <= obj:
            raise TypeError(
                "{0}.{1} must not be below {2} (inclusive), cannot pass {3} (type {4})".format(
                    self.classname, self.paramname, self.min, repr(obj), type(obj)
                )
            )
        elif not self.min_inclusive and not self.min < obj:
            raise TypeError(
                "{0}.{1} must not be below {2} (exclusive), cannot pass {3} (type {4})".format(
                    self.classname, self.paramname, self.min, repr(obj), type(obj)
                )
            )
        elif self.max_inclusive and not obj <= self.max:
            raise TypeError(
                "{0}.{1} must not be above {2} (inclusive), cannot pass {3} (type {4})".format(
                    self.classname, self.paramname, self.max, repr(obj), type(obj)
                )
            )
        elif not self.max_inclusive and not obj < self.max:
            raise TypeError(
                "{0}.{1} must not be above {2} (exclusive), cannot pass {3} (type {4})".format(
                    self.classname, self.paramname, self.max, repr(obj), type(obj)
                )
            )
        else:
            return float(obj)


class CheckInteger(Check):
    def __init__(
        self, classname, paramname, required, min=float("-inf"), max=float("inf")
    ):
        super(CheckInteger, self).__init__(classname, paramname, required)
        self.min = min
        self.max = max

    def __call__(self, obj):
        super(CheckInteger, self).__call__(obj)
        if obj is None:
            return obj
        elif not isinstance(obj, (numbers.Integral, numpy.integer)):
            raise TypeError(
                "{0}.{1} must be an integer, cannot pass {2} (type {3})".format(
                    self.classname, self.paramname, repr(obj), type(obj)
                )
            )
        elif not self.min <= obj:
            raise TypeError(
                "{0}.{1} must not be below {2} (inclusive), cannot pass {3} (type {4})".format(
                    self.classname, self.paramname, self.min, repr(obj), type(obj)
                )
            )
        elif not obj <= self.max:
            raise TypeError(
                "{0}.{1} must not be above {2} (inclusive), cannot pass {3} (type {4})".format(
                    self.classname, self.paramname, self.max, repr(obj), type(obj)
                )
            )
        else:
            return int(obj)


class CheckEnum(Check):
    def __init__(self, classname, paramname, required, choices, intlookup=None):
        super(CheckEnum, self).__init__(classname, paramname, required)
        self.choices = choices
        self.intlookup = intlookup

    def __call__(self, obj):
        super(CheckEnum, self).__call__(obj)
        if obj is None:
            return obj
        elif obj not in self.choices:
            raise TypeError(
                "{0}.{1} must be one of {2}, cannot pass {3} (type {4})".format(
                    self.classname, self.paramname, self.choices, repr(obj), type(obj)
                )
            )
        else:
            return self.choices[self.choices.index(obj)]

    def fromflatbuffers(self, obj):
        if obj is None:
            return obj
        elif self.intlookup is None:
            return self.choices[obj]
        else:
            return self.intlookup[obj]


class CheckClass(Check):
    def __init__(self, classname, paramname, required, type):
        super(CheckClass, self).__init__(classname, paramname, required)
        self.type = type

    def __call__(self, obj):
        super(CheckClass, self).__call__(obj)
        if obj is None:
            return obj
        elif not isinstance(obj, self.type):
            raise TypeError(
                "{0}.{1} must be a {2} object, cannot pass {3} (type {4})".format(
                    self.classname, self.paramname, self.type, repr(obj), type(obj)
                )
            )
        return obj

    def fromflatbuffers(self, obj):
        if obj is None:
            return obj
        else:
            return self.type._fromflatbuffers(obj)


class CheckKey(Check):
    def __init__(self, classname, paramname, required, type):
        super(CheckKey, self).__init__(classname, paramname, required)
        self.type = type

    def __call__(self, obj):
        super(CheckKey, self).__call__(obj)
        if obj is None:
            return obj
        elif self.type is str:
            if not (
                (sys.version_info[0] >= 3 and isinstance(obj, str))
                or (sys.version_info[0] < 3 and isinstance(obj, basestring))
            ):
                raise TypeError(
                    "{0}.{1} must be a string, cannot pass {2} (type {3})".format(
                        self.classname, self.paramname, repr(obj), type(obj)
                    )
                )
            return obj
        elif self.type is float:
            if not isinstance(obj, (numbers.Real, numpy.floating, numpy.integer)):
                raise TypeError(
                    "{0}.{1} must be a number, cannot pass {2} (type {3})".format(
                        self.classname, self.paramname, repr(obj), type(obj)
                    )
                )
            return float(obj)
        elif self.type is int:
            if not isinstance(obj, (numbers.Integral, numpy.integer)):
                raise TypeError(
                    "{0}.{1} must be an integer, cannot pass {2} (type {3})".format(
                        self.classname, self.paramname, repr(obj), type(obj)
                    )
                )
            return int(obj)
        else:
            if not isinstance(obj, self.type):
                raise TypeError(
                    "{0}.{1} must be a {2} object, cannot pass {3} (type {4})".format(
                        self.classname, self.paramname, self.type, repr(obj), type(obj)
                    )
                )
            return obj

    def fromflatbuffers(self, obj):
        if obj is None or self.type is float or self.type is int:
            return obj
        elif self.type is str:
            return obj.decode("utf-8")
        else:
            return self.type._fromflatbuffers(obj)


class CheckVector(Check):
    def __init__(
        self, classname, paramname, required, type, minlen=0, maxlen=float("inf")
    ):
        super(CheckVector, self).__init__(classname, paramname, required)
        self.type = type
        self.minlen = minlen
        self.maxlen = maxlen

    def __call__(self, obj):
        super(CheckVector, self).__call__(obj)
        if obj is None:
            return Vector(obj)
        elif not isinstance(obj, Iterable):
            raise TypeError(
                "{0}.{1} must be iterable, cannot pass {2} (type {3})".format(
                    self.classname, self.paramname, repr(obj), type(obj)
                )
            )

        if (sys.version_info[0] >= 3 and isinstance(obj, (str, bytes))) or (
            sys.version_info[0] < 3 and isinstance(obj, basestring)
        ):
            raise TypeError(
                "{0}.{1} must be iterable but not a string, cannot pass {2}".format(
                    self.classname, self.paramname, repr(obj)
                )
            )

        if not self.minlen <= len(obj) <= self.maxlen:
            raise TypeError(
                "{0}.{1} length must be between {2} and {3} (inclusive), cannot pass {4} (type ({5}))".format(
                    self.classname,
                    self.paramname,
                    self.minlen,
                    self.maxlen,
                    repr(obj),
                    type(obj),
                )
            )

        if self.type is str:
            for x in obj:
                if not (
                    (sys.version_info[0] >= 3 and isinstance(x, str))
                    or (sys.version_info[0] < 3 and isinstance(x, basestring))
                ):
                    raise TypeError(
                        "{0}.{1} elements must be strings, cannot pass {2} (type {3})".format(
                            self.classname, self.paramname, repr(x), type(x)
                        )
                    )
            return Vector(obj)
        elif self.type is int:
            return numpy.array(obj, dtype="<i8")
        elif self.type is float:
            return numpy.array(obj, dtype="<f8")
        elif isinstance(self.type, list):
            for x in obj:
                if not x in self.type:
                    raise TypeError(
                        "{0}.{1} elements must be one of {2}, cannot pass {3} (type {4})".format(
                            self.classname, self.paramname, self.type, repr(x), type(x)
                        )
                    )
            return Vector(self.type[self.type.index(x)] for x in obj)
        else:
            for x in obj:
                if not isinstance(x, self.type):
                    raise TypeError(
                        "{0}.{1} elements must be {2} objects, cannot pass {3} (type {4})".format(
                            self.classname, self.paramname, self.type, repr(x), type(x)
                        )
                    )
            return Vector(obj)


class CheckLookup(Check):
    def __init__(
        self, classname, paramname, required, type, minlen=0, maxlen=float("inf")
    ):
        super(CheckLookup, self).__init__(classname, paramname, required)
        self.type = type
        self.minlen = minlen
        self.maxlen = maxlen

    def __call__(self, obj):
        super(CheckLookup, self).__call__(obj)
        if obj is None:
            return Lookup(obj)
        elif not isinstance(obj, (dict, Mapping)):
            raise TypeError(
                "{0}.{1} must be a dict or Mapping, cannot pass {2} (type {3})".format(
                    self.classname, self.paramname, repr(obj), type(obj)
                )
            )

        if not self.minlen <= len(obj) <= self.maxlen:
            raise TypeError(
                "{0}.{1} length must be between {2} and {3} (inclusive), cannot pass {4} (type ({5}))".format(
                    self.classname,
                    self.paramname,
                    self.minlen,
                    self.maxlen,
                    repr(obj),
                    type(obj),
                )
            )

        if self.type is str:
            for x in obj.values():
                if not (
                    (sys.version_info[0] >= 3 and isinstance(x, str))
                    or (sys.version_info[0] < 3 and isinstance(x, basestring))
                ):
                    raise TypeError(
                        "{0}.{1} elements must be strings, cannot pass {2} (type {3})".format(
                            self.classname, self.paramname, repr(x), type(x)
                        )
                    )
            return Lookup(obj)
        elif self.type is float:
            for x in obj.values():
                if not isinstance(x, (numbers.Real, numpy.integer, numpy.floating)):
                    raise TypeError(
                        "{0}.{1} elements must be numbers, cannot pass {2} (type {3})".format(
                            self.classname, self.paramname, repr(x), type(x)
                        )
                    )
            return Lookup(float(x) for x in obj)
        elif isinstance(self.type, list):
            for x in obj.values():
                if not x in self.type:
                    raise TypeError(
                        "{0}.{1} elements must be one of {2}, cannot pass {3} (type {4})".format(
                            self.classname, self.paramname, self.type, repr(x), type(x)
                        )
                    )
            return Lookup(self.type[self.type.index(x)] for x in obj)
        else:
            for x in obj.values():
                if not isinstance(x, self.type):
                    raise TypeError(
                        "{0}.{1} elements must be {2} objects, cannot pass {3} (type {4})".format(
                            self.classname, self.paramname, self.type, repr(x), type(x)
                        )
                    )
            return Lookup(obj)


class CheckBuffer(Check):
    def __call__(self, obj):
        super(CheckBuffer, self).__call__(obj)
        if obj is None:
            return obj
        if isinstance(obj, numpy.ndarray) and not obj.flags.c_contiguous:
            obj = obj.copy(order="C")
        try:
            return numpy.frombuffer(obj, dtype=numpy.uint8)
        except AttributeError:
            raise TypeError(
                "{0}.{1} must be a buffer, cannot pass {2} (type {3})".format(
                    self.classname, self.paramname, repr(obj), type(obj)
                )
            )


class CheckSlice(Check):
    def __call__(self, obj):
        super(CheckSlice, self).__call__(obj)
        if obj is None:
            return obj
        elif not isinstance(obj, slice):
            raise TypeError(
                "{0}.{1} must be a slice, cannot pass {2} (type {3})".format(
                    self.classname, self.paramname, repr(obj), type(obj)
                )
            )
        return out

    def fromflatbuffers(self, obj):
        if obj is None:
            return obj
        else:
            return slice(
                obj.Start() if obj.HasStart() else None,
                obj.Stop() if obj.HasStop() else None,
                obj.Step() if obj.HasStep() else None,
            )
