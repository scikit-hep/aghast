# automatically generated by the FlatBuffers compiler, do not modify

# namespace: aghast_generated

import flatbuffers


class Profile(object):
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAsProfile(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Profile()
        x.Init(buf, n + offset)
        return x

    # Profile
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Profile
    def Expression(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # Profile
    def Statistics(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from .Statistics import Statistics

            obj = Statistics()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Profile
    def Title(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # Profile
    def Metadata(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from .Metadata import Metadata

            obj = Metadata()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Profile
    def Decoration(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from .Decoration import Decoration

            obj = Decoration()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None


def ProfileStart(builder):
    builder.StartObject(5)


def ProfileAddExpression(builder, expression):
    builder.PrependUOffsetTRelativeSlot(
        0, flatbuffers.number_types.UOffsetTFlags.py_type(expression), 0
    )


def ProfileAddStatistics(builder, statistics):
    builder.PrependUOffsetTRelativeSlot(
        1, flatbuffers.number_types.UOffsetTFlags.py_type(statistics), 0
    )


def ProfileAddTitle(builder, title):
    builder.PrependUOffsetTRelativeSlot(
        2, flatbuffers.number_types.UOffsetTFlags.py_type(title), 0
    )


def ProfileAddMetadata(builder, metadata):
    builder.PrependUOffsetTRelativeSlot(
        3, flatbuffers.number_types.UOffsetTFlags.py_type(metadata), 0
    )


def ProfileAddDecoration(builder, decoration):
    builder.PrependUOffsetTRelativeSlot(
        4, flatbuffers.number_types.UOffsetTFlags.py_type(decoration), 0
    )


def ProfileEnd(builder):
    return builder.EndObject()
