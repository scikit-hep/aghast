#!/usr/bin/env python

import flatbuffers

import histos.histos_generated
import histos.histos_generated.Book

class Book(object):
    def tobuffer(self, initialSize=1024):
        builder = flatbuffers.Builder(initialSize)
        book = self._toflatbuffers(builder)
        builder.Finish(book)
        return builder.Output()

    @classmethod
    def frombuffer(cls, buffer, offset=0):
        out = cls.__new__(cls)
        out._flatbuffers = histos.histos_generated.Book.Book.GetRootAsBook(buffer, offset)
        return out

    def __init__(self, identifier, title=None):
        self.identifier = identifier
        self.title = title

    @property
    def identifier(self):
        if not hasattr(self, "_identifier"):
            self._identifier = self._flatbuffers.Identifier()
        return self._identifier

    @identifier.setter
    def identifier(self, value):
        self._identifier = value

    @property
    def title(self):
        if not hasattr(self, "_title"):
            self._title = self._flatbuffers.Title()
        return self._title

    @title.setter
    def title(self, value):
        self._title = value

    def _toflatbuffers(self, builder):
        identifier = builder.CreateString(self._identifier)
        if self._title is not None:
            title = builder.CreateString(self._title)
        histos.histos_generated.Book.BookStart(builder)
        histos.histos_generated.Book.BookAddIdentifier(builder, identifier)
        if self._title is not None:
            histos.histos_generated.Book.BookAddTitle(builder, title)
        return histos.histos_generated.Book.BookEnd(builder)
