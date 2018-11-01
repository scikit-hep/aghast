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

    def _toflatbuffers(self, builder):
        identifier = builder.CreateString(self._identifier)
        if self._title is not None:
            title = builder.CreateString(self._title)
        histos.histos_generated.Book.BookStart(builder)
        histos.histos_generated.Book.BookAddIdentifer(builder, identifier)
        if self._title is not None:
            histos.histos_generated.Book.BookAddTitle(builder, title)
        return histos.histos_generated.Book.BookEnd(builder)

    def __init__(self, identifier, title=None):
        self.identifier = identifier
        self.title = title

    @property
    def identifier(self):
        return self._identifier

    @identifier.setter
    def identifier(self, value):
        self._identifier = value

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, value):
        self._title = value
