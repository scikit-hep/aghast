#!/usr/bin/env python

import flatbuffers
import histos_generated

class Book(object):
    def tobuffer(self):
        builder = flatbuffers.Builder()
        book = self._toflatbuffers()
        builder.Finish(book)
        return builder.Output()

    def _toflatbuffers(self, builder):
        histos_generated.Book.BookStart(builder)
        histos_generated.Book.BookAddIdentifier(builder, self._identifier)
        if self._title is not None:
            histos_generated.Book.BookAddTitle(builder, self._title)
        return histos_generated.Book.BookEnd(builder)

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
