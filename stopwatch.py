#!/usr/bin/python

import time


class Stopwatch:
    def __init__(self):
        self.__started = False
        self.__stop = 0

    def start(self):
        if self.__started:
            raise RuntimeError("Stopwatch already started")
        self.__started = True
        self.__stop = time.time()
        return self.__stop

    def stop(self):
        if not self.__started:
            raise RuntimeError("The stopwatch was never started")
        self.__started = False
        t = time.time()
        return t, t - self.__stop
