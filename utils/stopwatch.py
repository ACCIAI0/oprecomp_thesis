#!/usr/bin/python

import time


class Stopwatch:
    def __init__(self):
        self.__started = False
        self.__stop = 0
        self.__exec_time = 0

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
        duration = t - self.__stop
        self.__exec_time += duration
        return t, duration

    def get_duration(self):
        return self.__exec_time


stop_w = Stopwatch()
