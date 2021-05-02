#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 12:32:50 2021

@author: jacob
"""


import os
import argparse

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in


class HeatMapOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Thermal tracker")

        # PATHS
        self.parser.add_argument("--dataPath",
                                 type=str,
                                 help="path to the sensor data",
                                 default=(file_dir))
        self.parser.add_argument("--fileFormat",
                                 type=str,
                                 help="format of the raw data",
                                 default="rdat")
        self.parser.add_argument("--configFilePath",
                                 type=str,
                                 help="path to the config file",
                                 default='/home/jacob/Documents/Code/Python/Calumino/pipeline.json')
        self.parser.add_argument("--sensorNames",
                                 nargs="+",
                                 type=str,
                                 help="name of the movie to write",
                                 default=['0000bd99d28d'])
        self.parser.add_argument("--startTimeDate",
                                 nargs="+",
                                 type=int,
                                 help="Starting time in local date and time format yyyy mm dd hh mm ss",
                                 default=[2021, 4, 1])
        self.parser.add_argument("--endTimeDate",
                                 nargs="+",
                                 type=int,
                                 help="Ending time in local date and time format yyyy mm dd hh mm ss",
                                 default=[2021, 5, 1])
        self.parser.add_argument("--timeZone",
                                 type=str,
                                 default='Australia/Sydney',
                                 help="lcoal time zone e.g., Australia/Sydney")
        self.parser.add_argument("--measurement",
                                 type=str,
                                 choices=["mean", "STD", "PDF"],
                                 default='mean',
                                 help="type of information to plot on the heat map")

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
