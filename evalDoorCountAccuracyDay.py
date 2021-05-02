#!/usr/bin/env python
# Copyright (C) Calumino. All rights reserved.
# Principal authors: Eric Chong, Yaqub Jonmohamadi (Calumino 2021).

"""

# Evaluate door counter accuracy

# Assumption
============
To measure miscount, we need to determine the difference between the measured count and the true count: Nm - Nt = Nerr.
Because we do not have ground truth of the door counter, we have somehow guess a time when the counter is likely to be
zero (e.g., 3am - designated end of day). To increase our confidence of this assumption, we can further check for an
inactive period prior to day end.

Miscounts are skipped for days without sufficient inactive period e.g., nonzero count at day end but inactive period is
less than threshold. Because we can't be certain that the room is actually empty at




"""

from __future__ import division
import sys
import os
import time
import numpy as np
import ast
import glob
import json
import shutil
import webbrowser

#from past.utils import old_div
import datetime, pytz
from matplotlib import pyplot as plt
import csv
# time zones for existing counters
utc = pytz.utc
sydney = pytz.timezone('Australia/Sydney')
brisbane = pytz.timezone('Australia/Brisbane')
perth = pytz.timezone('Australia/Perth')
amsterdam = pytz.timezone('Europe/Amsterdam')
paris = pytz.timezone('Europe/Paris')
headVSpass_flag = 0       # how to plot the accuracy: 0 uses the daily head count (num entry only) and 1 uses num entry + num exit.


Epsil = 0.001
hourStart = 6
hourEnd   = 21
daysList = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
monthList =['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
hourDict = {'hours' :[ii+hourStart for ii in range((hourEnd - hourStart) + 1)],
            'labels':[str(ii+hourStart -1) + '-' + str(ii+hourStart) for ii in range((hourEnd - hourStart) + 1)],
            'index' :[ii for ii in range((hourEnd - hourStart) + 1)] }


counterDict = {
    'Names'    :   ['cognian01', 'cognian02', 'hyrde1' ,'hyrde2' ,'hyrde3', 'legrand-hex',
                    'officeEric1','officeEric2','vae-demo1','vae-perth2','vae-perth3','ybf1'],
    'TimeZones':   [sydney     , sydney     ,amsterdam ,amsterdam,amsterdam,paris        ,
                    sydney     ,sydney      ,brisbane  ,perth    ,perth    , sydney],
    'ResetNightly':[True       ,True       ,  True     ,True     ,True    ,True          ,
                    False      , False       ,True         ,True        ,True        ,True ],
}

StatDict = {'Names': [], 'total counts':[], 'miscounts':[],
            'accuracy':[], 'number of resets':[], 'period':[] }
#counterDict['WeekDaysMeans'] = [meanHours for i in range(len(counterDict['Names']))]

dailyEndTime = datetime.time(3)    # 3am. If rqd min and sec, datetime.time(3, 10, 15)
nextHourTime = datetime.time(0,59,59,999999)    # almost an hour period
skipDayWithTimeGap = True        # time gap between two consecutive log entries
skipPartialDay = False           # a full day has 24 hours if False days with log entries less than 24 hrs are included
inactivePeriodSec = 2 * 60 * 60  # 2 hours
timeGapThresholdSec = 10 * 60    # 10 min
minEvalPeriodSec = 12 * 60 * 60  # at least 12 hours in a single evaluation day
largeCountChange = 4             # threshold for detecting big changes to count


def createHTML(outputPath, HTML_Path):
    prefixes = ['<!DOCTYPE html>', '<html>', '<body>', '<h1>Door counting evaluation results</h1>']
    suffixes = ['</body>', '</html>']
    #imageCommons = []
    if os.sep == '\\':
        PreFix = '<img src="file://'
    else:
        PreFix = '<img src="'
    file = open(HTML_Path + os.sep +"DayAndHour.html", "w")
    for prefix in prefixes:
        file.write(prefix + '\n')

    for zoneIndex in counterDict['Names']:
        file.write('<h2>' + zoneIndex + '</h2> \n')
        images = sorted(glob.glob(outputPath + os.sep + zoneIndex + '*'))

        for image in images:
            file.write(PreFix+image+'" width="570" height="500"> \n')

    file.write(prefixes[0] +'\n')
    file.write(suffixes[1] +'\n')
    file.close()          
    #return datetime.datetime.fromtimestamp(currentTimeStamp/1000).strftime("%A")

def createCSV(CSV_path='./'):
    CSV_path = os.path.join(CSV_path)

    with open(CSV_path+'Sensor_Stats.csv', 'a', newline='\n') as fd:
        writer = csv.writer(fd)
        line = []
        Keys = list(StatDict.keys())
        for Key in (Keys):
            line = [Key]
            for col in range(len(StatDict[Key])):
                line.append(StatDict[Key][col])
            writer.writerow(line)
        writer.writerow('')
            #fd.writerow('\n')

def ep_to_day(currentTimeStamp):
    return datetime.datetime.fromtimestamp(currentTimeStamp/1000).strftime("%A")

def timestamp2datetime(timestamp):
    try:
        # we assume timestamp is already in UTC !!!!
        return datetime.datetime.utcfromtimestamp(timestamp)
    except:
        return timestampMilli2datetime(timestamp)


def timestampMilli2datetime(timestamp, tz=None):
    # try to convert with milli assumed
    #t = int(old_div(timestamp, 1e3))
    t = int(timestamp // 1e3)
    ms = (timestamp - t * 1e3)
    dt = datetime.datetime.utcfromtimestamp(t) + datetime.timedelta(milliseconds=ms)

    if tz is not None:
        utc_tz = pytz.utc
        dt = utc_tz.localize(dt)
        dt = dt.astimezone(pytz.timezone(tz))
    return dt


def saveCounterDatabase(databasePath, counter, data):
    if not os.path.exists(databasePath):
        os.makedirs(databasePath)
        print(os.path.abspath(os.path.join( os.getcwd())))
    with open(databasePath + os.sep + counter + '.json', 'w') as f:
        json.dump(data, f)
    return None


def loadCounterDatabase(databasePath, counter):
    fullPath = databasePath + os.sep + counter + '.json'
    if not os.path.exists(fullPath):
        return None

    with open(fullPath, 'r') as json_file:
        data = json.load(json_file)
    return data


# database format:
# db = {"ID": str, "lastProcessedTimeStamp": int, "TZ": str, "processedFileStats": dict, "countData": list}
# jsonData = serializeData(counterID, lastProcTimestamp, counterTimeZones[counter], logFileStats, countData)
def serializeData(counterID, lastProcessedTimeStamp, counterTZ, processedLogStats, countData):
    jsonData = {'ID': counterID,
                'lastProcessedTimeStamp': lastProcessedTimeStamp,
                'TZ': counterTZ,
                'processedFileStats': processedLogStats,
                'countData': countData}
    return jsonData


def deserializeData(db):
    counterID = db['ID']
    lastProcessedTimeStamp = db['lastProcessedTimeStamp']
    counterTZ = db['TZ']
    processedLogStats = db['processedFileStats']
    countData = db['countData']
    return counterID, lastProcessedTimeStamp, counterTZ, processedLogStats, countData


def getCounterLogs(logPath, counter):
    fList = glob.glob(logPath + os.sep + counter + '*.log')
    fList.sort()
    return fList


def verifyIDs(currentID, dbID):
    return currentID == dbID


def isProcessed(logFile, processFileStats, lastProcessedTimestamp):
    processed = False
    fileSize, fileStartTS, fileEndTS = processFileStats
    currentFileSize = os.path.getsize(logFile)
    if currentFileSize != fileSize:
        return processed
    if lastProcessedTimestamp > fileEndTS:
        processed = True
    return processed


# Conditions for a RESET event
# 1. EVT_AUTO_RESET (explicit)
# 2. (implicit)
#     - change from a large non-zero to zero
# this is detected elsewhere as it requires ongoing monitoring of counts over several hours
#   - non-zero count stays the same for a long period of time prior to resetting
#   - visibleCount = 0 while peopleCount is non-zero during that long constant period
def detectReset(currentCount, previousCount, eventType, changeThreshold):
    isReset = False

    if eventType == 'EVT_AUTO_RESET':    # detect the reset using te flag
        isReset = True
        return isReset

    if currentCount == 0:               # detect the reset using the sudden changes in the number of head count
        if np.abs(previousCount) >= changeThreshold:
            isReset = True

    return isReset


def isNewDay(currentTime, endDayTime):
    return currentTime > endDayTime

def isNewHour(currentTime, nextHour):
    return currentTime > nextHour

def getNextHourTime(currentDateTime):
    return currentDateTime.replace(minute=nextHourTime.minute, second=nextHourTime.second, microsecond=nextHourTime.microsecond)

def getEndDayDateTime(currentDateTime, endTime, lastDateTime=None):
    endDateTime = currentDateTime.replace(hour=endTime.hour, minute=endTime.minute, second=endTime.second)

    if endTime.hour < 12:
        endDateTime = endDateTime + datetime.timedelta(days=1)

    # this moves end day time to allow for log with less than 1 full day of data eg end time = 2am but log finishes at 12am.
    if lastDateTime is not None:
        diff = lastDateTime - currentDateTime
        if endDateTime > lastDateTime and diff.total_seconds() > minEvalPeriodSec:
            endDateTime = lastDateTime - datetime.timedelta(seconds=1)
    return endDateTime


def detectTimeGap(currentTime, previousTime, gapPeriod=timeGapThresholdSec):
    return (currentTime - previousTime).seconds > gapPeriod


def backupDB(backupPath, databasePath, counter):
    if not os.path.exists(backupPath):
        os.makedirs(backupPath)

    dateStr = datetime.datetime.today().strftime("%Y%m%d")

    fullPath = databasePath + os.sep + counter + '.json'
    outPath = backupPath + os.sep + dateStr + '_' + counter + '.json'
    if os.path.exists(fullPath):
        shutil.copyfile(fullPath, outPath)
    return None


# currently a log is allowed to get up to 25MB in size before a new log is created for new entries. However this
def handleLogDisconuity(logLines):
    # check first and last lines - if they are incomplete - discard them
    # => provide a warning if a reset event is detected?
    start = 0
    end = None
    firstLine = logLines[0]
    lastLine = logLines[-1]

    if firstLine[:9] != '{"schema"':
        start = 1
    if lastLine[-1] != '\n':
        end = -1

    return logLines[start:end]


def retrieveLog(logPath, logFile, fromTS=None):
    with open(logPath + os.sep + logFile, 'r') as f:
        lines = f.readlines()

    lines = handleLogDisconuity(lines)

    log = []
    fileSize = os.path.getsize(logPath + os.sep + logFile)
    firstTimeStamp = ast.literal_eval(lines[0])['utcMSecs']
    lastTimeStamp = ast.literal_eval(lines[-1])['utcMSecs']
    logStats = [fileSize, firstTimeStamp, lastTimeStamp]
    start = 0
    if fromTS is not None:
        for idx, line in enumerate(lines[start:]):
            try:
                lineDict = ast.literal_eval(line)
            except:
                continue
            if lineDict["utcMSecs"] >= fromTS:
                start = idx
                break

    if start < len(lines):
        #log = [ast.literal_eval(line) for line in lines[start:]]
        for line in lines[start:]:
            try:
                lineDict = ast.literal_eval(line)
            except:
                # ignore corrupted lines
                continue
            log.append(lineDict)
    return log, logStats


# with no reset the miscount is accumulated from start, so to determine miscount at day end we need to know the count
# at the beginning of each daily counting period - this dayStartMiscount can be determined
# 1. just after the day end time (e.g., 3am) + inactive period if within a continuous log, or
# 2. passed in from the previous log if day start is at 3am (so no prior in active period can be detected)
# skip if time gap is detected, if room is likely non-empty at designated day end
def parseLog(log, localtz, reset=True):

    countData = []
    lastProcessedTimeStamp = None
    residualLog = []
    hourlyStats = {}
    hourlyHeadCount = 0

    firstTimeStamp = log[0]['utcMSecs']
    utcdt = timestamp2datetime(firstTimeStamp)
    utcdt = utc.localize(utcdt)
    localdt = utcdt.astimezone(localtz)

    lastTimeStamp = log[-1]['utcMSecs']
    lastdt = timestamp2datetime(lastTimeStamp)
    lastdt = utc.localize(lastdt)
    lastdt = lastdt.astimezone(localtz)

    endDateTime = getEndDayDateTime(localdt, dailyEndTime, None)
    newHour     = getNextHourTime(localdt)

    # this is required if there is no reset at day end: daily count = coundDayEnd - countDayStart
    countDayStart = log[0]['data']['peopleCount']
    prePeopleCount = log[0]['data']['peopleCount']
    prelocaldt = localdt
    hourlyHeadMax = max(0,prePeopleCount)

    # this is the total number of people that have entered the room
    dayHeadCount = 0
    dayPassCount = 0
    currentDate = localdt.date()
    
    hourlyStats['hour'] = []
    hourlyStats['perHourHead'] = []
    hourlyStats['perHourMax'] = []
    hourlyStats['meanHours']   = np.zeros((len(daysList), (hourEnd - hourStart) + 1 ), np.float32) 
    hourlyStats['meanHoursMax']   = np.zeros((len(daysList), (hourEnd - hourStart) + 1 ), np.float32) 
    hourlyStats['meanDivider'] = np.zeros((len(daysList), (hourEnd - hourStart) + 1 ), np.float32)
    weekCountDict = None 
    
    lastNonZeroCountBeforeEndDay = None
    lastNonZeroCountStart = None
    lastNonZeroCountEnd = None
    accumulatedResetError = []

    timeGapFound = False
    resetDetected = False
    resetDetectedTime = None

    for i, lineDict in enumerate(log):
        currentTimeStamp = lineDict['utcMSecs']
        if currentTimeStamp == 1619531204901:
            print('It is the time')
        utcdt = timestamp2datetime(currentTimeStamp)
        utcdt = utc.localize(utcdt)
        localdt = utcdt.astimezone(localtz)
        currentPeopleCount = lineDict['data']['peopleCount']
        #numEntry = np.max((0, currentPeopleCount - prePeopleCount))


        event = lineDict['data']['eventType']
        if currentPeopleCount < 0:
            currentPeopleCount = 0
        numEntry = currentPeopleCount - prePeopleCount
        if numEntry < 0:
            if event != 'EVT_AUTO_RESET':
                dayPassCount += abs(numEntry)
                numEntry = 0
            elif event == 'EVT_AUTO_RESET':
                numEntry = 0
            else:
                numEntry = 0


        if numEntry:                                                              #??????????
            dayHeadCount += numEntry
            hourlyHeadCount += numEntry
            hourlyHeadMax = max(currentPeopleCount, hourlyHeadMax)

        if detectReset(currentPeopleCount, prePeopleCount, event, largeCountChange):
            # error not accumulated anymore
            resetDetected = True
            resetDetectedTime = localdt


        if currentPeopleCount != prePeopleCount:
            if currentPeopleCount:
                lastNonZeroCountBeforeEndDay = currentPeopleCount
                lastNonZeroCountStart = localdt
            else:                        # Why is this nonZero? it comes here when currentPeopleCount=0 >> lastNonZeroCountEnd = prelocaldt
                lastNonZeroCountEnd = localdt
                if event == 'EVT_AUTO_RESET':  # detect the reset using te flag
                    print("prePeopleCount={prePeopleCount}".format(**locals()))
                    accumulatedResetError.append(prePeopleCount)

        if not timeGapFound:
            timeGapFound = detectTimeGap(localdt, prelocaldt, timeGapThresholdSec)

        prePeopleCount = currentPeopleCount
        prelocaldt = localdt

        if isNewHour(localdt, newHour):                                   # at the end of an hour it saves the current head count for that hour 
            newHour = getNextHourTime(localdt)
            if (newHour.hour >= hourStart) and (newHour.hour<=hourEnd):
                hourlyStats['hour'].append(int(newHour.hour))
                hourlyStats['perHourHead'].append(int(hourlyHeadCount))
                hourlyStats['perHourMax'].append(int(hourlyHeadMax))
                hourlyHeadCount = 0
                hourlyHeadMax = max(0, currentPeopleCount)
                #currentDay = datetime.datetime.fromtimestamp(datetime.datetime.timestamp(utcdt)).strftime("%A")
                currentDay = daysList[utcdt.weekday()]
                hourlyStats['meanHours'][daysList.index(currentDay), 
                            hourDict['hours'].index(newHour.hour)] += hourlyStats['perHourHead'][-1] 
                hourlyStats['meanHoursMax'][daysList.index(currentDay), 
                            hourDict['hours'].index(newHour.hour)] += hourlyStats['perHourMax'][-1]                     
                hourlyStats['meanDivider'][daysList.index(currentDay),hourDict['hours'].index(newHour.hour)] += 1
                                        

        if isNewDay(localdt, endDateTime):
            currentDay = daysList[utcdt.weekday()]
            print(localdt)
            # this is at day end (eg. 3am or something like that)
            #print(currentDay)

            errorCount = 0
            if lastNonZeroCountStart is not None:
                if lastNonZeroCountEnd is None:
                    lastNonZeroCountEnd = localdt
                elif lastNonZeroCountEnd < lastNonZeroCountStart:
                    lastNonZeroCountEnd = localdt
                if (lastNonZeroCountEnd - lastNonZeroCountStart).seconds > inactivePeriodSec:
                    errorCount = lastNonZeroCountBeforeEndDay
                    if  len(accumulatedResetError) > 1:
                        for resetIndex in accumulatedResetError[:-1]:
                            errorCount += resetIndex
                elif resetDetected:
                    if resetDetectedTime >= lastNonZeroCountEnd:
                        errorCount = lastNonZeroCountBeforeEndDay
                        if len(accumulatedResetError) > 1:
                            for resetIndex in accumulatedResetError[:-1]:
                                errorCount += resetIndex
                    else:
                        for resetIndex in accumulatedResetError[:]:
                            errorCount += resetIndex
            # else:
            #     # this means no change detected at all
            #     errorCount = countDayStart

            if not reset and lastNonZeroCountStart is not None:
                errorCount = errorCount - countDayStart

            countDayStart = currentPeopleCount
                #countDayStart += errorCount
                # if there is an unexpected reset for a counter that's not supposed to reset
                # if resetDetected:
                #     if resetDetectedTime >= lastNonZeroCountEnd:
                #         countDayStart = currentPeopleCount


            dayCountDict = None
            if skipDayWithTimeGap and timeGapFound:
                print('Time gap found. Skipping ', currentDate.strftime('%m-%d'))
            else:
                dayCountDict = {'date': currentDate.strftime('%m-%d'),
                                'day': currentDay,
                                'head_count': int(dayHeadCount),
                                'pass_count': int(dayPassCount),
                                'end_day_error_count': errorCount, 'hours': hourlyStats['hour'],
                                'perHourHead': hourlyStats['perHourHead'],
                                'perHourMax': hourlyStats['perHourMax'],
                                'Num_resets': len(accumulatedResetError)}

                weekCountDict = {'meanHours': hourlyStats['meanHours'].tolist(),
                                 'meanHoursMax': hourlyStats['meanHoursMax'].tolist(),
                                 'meanDivider': hourlyStats['meanDivider'].tolist() }   

            endDateTime = getEndDayDateTime(localdt, dailyEndTime, None)
            dayHeadCount = 0
            dayPassCount = 0
            lastNonZeroCountBeforeEndDay = None
            lastNonZeroCountStart = None
            lastNonZeroCountEnd = None
            currentDate = localdt.date()
            timeGapFound = False
            resetDetected = False
            resetDetectedTime = None
            accumulatedResetError = []

            hourlyStats['hour'] = []
            hourlyStats['perHourHead'] = []  
            hourlyStats['perHourMax'] = []  
            
            lastProcessedTimeStamp = currentTimeStamp
            if dayCountDict is not None:
                countData.append(dayCountDict)


            print("errorCount={errorCount}".format(**locals()))

            if lastdt < endDateTime:
                if i < len(log) - 1:
                    # make sure i is valid within list length limit
                    residualLog = log[i + 1:]
                    break



    if  weekCountDict:      
        countData.append(weekCountDict)
    return  countData, lastProcessedTimeStamp, residualLog


def combineLogLists(log1, log2):
    log1TimeStamp = log1[-1]["utcMSecs"]

    start = None
    for i, line in enumerate(log2):
        log2TimeStamp = line["utcMSecs"]
        if log2TimeStamp > log1TimeStamp:
            start = i
            break

    if start is not None:
        log = log1 + log2[start:]
    else:
        log = log1
    return log


def processCounterLogs(logFiles, db, counter):
    residualLog = []
    countData = []
    lastProcessedTimeStamp = None
    logID = None
    logStatsDict = {}
    if db is not None:
        countData = db["countData"]
        lastProcessedTimeStamp = db["lastProcessedTimeStamp"]
        logStatsDict = db["processedFileStats"]
    for logf in logFiles:
        logfbase = os.path.basename(logf)
        if db is not None:
            if logfbase in db["processedFileStats"]:
                if isProcessed(logf, db["processedFileStats"][os.path.basename(logf)], db["lastProcessedTimeStamp"]):
                    continue
        logPath, logFile = os.path.split(logf)
        logListDict, logStats = retrieveLog(logPath, logFile, lastProcessedTimeStamp)
        logStatsDict[logfbase] = logStats
        logID = logListDict[0]["data"]["serialNum"] + '_' + logListDict[0]["data"]["mac"]

        if residualLog:
            logListDict = combineLogLists(residualLog, logListDict)
        if db is not None:
            if not verifyIDs(logID, db["ID"]):
                print("*** Error: Log and DB IDs mismatch!" + '  Log ID: ' + logID + '  DB ID: ' + db["ID"])
            #assert (verifyIDs(logID, db["ID"])), "Log and DB IDs mismatch!" + '  Log ID: ' + logID + '  DB ID: ' + db["ID"]


        logCountData, logLastProcessedTimeStamp, residualLog = parseLog(logListDict, counterDict['TimeZones'][counter], counterDict['ResetNightly'][counter])
        if logLastProcessedTimeStamp is not None:
            if lastProcessedTimeStamp is None:
                lastProcessedTimeStamp = logLastProcessedTimeStamp
            elif logLastProcessedTimeStamp > lastProcessedTimeStamp:
                lastProcessedTimeStamp = logLastProcessedTimeStamp
        # TODO: combine existing count data with the latest count data - need to check if non-unique date entries exist?
        countData = countData + logCountData

    return logID, countData, lastProcessedTimeStamp, logStatsDict


def determineDailyError(countDict):
    dailyErrCount = []
    dailyHeadCount = []
    dailyPassCount = []
    dailyResetCount = []
    dates = []
    days = []
    hours = []
    hourlyCounts = {}
    hourlyCounts['perHourHead'] = []
    hourlyCounts['perHourMax'] = []
    hourlyCounts['perHourCountWeekAv'] = np.zeros((len(daysList), (hourEnd - hourStart) + 1 ), np.float32)
    hourlyCounts['perHourMaxWeekAv'] = np.zeros((len(daysList), (hourEnd - hourStart) + 1 ), np.float32)
    hourlyCounts['perHourDivider'] = np.zeros((len(daysList), (hourEnd - hourStart) + 1 ), np.float32)
    for ii in range(len(countDict)):
        if  'meanDivider' in countDict[ii]:
            hourlyCounts['perHourMaxWeekAv'][:,:]+= countDict[ii]['meanHoursMax']                
            hourlyCounts['perHourCountWeekAv'][:,:]+= countDict[ii]['meanHours']                
            hourlyCounts['perHourDivider'][:,:]+= countDict[ii]['meanDivider']                
                        
        else:
            d = countDict[ii]
        
            dailyErrCount.append(d['end_day_error_count'])
            dailyHeadCount.append(d['head_count'])
            if 'Num_resets' in d:
                dailyResetCount.append(d['Num_resets'])

            if 'pass_count' in d:
                dailyPassCount.append(d['pass_count'])
            dates.append(d['date'])
            days.append(d['day'])
        
#             withingHours = [ h for h,ii in enumerate(d['hours']) if ((ii>=hourStart) and (ii<=hourEnd)) ]
#             if withingHours:
            hours.append(d['hours'])
            hourlyCounts['perHourHead'].append(d['perHourHead'])
            hourlyCounts['perHourMax'].append(d['perHourMax'])

#             else:
#                 hours.append(empty)
#                 hourlyCounts['perHourHead'].append(empty)                    
#                 hourlyCounts['perHourMax'].append(empty)                    
    
    hourlyCounts['perHourCountWeekAv'][:,:]/=hourlyCounts['perHourDivider'][:]+Epsil
    hourlyCounts['perHourMaxWeekAv'][:,:]  /=hourlyCounts['perHourDivider'][:]+Epsil
    hourlyCounts['perHourCountWeekAv']= (hourlyCounts['perHourCountWeekAv'])
    hourlyCounts['perHourMaxWeekAv']  = (hourlyCounts['perHourMaxWeekAv'])
    hourlyCounts['perHourCountWeekAv'] = hourlyCounts['perHourCountWeekAv'].tolist()
    hourlyCounts['perHourMaxWeekAv'] = hourlyCounts['perHourMaxWeekAv'].tolist()
    #hourlyCounts['meanDivider'] = hourlyCounts['meanDivider'][0].tolist()
    return dates, np.array(dailyErrCount), np.array(dailyHeadCount), np.array(dailyResetCount), np.array(dailyPassCount), hours, hourlyCounts, days


def determineAccumulatedError(dailyErrorCount, dailyHeadCount):
    accumulatedErrorCount = []
    accumulatedHeadCount = []

    for i in range(dailyErrorCount.shape[0]):
        accumulatedErrorCount.append(np.sum(dailyErrorCount[:i + 1]))
        accumulatedHeadCount.append(np.sum(dailyHeadCount[:i + 1]))
    return np.array(accumulatedErrorCount), np.array(accumulatedHeadCount)

def plotGraphsNo(xlabels, errCount, totCount, title=None, savePath=None, fileName=None):
	n = len(xlabels)
	ymin = np.min(errCount)
	ymax = np.max(errCount)
	yt = np.arange(ymin, ymax + 1)
	x = np.arange(n)
	fig = plt.figure(np.random.randint(0, 1000))
	fig.set_size_inches(15, 12)
	plt.subplots_adjust(hspace=0.35)
	plt.subplot(3, 1, 1)
	plt.bar(x, errCount, color='orange')
	plt.xlim(-1, n)
	plt.xticks(x, xlabels, rotation=90)
	plt.yticks(yt)
	# matplotlib.ticker.MaxNLocator(integer=True)
	plt.title('Error count')
	plt.subplot(3, 1, 2)
	plt.bar(x, totCount, color='blue')
	plt.xlim(-1, n)
	plt.xticks(x, xlabels, rotation=90)
	plt.title('Head count')
	plt.subplot(3, 1, 3)
	plt.bar(x, np.abs(errCount / (totCount+Epsil)) * 100, color='green')
	plt.xlim(-1, n)
	plt.xticks(x, xlabels, rotation=90)
	plt.ylabel('% error')
	plt.title('Error rate')
	plt.xlabel('Date')

	if title is not None:
		plt.suptitle(title, fontsize=20, y=0.94)

	if savePath is not None and fileName is not None:
		if not os.path.exists(savePath):
			os.makedirs(savePath)
		plt.savefig(savePath + os.sep + fileName, dpi=150, bbox_inches='tight', pad_inches=0.1)
		plt.close()
	return None

def plotGraphs(xlabels, errCount, totCount, title=None, savePath=None, fileName=None):
    if len(totCount) == 1:
        totCount = totCount[0]
    else:
        if  headVSpass_flag == 0:               # head count is used to calculate the accuracy: TC = dailyHeadCount
            totCount = totCount[0]
        else:
            totCount = totCount[1]              # pass count is used to calculate the accuracy: TC = dailyPassCount = entry+exit
    n = len(xlabels)
    ymin = np.min(errCount)
    ymax = np.max(errCount)
    yt = np.arange(ymin, ymax + 1)
    x = np.arange(n)
    #fig = plt.figure(np.random.randint(0, 1000))
    #fig, ax = plt.subplots(np.random.randint(0, 1000))
    fig, ax = plt.subplots(3, 1)
    fig.set_size_inches(15, 12)
    plt.subplots_adjust(hspace=0.35)
    #plt.subplot(3, 1, 1)
    ax[0].bar(x, errCount, color='orange')
    ax[0].set_xlim(-1,n)
    #plt.xlim(-1, n)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(xlabels, rotation=90)
    ax[0].set_yticks(yt)
    # matplotlib.ticker.MaxNLocator(integer=True)
    ax[0].set_title('Error count')
    #plt.subplot(3, 1, 2)
    Labell = 'Head count'
    sensor_accuracy = None
    if  fileName[-15:] == 'daily_error.png':
        sensor_accuracy = 0
        accuracy = ['0']*len(totCount)
        for i in range(len(errCount)):
            if totCount[i] != 0:
                accuraci = int(100 * (1 - (errCount[i] / (totCount[i] + Epsil))))
                accuracy[i] = chr(37) + str(accuraci)
                sensor_accuracy += accuraci*totCount[i]
            else:
                accuracy[i] = ' '
        sensor_accuracy = int(round(sensor_accuracy/(sum(totCount)+Epsil)))
        ax[1].bar(x, totCount, yerr= errCount, color='blue')
        #labels = [chr(37) + "%d" % i for i in range(len(errCount))]
        labels = [i for i in accuracy]
        rects = ax[1].patches
        for rect, label in zip(rects, labels):
            height = rect.get_height()
            ax[1].text(rect.get_x() + rect.get_width() / 2, height + 0, label, ha='left', va='bottom')

        Labell +=' with mean accuracy of ' + chr(37) + str(sensor_accuracy)

    else:
        ax[1].bar(x, totCount, color='blue')

    ax[1].set_xlim(-1, n)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(xlabels, rotation=90)
    ax[1].set_title(Labell)
    #plt.subplot(3, 1, 3)
    #plt.bar(x, np.abs(errCount / (totCount+Epsil)) * 100, color='green')
    ax[2].bar(x, np.abs(errCount / (totCount+Epsil)) * 100, color='green')
    ax[2].set_xlim(-1, n)
    ax[2].set_xticks(x)
    ax[2].set_xticklabels(xlabels, rotation=90)
    ax[2].set_ylabel('% error')
    ax[2].set_title('Error rate')
    ax[2].set_xlabel('Date')

    if title is not None:
        plt.suptitle(title, fontsize=20, y=0.94)

    if savePath is not None and fileName is not None:
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        plt.savefig(savePath + os.sep + fileName, dpi=150, bbox_inches='tight', pad_inches=0.1)
        plt.close()
    return sensor_accuracy

def plotHourlyGraphs(allHours, hourlyCount, dayLabels, title=None, savePath=None):
    print(title)
    month = monthList[int(dayLabels[0][0][0:2])]
    for dayIndex, day in enumerate(dayLabels[1]):
        if allHours[dayIndex]:
            
            totlCounts = []
            n = len(allHours[dayIndex])
            for i in range(n):
                totlCounts.append(sum(hourlyCount['perHourHead'][dayIndex][0:i+1]))
            
            fig = plt.figure(np.random.randint(0, 1000))
            fig.set_size_inches(15, 13)
            plt.subplots_adjust(hspace=0.35)
            start = hourDict['hours'].index(allHours[dayIndex][0])
            end   = hourDict['hours'].index(allHours[dayIndex][-1])
            
            plt.subplot(3, 1, 1)           
            plt.bar(hourDict['labels'], hourlyCount['perHourCountWeekAv'][daysList.index(day)],
                       color='#0000FF', align='center', width=-.9, edgecolor='#AAAAFF', alpha=0.5)  

            plt.bar(hourDict['labels'][start:end+1], 
                       hourlyCount['perHourHead'][dayIndex],                       
                       color='#0000FF', align='center', width=-.6, edgecolor='#AAAAFF')
            
            plt.ylabel('person')
            plt.xlim([hourDict['labels'][start], hourDict['labels'][end]])
            plt.xlabel('hour of the day')
            plt.xticks(hourDict['labels'][start:end+1], hourDict['labels'][start:end+1], rotation=90)
            plt.title('Hourly head count')
            plt.legend(['on a typical ' + day, 'this ' + day +' of ' + month])
          
            plt.subplot(3, 1, 2)
            plt.bar(hourDict['labels'][start:end+1], totlCounts, color='green', align='center', width=-.9)
            plt.xlim([hourDict['labels'][start], hourDict['labels'][end]])
            plt.ylabel('person')
            plt.xlabel('hour of the day')
            plt.xticks(hourDict['labels'][start:end+1], hourDict['labels'][start:end+1], rotation=90)
            plt.title('Accumulated hourly head count')
    
            plt.subplot(3, 1, 3)
            plt.bar(hourDict['labels'], hourlyCount['perHourMaxWeekAv'][daysList.index(day)],
                       color='#FF0000', align='center', width=-.9, edgecolor='#FFAAAA', alpha=0.5)  

            plt.bar(hourDict['labels'][start:end+1], 
                       hourlyCount['perHourMax'][dayIndex],                       
                       color='#FF0000', align='center', width=-.6, edgecolor='#FFAAAA')
            
            plt.ylabel('person')
            plt.xlim([hourDict['labels'][start], hourDict['labels'][end]])
            plt.xlabel('hour of the day')
            plt.xticks(hourDict['labels'][start:end+1], hourDict['labels'][start:end+1], rotation=90)
            plt.title('Hourly head count')
            plt.legend(['on a typical ' + day, 'this ' + day +' of ' + month])
            plt.title('Maximum hourly head count')
            plt.legend(['on a typical ' + day, 'this ' + day +' of ' + month])
            if title is not None:
                plt.suptitle(title, fontsize=20, y=0.94)
    
            if savePath is not None and dayLabels is not None:
                if not os.path.exists(savePath):
                    os.makedirs(savePath)
                plt.savefig(savePath + os.sep + dayLabels[0][-1] + '_' + dayLabels[0][dayIndex] + '_hourly.png', dpi=150, bbox_inches='tight', pad_inches=0.1)
                plt.close()
    return None


def displayHTML(htmlPath, filename):
    fullPath = htmlPath + os.sep + filename
    webbrowser.open_new_tab(fullPath)
    return None


# database format:
# db = {"ID": str, "lastProcessedTimeStamp": int, "TZ": str, "processedFileStats": dict, "countData": list}
def main(logPath, databasePath, outPath, backupPath, HTML_Path):
    for counter in range(len(counterDict['Names'])):
    #for counter in range(3):
    #    counter = counter + 0
        db = loadCounterDatabase(databasePath, counterDict['Names'][counter])
        #db = None
        logFiles = getCounterLogs(logPath, counterDict['Names'][counter])
        print("Processing counter:", counterDict['Names'][counter])
        if logFiles:

            counterID, countData, lastProcTimestamp, logFileStats = processCounterLogs(logFiles, db, counter)
            xlabels, dailyErrorCount, dailyHeadCount, dailyResetCount, dailyPassCount, allHours, hourlyCounts, days = determineDailyError(countData)
            accErrorCount, accHeadCount = determineAccumulatedError(dailyErrorCount, dailyHeadCount)
            sensor_accuracy = plotGraphs(xlabels,
                       dailyErrorCount,
                       [dailyHeadCount, dailyPassCount],
                       counterDict['Names'][counter] + ' Daily door count',
                       outPath,
                       counterDict['Names'][counter] + '_daily_error.png')
            plotGraphs(xlabels,
                       accErrorCount,
                       [accHeadCount],
                       counterDict['Names'][counter] + ' Accumulated door count',
                       outPath,
                       counterDict['Names'][counter] + '_accumulated_error.png')
            xlabels.append(counterDict['Names'][counter])
            plotHourlyGraphs(allHours,  hourlyCounts,
                       [xlabels, days],
                       counterDict['Names'][counter] + ' hourly door count',
                       outPath)
            jsonData = serializeData(counterID, lastProcTimestamp, counterDict['TimeZones'][counter].zone, logFileStats, countData)
            backupDB(backupPath, databasePath, counterDict['Names'][counter])
            #saveCounterDatabase('/home/jacob/Downloads/', counterDict['Names'][counter], jsonData)
            saveCounterDatabase(databasePath, counterDict['Names'][counter], jsonData)
            StatDict['Names'].append(counterDict['Names'][counter])
            StatDict['total counts'].append(sum(accHeadCount))
            StatDict['miscounts'].append(sum(accErrorCount))
            StatDict['number of resets'].append(sum(dailyResetCount))
            StatDict['accuracy'].append(sensor_accuracy)
            StatDict['period'].append([xlabels[0] + ' to ' + xlabels[-2]])
    createHTML(outPath, HTML_Path)
    createCSV(HTML_Path)
    return None

#%%
if __name__ == "__main__":
    logPath = '/home/jacob/Documents/Code/Python/Calumino/data/logs'
    databasePath = '/home/jacob/Documents/Code/Python/Calumino/data/logs/database2'
    outputPath = '/home/jacob/Documents/Code/Python/Calumino/data/logs/plots'
    #webSrcPath = r'C:\Users\eric\Downloads\data\210209 daily logs\web'
    backUpPath = '/home/jacob/Documents/Code/Python/Calumino/data/logs/backup'
    #webpage = 'counter_results.html'
    HTML_Path = '/home/jacob/Documents/Code/Python/Calumino/'

    main(logPath, databasePath, outputPath, backUpPath, HTML_Path)
    #displayHTML(webSrcPath, webpage)
    #write_XCEL(XCEL_path=HTML_Path)
    sys.exit()
