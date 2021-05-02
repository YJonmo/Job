'''
Code by Yaqub 19/4/2021
# Copyright (C) Calumino. All rights reserved.
# Principal author: Yaqub Jonmohamadi (Calumino 2021).
'''
from __future__ import division
import sys
import cv2
import numpy as np
from cogency.cobus import pipeline as putil
from calumino.config import profiles
from cogency.utils import general as gutil

#from heatmapoptions import HeatMapOptions
import datetime, pytz
import glob
import os
#%% ########################################### USER INPUTS (Linux) #################################################
##opts = HeatMapOptions()
##options = opts.parse()
options = {}
options['dataPath']      = '/home/jacob/Documents/Code/Python/Calumino'
options['configFilePath']= '/home/jacob/Documents/Code/Python/Calumino/pipeline.json'
options['sensorNames']   = ['00002b399125']
options['startTimeDate'] = [2021, 4, 15, 18, 59, 52]
options['endTimeDate']   = [2021, 4, 15, 19, 13, 30]
options['timeZone']      = 'Australia/Sydney'
options['fileFormat']    = 'rdat'
#%% #########################################################################################################


#%% ########################################### USER INPUTS (Windows) #################################################
# #opts = HeatMapOptions()
# #options = opts.parse()
# options = {}
# options['dataPath']      = r'C:\ubuntu-data-exchange\SampleYaqub'
# options['configFilePath']= r'C:\ubuntu-data-exchange\pipeline.json'
# # options['sensorNames']   = ['0000906731c9', '0000bd1db869']
# options['sensorNames']   = ['00002b399125']
# options['startTimeDate'] = [2021, 4, 15, 19, 5, 52]
# options['endTimeDate']   = [2021, 4, 15, 19, 9, 50]
# options['timeZone']      = 'Australia/Sydney'
# options['fileFormat']    = 'rdat'
#%% #########################################################################################################

#File_pre = r'C:\Users\NUC - CALUMINO\Desktop\WebcamRecordings\201026 Extraction\0000906731c9'
#os.path.join(File_pre)
#glob.glob(File_pre + os.sep + '*' )
# dir_name = r'C:\Users\Felix\Desktop\rawData\demo.thermal.ai\SMB_testing_office\0000b3e4d090-fall1'
# file_name = '0000b3e4d090_20210301032203_763000'
# out_dir = r'C:\Users\Felix\Dropbox (Calumino)\CALUMINO_FELIX\txt raw'
# # dataOutPath = dir_name + os.sep + 'rawData'
# dataOutPath = out_dir + os.sep + file_name + '_tempF' + str(tempFactor_v)


dateDict = {'dateTime': ['year', 'month', 'day', 'hour', 'minute', 'second'] }
blobHeatMaps = {'blobID':[], 'type': [], 'age': [], 'arrowField': [], 'heatMap': [] }
width = 38
height = 17
epsilon = 0.001


UTC_Zone = pytz.utc
localZone = pytz.timezone(options['timeZone'])

print(options)

def appendDate(date):
    draftDate = datetime.datetime(2000,1,1)
    for i in range(len(date)):
        if i == len(dateDict['dateTime']):
            print('Date format error. It only accepts yyyy mm dd hh mm ss format with minimum yyyy mm dd as inputs')
        else:
            draftDate = draftDate.replace(**{dateDict['dateTime'][i]: date[i]})
    return draftDate

def arrowedPath(arrowField, arrows):
    for i in range(len(arrows)-1):
        cv2.arrowedLine(arrowField, (arrows[i][0]*2, arrows[i][1]*2), (arrows[i+1][0]*2, arrows[i+1][1]*2), (255, 255, 255), 1, 8, 0, 0.3)
    return arrowField

def updateHeatPlot(heatMap, arrowField):
    Keys = list(blobHeatMaps.keys())
    for blobID in blobHeatMaps['blobID']:
        blobIndex = blobHeatMaps['blobID'].index(blobID)
        if blobHeatMaps['type'][blobIndex] == 'person':
            #heatMap += np.squeeze(blobHeatMaps['heatMap'][0][blobIndex, :, :]) * blobHeatMaps['age'][blobIndex]
            heatMap += np.squeeze(blobHeatMaps['heatMap'][0][blobIndex, :, :])
            arrowField = arrowedPath(arrowField, blobHeatMaps['arrowField'][blobIndex])
        for Key in Keys[:-1]:
            blobHeatMaps[Key].remove(blobHeatMaps[Key][blobIndex])
        if len(blobHeatMaps['heatMap'][0]) == 1:
            blobHeatMaps['heatMap'] = []
            #blobHeatMaps['arrowField'] = []
        else:
            blobHeatMaps['heatMap'][0] = np.delete(blobHeatMaps['heatMap'][0], blobIndex, 0)
            #blobHeatMaps['arrowField'][0] = np.delete(blobHeatMaps['arrowField'][0], blobIndex, 0)

    heatPlot = (np.log(heatMap+1) / (np.max(np.log(heatMap+1)) + epsilon)) * 255
    heatPlot = cv2.applyColorMap(heatPlot.astype(np.uint8), cv2.COLORMAP_JET)
    return heatMap, heatPlot, arrowField


def main():
    startTimeDate = appendDate(options['startTimeDate'])
    startTimeDate = localZone.localize(startTimeDate)
    startTimeDate = startTimeDate.astimezone(UTC_Zone)

    endTimeDate = appendDate(options['endTimeDate'] )
    endTimeDate = localZone.localize(endTimeDate)
    endTimeDate = endTimeDate.astimezone(UTC_Zone)

    startFrame = int(datetime.datetime.timestamp(startTimeDate)*1000)
    print('Start time and date: ' + str(startTimeDate) + 'time zone: ' + str(startTimeDate.tzinfo))
    lastFrame  = int(datetime.datetime.timestamp(endTimeDate)*1000)
    print('End time and date: ' + str(endTimeDate) + 'time zone: ' + str(endTimeDate.tzinfo))
    #startFrame = 4800
    #lastFrame = 16200
    #files = []
    for sensorIndex in range(len(options['sensorNames'])):
        files_path = os.path.join(options['dataPath'] + os.sep + options['sensorNames'][sensorIndex]
                                + os.sep )
        files = sorted(glob.glob(files_path + options['sensorNames'][sensorIndex] + "*." + options['fileFormat']))

        if  files:
            configFilePath = options['configFilePath']
            debugWinWidth = 16 * 4 * 6
            debugWinHeight = int(debugWinWidth * (16 / 36))
            heatMap = np.zeros((height*4, width*4), np.float64)
            arrowField = np.zeros((height*8, width*8), np.uint8)

            heatPlot = []
            print(files[0])
            config = putil.LoadConfigFromFile(configFilePath, None)
            config = profiles.LoadProfileWithSource(files[0],config = config, recodeTimestamps=False)

            putil.SetGlobalParameter(config, 'showGui', 1)
            putil.SetGlobalParameter(config, 'pipelineType', 'device')
            putil.SetGlobalParameter(config, 'deviceType', 'cts-single')
            #putil.SetNodeParameter(config, 'src', 'showGui', 1)
            putil.SetNodeParameter(config, 'segmentor', 'showGui', 1)
            putil.SetNodeParameter(config, 'doorcounter', 'showGui', 1)
            putil.SetNodeParameter(config, 'tracker', 'showGui', 1)
            pipeline = putil.ActivatePipeline(config)

            pipeline.seekTimestamp(startFrame)
            #index = startFrame

            pipeline.setDebugViewSettings(cols=2, size=[debugWinHeight, debugWinWidth])
            cv2.namedWindow('heat map', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('heat map', 280, 180)
            cv2.namedWindow('arrow field', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('arrow field', 280, 180)
            while pipeline.isRunning:
                print('.....................step %d' % (pipeline.position()))
                print('.......................first timestamp: ' + str(startFrame))
                print('.....................current timestamp: ' + str(pipeline.dataCache.registry['src']['timestamp']))
                print('........................last timestamp: ' + str(lastFrame))
                pipeline.step()

                if pipeline.dataCache.registry['src']['timestamp']:
                    if pipeline.dataCache.registry['src']['timestamp'] > lastFrame:
                        heatMap, heatPlot, arrowField = updateHeatPlot(heatMap, arrowField)
                        cv2.imshow('heat map', heatPlot)
                        cv2.imshow('arrow field', arrowField)
                        cv2.waitKey(1000)
                        pipeline.stop()
                        pipeline.closeall()
                        break
                    else:
                        if pipeline.dataCache.registry['tracker']['trk']:  # check to see if there is any blob
                            for indexObject in range(len(pipeline.dataCache.registry['tracker']['tstate'][
                                                             'trackedObjects'])):  # for every blob try to see if is new
                                print(pipeline.dataCache.registry['tracker']['tstate']['trackedObjects'][indexObject][
                                          'type'])
                                noBackGround = cv2.resize(pipeline.dataCache.registry['segmentor']['tmr'],
                                                          (width * 4, height * 4),
                                                          interpolation=cv2.INTER_LINEAR)
                                try:  # if the blob is not new, then update that blob in the blobHeatMaps
                                    print(
                                        pipeline.dataCache.registry['tracker']['tstate']['trackedObjects'][0]['trail'])
                                    blobIndex = blobHeatMaps['blobID'].index(
                                        pipeline.dataCache.registry['tracker']['tstate']['trackedObjects'][indexObject][
                                            'blobID'])
                                    blobHeatMaps['blobID'][blobIndex] = \
                                        pipeline.dataCache.registry['tracker']['tstate']['trackedObjects'][indexObject][
                                            'blobID']
                                    blobHeatMaps['age'][blobIndex] = \
                                        pipeline.dataCache.registry['tracker']['tstate']['trackedObjects'][indexObject][
                                            'age']
                                    if 7 ==pipeline.dataCache.registry['tracker']['tstate']['trackedObjects'][indexObject]['blobID']:
                                        print(pipeline.dataCache.registry['tracker']['tstate']['trackedObjects'][indexObject]['trail'])
                                    blobHeatMaps['arrowField'][blobIndex].append(pipeline.dataCache.registry['tracker'][
                                        'tstate']['trackedObjects'][indexObject]['trail'][-1].copy())

                                    blobHeatMaps['heatMap'][0][blobIndex, :, :] += \
                                        cv2.bitwise_and(noBackGround, noBackGround,
                                                        mask=pipeline.dataCache.registry['tracker']
                                                        ['tstate']['trackedObjects'][indexObject]['blobFeatures'][
                                                            'mask'])
                                    if blobHeatMaps['type'][blobIndex] != 'person':
                                        blobHeatMaps['type'][blobIndex] = \
                                            pipeline.dataCache.registry['tracker']['tstate']['trackedObjects'][
                                                indexObject]['type']
                                except:  # if the blob is new then add it to the blobHeatMaps
                                    blobHeatMaps['blobID'].append(
                                        pipeline.dataCache.registry['tracker']['tstate']['trackedObjects'][indexObject][
                                            'blobID'])
                                    blobHeatMaps['type'].append(
                                        pipeline.dataCache.registry['tracker']['tstate']['trackedObjects'][indexObject][
                                            'type'])
                                    # blobHeatMaps['mask'].append(pipeline.dataCache.registry['tracker']['tstate']['trackedObjects'][indexObject]['mask'])
                                    blobHeatMaps['age'].append(
                                        pipeline.dataCache.registry['tracker']['tstate']['trackedObjects'][indexObject][
                                            'age'])
                                    if 7 ==pipeline.dataCache.registry['tracker']['tstate']['trackedObjects'][indexObject]['blobID']:
                                        print('here')
                                    blobHeatMaps['arrowField'].append(pipeline.dataCache.registry['tracker']
                                        ['tstate']['trackedObjects'][indexObject]['trail'].copy())

                                    stackHeatMap = np.expand_dims(
                                        cv2.bitwise_and(noBackGround, noBackGround,
                                                        mask=pipeline.dataCache.registry['tracker']
                                                        ['tstate']['trackedObjects'][indexObject]['blobFeatures'][
                                                            'mask']), axis=0)
                                    if len(blobHeatMaps['heatMap']) == 0:
                                        blobHeatMaps['heatMap'].append(stackHeatMap)
                                    else:
                                        blobHeatMaps['heatMap'][0] = np.concatenate(
                                            (blobHeatMaps['heatMap'][0], stackHeatMap), axis=0)


                        elif blobHeatMaps['blobID'] != []:
                            heatMap, heatPlot, arrowField = updateHeatPlot(heatMap, arrowField)
                            cv2.imshow('heat map', heatPlot)
                            cv2.imshow('arrow field', arrowField)
                            cv2.waitKey(1)
                else:
                    heatMap, heatPlot, arrowField = updateHeatPlot(heatMap, arrowField)
                    cv2.imshow('heat map', heatPlot)
                    cv2.imshow('arrow field', arrowField)
                    cv2.waitKey(1000)
                    break

            if len(heatPlot)!=0:
                heatPlot = cv2.resize(heatPlot,(width * 8, height * 8), interpolation=cv2.INTER_LINEAR)
                image_label = str(startTimeDate.date()) + ' ' + str(startTimeDate.hour).rjust(2,'0') + ' ' + str(
                    startTimeDate.minute).rjust(2,'0') + ' ' + str(startTimeDate.second).rjust(2,'0')
                cv2.imwrite(os.path.join(files_path+image_label+'_HeatMap.png'), heatPlot)
                #cv2.imwrite(os.path.join(files_path+str(startTimeDate)+' to '+str(endTimeDate)+'HeatMap.png'), heatPlot)
                cv2.imwrite(os.path.join(files_path+image_label+'_Arrow.png'), arrowField)
                print('Results are saved in: ' + files_path + '\n')
        else:
            print('###################################################################################' )
            print('No file was found for this sensor: ' + str(options['sensorNames'][sensorIndex]) + ' for the given time window.')
            print('###################################################################################')
    sys.exit()

if __name__ == "__main__":
    main()

