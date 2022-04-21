# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 13:47:01 2018

@author: yaqub
"""

import numpy as np
import cv2
import glob
import os.path

Path_Current = os.path.abspath(os.path.join( os.getcwd()))
Path_Image = '/home/yaqub/Documents/Robotics/Images/Contouring Samples/Contouring Samples/AllTheImages'
Path_Save =  '/home/yaqub/Documents/Robotics/Images/Contouring Samples/Contouring Samples/AllTheImages_2'
os.chdir(Path_Image)
images = glob.glob('*.png')


for fname in images:
    
    os.chdir(Path_Image)
    frame = cv2.imread(fname)
    Ret, thresholded= cv2.threshold(frame, 130, 255 ,cv2.THRESH_BINARY)
    os.chdir(Path_Save)
    cv2.imwrite(fname, thresholded)
    
#cv2.threshold





Path_Image = '/home/yaqub/Documents/Robotics/Images/Contouring Samples/Contouring Samples/AllTheContours_0_1'
Path_Save =  '/home/yaqub/Documents/Robotics/Images/Contouring Samples/Contouring Samples/AllTheContours_0_1'
os.chdir(Path_Image)
images = glob.glob('*.tif')
Save_Name =list('Sen[1]23-02-2018 15-02-09_Frame00060_mask.tif')

for fname in images:
       
    Save_Name[0:36] = str(fname[:36])
    frame = cv2.imread(fname)
    Save_Name2 = ''.join(Save_Name)     
    print(Save_Name2) 
    cv2.imwrite(Save_Name2, frame)




Path_Image = '/home/yaqub/Documents/Robotics/Images/Contouring Samples/Contouring Samples/AllTheContours_0_1'
Path_Save =  '/home/yaqub/Documents/Robotics/Images/Contouring Samples/Contouring Samples/AllTheContours_0_1'
os.chdir(Path_Image)
images = glob.glob('*.tif')

for fname in images:
    
    os.chdir(Path_Image)
    frame = cv2.imread(fname, cv2.CV_8UC1)
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    os.chdir(Path_Save)
    cv2.imwrite(fname, frame[:,:])
    


# this is for adding the pad to the iamges.

Path_Image = '/home/yaqub/Documents/Robotics/Images/Contouring Samples/Contouring Samples/AllTheImages_RGB'
os.chdir(Path_Image)
images = sorted(glob.glob('*.tif'))
#globalframe = np.zeros((256,256), np.uint8)
globalframe = np.zeros((256,256,3), np.uint8)
I = 0 
for  fname in images:
    #frame = cv2.imread(fname, cv2.CV_8UC1)
    frame = cv2.imread(fname)
    globalframe[0:250,0:250,:] = frame
    cv2.imwrite(str(I) + '.tif', globalframe)
    I += 1
    
    
# This is aparently for changing the name of the images.
    
Path_label = '/home/yaqub/Documents/Robotics/Codes/Python/unet-master/data/train3_Augmented/label'
os.chdir(Path_label)
images = sorted(glob.glob('*.tif'))
frame = cv2.imread(images[0], cv2.CV_8UC1)
globalframe = np.zeros(frame.shape, np.uint8)
I = 0 
for  fname in images:
    frame = cv2.imread(fname, cv2.CV_8UC1)
#    globalframe[0:256,0:256] = frame
    globalframe[0:256,0:256] = frame
#    cv2.imwrite(str(I) + '_mask.tif', globalframe)
    cv2.imwrite(fname[:-4] + '_mask.tif', globalframe)
    I += 1







'''
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 10:29:46 2018

@author: yaqub
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 12:05:23 2018
@author: yjon701
"""

import numpy as np
import cv2
import os.path
import time
import datetime
from multiprocessing import Process, Value, Array
import multiprocessing
import matplotlib.pyplot as plt





def globalVariables():
    # Get the width and height of frame
#    width = int(capR.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
#    height = int(capR.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
    global width  
    width = 384
    global height 
    height = 384
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    global fourcc
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    

    
    
def cam_loopR(pipe_parentR):
    capR = cv2.VideoCapture(0) 
    while True:
        _ , img = capR.read()
        if img is not None:
            pipe_parentR.send(img)
        if RecordCommand == 3:
            capR.release()
            
def cam_loopL(pipe_parentL):
    capL = cv2.VideoCapture(1) 
    while True:
        _ , imgL = capL.read()
        if imgL is not None:
            pipe_parentL.send(imgL)
        if RecordCommand == 3:
            capL.release()
                
        
def show_loop_stereo(pipe_childR, pipe_childL):
    RecordCommand = 5 
    stereoframe = np.zeros((height,width*2, 3), np.uint8)
    RighAndLeft_frame =  np.zeros((height,width*2, 3), np.uint8)
    RighAndLeft_frame[:,0:width,:] =  RighAndLeft_frame[:,0:width,:] + 100  # Right fram is the bright one 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    File_name_PreFix = str('%s' %datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S'))
    VideoRL = cv2.VideoWriter(File_name_PreFix, fourcc, 30.0, (height, width*2))
    while True:
        from_queueR = pipe_childR.recv()
        #cv2.imshow('Right', from_queueR)
        
        from_queueL = pipe_childL.recv()
        #cv2.imshow('Left', from_queueL)
        stereoframe[:,0:width,:] = from_queueR 
        stereoframe[:,width:,:]  = from_queueL     
        cv2.imshow('Frame', stereoframe)
        KeyBoard =  cv2.waitKey(10) & 0xFF 
        if   KeyBoard == ord('r'):    
            RecordCommand = 0
        elif KeyBoard == ord('s'):     
            RecordCommand = 2
        elif KeyBoard == ord('q'):
            RecordCommand = 3
            
            
        if RecordCommand == 1:
            VideoRL.write(stereoframe)
        elif RecordCommand == 2:
            VideoRL.release()
        elif RecordCommand == 0:
            File_name_PreFix = str('%s' %datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S'))
            VideoRL = cv2.VideoWriter((File_name_PreFix + '.mp4'), fourcc, 30.0, (width*2, height*1))
            RecordCommand = 1

            CalibFile = h5py.File((File_name_PreFix +'.hdf5'),'w')
            keys = list(CalibFile.keys())
            FrameR =     np.array(CalibFile[keys[0]+'/Time'])
            FrameL =     np.array(CalibFile[keys[0]+'/Time'])
            
            stereoframe = RighAndLeft_frame
            VideoRL.write(stereoframe)            
            
        elif RecordCommand == 3:
            VideoRL.release()
            CalibFile.close()
            break
    
    
    
  
            
 
def show_loopR(pipe_childR):
    cv2.namedWindow('pepe')
 
    while True:
        from_queue = pipe_childR.recv()
        cv2.imshow('pepe', from_queue)
        cv2.waitKey(1)  
    

            
def show_loopL(pipe_childL):
    plt.figure()
    #cv2.namedWindow('pepeLasda')
 
    while True:
        from_queueL = pipe_childL.recv()
        cv2.imshow('pepeLasda', from_queueL)
        #cv2.waitKey(1)  
        plt.imshow(from_queueL)
        plt.show()
        




def CameraReaderR(Video_Name_R):
    capR = cv2.VideoCapture(1)
    print(1)
#    print(fourcc)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    print(2)
    VideoR = cv2.VideoWriter('Video_Name_R.mp4', fourcc, 30.0, (width, height))
    print(3)
    while (CamR_Is_Read.value != 2):
        retR, frameR = capR.read()
        CamR_Is_Read.value = 1
        print(frameR)
        cv2.imshow('Right frame', frameR)
        cv2.waitKey(1) 
        print(5)        
        if RecordCommand == 1:
            VideoR.write(frameR)
         
    VideoR.release()
    capR.release()
        

def CameraReaderL(Video_Name_L):
    capL = cv2.VideoCapture(0)
    VideoL = cv2.VideoWriter(Video_Name_L, fourcc, 30.0, (width, height))
    while CamL_Is_Read.value != 2:
        retL, frameL = capR.read()
        CamL_Is_Read.value = 1
        if RecordCommand == 1:
            VideoL.write(frameL)
    
    VideoL.release()
    capL.release()
        


if __name__ == "__main__":  
    globalVariables()
    CamR_Is_Read = Value('i', 0)
    CamR_Is_Read.value = 0
    CamL_Is_Read = Value('i', 0)
    CamL_Is_Read.value = 0
    RecordCommand = Value('i', 0)
    RecordCommand.value = 5
    BlankFrame = np.zeros((height, width,3), np.uint8)
    while(RecordCommand != 3):

#        if (CamR_Is_Read == 1) & (CamL_Is_Read == 1):
#            stereoframe[:,0:width,:] = frameR 
#            stereoframe[:,width:,:]  = frameL 
#            cv2.imshow('frame Right', stereoframe)
#            CamR_Is_Read = 0
#            CamL_Is_Read = 0
            
        cv2.imshow('blank frame', BlankFrame)
        KeyBoard =  cv2.waitKey(10) & 0xFF
        if   KeyBoard == ord('h'):    
            print('rrrrrrrrrrrrrrrrrrrrrrrrrrrr')
#            File_name_PreFix = str('%s' %datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S'))
#            Video_Name_R = File_name_PreFix + '-' + 'R' + '.mp4'
#            Video_Name_L = File_name_PreFix + '-' + 'L' + '.mp4'
#            Pros_CameraReaderR = Process(target=CameraReaderR, args=(Video_Name_R, ))
#            Pros_CameraReaderR.start() 
#            #Pros_CameraReaderL = Process(target=CameraReaderR, args=(len(Video_Name_L), ))
#            #Pros_CameraReaderL.start() 
            
        elif  KeyBoard == ord('r'):
            logger = multiprocessing.log_to_stderr()
            logger.setLevel(multiprocessing.SUBDEBUG)
         
            pipe_parentR, pipe_childR = multiprocessing.Pipe()
            pipe_parentL, pipe_childL = multiprocessing.Pipe()
         
            cam_processR = multiprocessing.Process(target=cam_loopR,args=(pipe_parentR, ))
            cam_processR.start()
            cam_processL = multiprocessing.Process(target=cam_loopL,args=(pipe_parentL, ))
            cam_processL.start()

            
         
#            show_processR = multiprocessing.Process(target=show_loopR,args=(pipe_childR, ))
#            show_processR.start()
#            show_processL = multiprocessing.Process(target=show_loopL,args=(pipe_childL, ))
#            show_processL.start()
            show_processRL = multiprocessing.Process(target=show_loop_stereo,args=(pipe_childR, pipe_childL, ))
            show_processRL.start()
            
         
            cam_processR.join()
            #show_loopR.join()
            cam_processL.join()
            #show_loopL.join()
            show_loop_stereo.join()
                        
        elif KeyBoard == ord('s'):
            CamR_Is_Read.value = 2 
            CamL_Is_Read.value = 2
            time.sleep(0.100)
            Pros_CameraReaderR.terminate()
            Pros_CameraReaderL.terminate()            
                        
                        
        elif KeyBoard == ord('q'):
            CamR_Is_Read.value = 2 
            CamL_Is_Read.value = 2
            time.sleep(0.100)
            Pros_CameraReaderR.terminate()
            Pros_CameraReaderL.terminate()    
            break
    
  

        # Display the resulting frame
    #    cv2.imshow('frame Right',frameR)
    #    cv2.imshow('frame Left' ,frameL)
        

    # When everything done, release the capture

    cv2.destroyAllWindows()























# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 10:29:46 2018

@author: yaqub
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 12:05:23 2018
@author: yjon701
"""

import numpy as np
import cv2
import os.path
import time
import datetime
from multiprocessing import Process, Value, Array
import multiprocessing
import matplotlib.pyplot as plt





def globalVariables():
    # Get the width and height of frame
#    width = int(capR.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
#    height = int(capR.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
    global width  
    width = 384
    global height 
    height = 384
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    global fourcc
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    

    
    
def cam_loopR(pipe_parentR):
    capR = cv2.VideoCapture(0) 
    while True:
        _ , img = capR.read()
        if img is not None:
            pipe_parentR.send(img)
        if RecordCommand == 3:
            capR.release()
            
def cam_loopL(pipe_parentL):
    capL = cv2.VideoCapture(1) 
    while True:
        _ , imgL = capL.read()
        if imgL is not None:
            pipe_parentL.send(imgL)
        if RecordCommand == 3:
            capL.release()
                
        
def show_loop_stereo(pipe_childR, pipe_childL):
    RecordCommand = 5 
    Frame_Index = 0
    Time_Right = np.zeros((9000), np.float64)
    Time_Left = np.zeros((9000), np.float64)
    stereoframe = np.zeros((height,width*2, 3), np.uint8)
    RighAndLeft_frame =  np.zeros((height,width*2, 3), np.uint8)
    RighAndLeft_frame[:,0:width,:] =  RighAndLeft_frame[:,0:width,:] + 100  # Right fram is the bright one 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    File_name_PreFix = str('%s' %datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S'))
    VideoRL = cv2.VideoWriter(File_name_PreFix, fourcc, 30.0, (height, width*2))
    while True:
        from_queueR = pipe_childR.recv()
        Time_Right[Frame_Index] = time.time()
        #cv2.imshow('Right', from_queueR)
        
        from_queueL = pipe_childL.recv()
        Time_Left[Frame_Index] = time.time()        
        #cv2.imshow('Left', from_queueL)
        stereoframe[:,0:width,:] = from_queueR 
        stereoframe[:,width:,:]  = from_queueL     
        cv2.imshow('Frame', stereoframe)
        KeyBoard =  cv2.waitKey(10) & 0xFF 
        if   KeyBoard == ord('r'):    
            RecordCommand = 0
        elif KeyBoard == ord('s'):     
            RecordCommand = 2
        elif KeyBoard == ord('q'):
            RecordCommand = 3
            
            
        if RecordCommand == 1:
            VideoRL.write(stereoframe)
            Frame_Index = Frame_Index + 1
        elif RecordCommand == 2:
            VideoRL.release()
        elif RecordCommand == 0:
            File_name_PreFix = str('%s' %datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S'))
            VideoRL = cv2.VideoWriter((File_name_PreFix + '.mp4'), fourcc, 30.0, (width*2, height*1))
            RecordCommand = 1
            stereoframe = RighAndLeft_frame
            VideoRL.write(stereoframe)      
            Time_Right = np.zeros((9000), np.float64)
            Time_Left = np.zeros((9000), np.float64)
            Frame_Index = 1            
            
        elif RecordCommand == 3:
            VideoRL.release()
            File = h5py.File(File_name, "w")
            SubGroup1 = File.create_group("Time")
            File.create_dataset('Time/Right', data = Time_Right) 
            File.create_dataset('Time/Left', data = Time_Left)             
            File.close()
            break
    
    
    
  
            
 
def show_loopR(pipe_childR):
    cv2.namedWindow('pepe')
 
    while True:
        from_queue = pipe_childR.recv()
        cv2.imshow('pepe', from_queue)
        cv2.waitKey(1)  
    

            
def show_loopL(pipe_childL):
    plt.figure()
    #cv2.namedWindow('pepeLasda')
 
    while True:
        from_queueL = pipe_childL.recv()
        cv2.imshow('pepeLasda', from_queueL)
        #cv2.waitKey(1)  
        plt.imshow(from_queueL)
        plt.show()
        




def CameraReaderR(Video_Name_R):
    capR = cv2.VideoCapture(1)
    print(1)
#    print(fourcc)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    print(2)
    VideoR = cv2.VideoWriter('Video_Name_R.mp4', fourcc, 30.0, (width, height))
    print(3)
    while (CamR_Is_Read.value != 2):
        retR, frameR = capR.read()
        CamR_Is_Read.value = 1
        print(frameR)
        cv2.imshow('Right frame', frameR)
        cv2.waitKey(1) 
        print(5)        
        if RecordCommand == 1:
            VideoR.write(frameR)
         
    VideoR.release()
    capR.release()
        

def CameraReaderL(Video_Name_L):
    capL = cv2.VideoCapture(0)
    VideoL = cv2.VideoWriter(Video_Name_L, fourcc, 30.0, (width, height))
    while CamL_Is_Read.value != 2:
        retL, frameL = capR.read()
        CamL_Is_Read.value = 1
        if RecordCommand == 1:
            VideoL.write(frameL)
    
    VideoL.release()
    capL.release()
        


if __name__ == "__main__":  
    globalVariables()
    CamR_Is_Read = Value('i', 0)
    CamR_Is_Read.value = 0
    CamL_Is_Read = Value('i', 0)
    CamL_Is_Read.value = 0
    RecordCommand = Value('i', 0)
    RecordCommand.value = 5
    BlankFrame = np.zeros((height, width,3), np.uint8)
    while(RecordCommand != 3):

#        if (CamR_Is_Read == 1) & (CamL_Is_Read == 1):
#            stereoframe[:,0:width,:] = frameR 
#            stereoframe[:,width:,:]  = frameL 
#            cv2.imshow('frame Right', stereoframe)
#            CamR_Is_Read = 0
#            CamL_Is_Read = 0
            
        cv2.imshow('blank frame', BlankFrame)
        KeyBoard =  cv2.waitKey(10) & 0xFF
        if   KeyBoard == ord('h'):    
            print('rrrrrrrrrrrrrrrrrrrrrrrrrrrr')
#            File_name_PreFix = str('%s' %datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S'))
#            Video_Name_R = File_name_PreFix + '-' + 'R' + '.mp4'
#            Video_Name_L = File_name_PreFix + '-' + 'L' + '.mp4'
#            Pros_CameraReaderR = Process(target=CameraReaderR, args=(Video_Name_R, ))
#            Pros_CameraReaderR.start() 
#            #Pros_CameraReaderL = Process(target=CameraReaderR, args=(len(Video_Name_L), ))
#            #Pros_CameraReaderL.start() 
            
        elif  KeyBoard == ord('r'):
            logger = multiprocessing.log_to_stderr()
            logger.setLevel(multiprocessing.SUBDEBUG)
         
            pipe_parentR, pipe_childR = multiprocessing.Pipe()
            pipe_parentL, pipe_childL = multiprocessing.Pipe()
         
            cam_processR = multiprocessing.Process(target=cam_loopR,args=(pipe_parentR, ))
            cam_processR.start()
            cam_processL = multiprocessing.Process(target=cam_loopL,args=(pipe_parentL, ))
            cam_processL.start()

            
         
#            show_processR = multiprocessing.Process(target=show_loopR,args=(pipe_childR, ))
#            show_processR.start()
#            show_processL = multiprocessing.Process(target=show_loopL,args=(pipe_childL, ))
#            show_processL.start()
            show_processRL = multiprocessing.Process(target=show_loop_stereo,args=(pipe_childR, pipe_childL, ))
            show_processRL.start()
            
         
            cam_processR.join()
            #show_loopR.join()
            cam_processL.join()
            #show_loopL.join()
            show_loop_stereo.join()
                        
        elif KeyBoard == ord('s'):
            CamR_Is_Read.value = 2 
            CamL_Is_Read.value = 2
            time.sleep(0.100)
            Pros_CameraReaderR.terminate()
            Pros_CameraReaderL.terminate()            
                        
                        
        elif KeyBoard == ord('q'):
            CamR_Is_Read.value = 2 
            CamL_Is_Read.value = 2
            time.sleep(0.100)
            Pros_CameraReaderR.terminate()
            Pros_CameraReaderL.terminate()    
            break
    
  

        # Display the resulting frame
    #    cv2.imshow('frame Right',frameR)
    #    cv2.imshow('frame Left' ,frameL)
        

    # When everything done, release the capture

    cv2.destroyAllWindows()





















# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 21:32:52 2018

@author: yaqub
"""


'''
import h5py
fname = 'mytestfile.hdf5'

file = h5py.File(fname,'r')
dataset = file['mydataset']
subgroup1 = dataset[0:]
print subgroup1
print len(subgroup1)

dataset2 = file['subgroup2/dataset_three']
print dataset2[0:]
print len(dataset2)
'''



'''
import h5py
import matplotlib.pyplot as plt
import numpy as np
import time

File_name = "Opterode_RecordingAt" + str('%i' %time.time())+ ".hdf5"
f = h5py.File(File_name, "w")
Spec_sub1 = f.create_group("Spectrumeter")
Spec_specification = Spec_sub1.create_dataset("Spectrumeter", (10,), dtype='f')
Spec_specification.attrs['Serial Number'] = np.string_('12345')
Spec_specification.attrs['Model'] = np.string_('H1+123')
wavelength = np.random.rand(2048,1)
Spec_wavelength =  f.create_dataset('Spectrumeter/Wavelength', data = wavelength)
#Spec_wavelength =  f.create_dataset('Spectrumeter/Wavelength',  (len(wavelength),), dtype='f')
#Spec_wavelength = wavelength
intensities = np.random.rand(2048,2)
Spec_intensities = f.create_dataset('Spectrumeter/Intensities', data = intensities)
#Spec_intensities = f.create_dataset('Spectrumeter/Intensities', (len(wavelength),2), dtype='f')
#Spec_intensities = intensities
f.close()
'''



import h5py
import matplotlib.pyplot as plt
import numpy as np
import os.path


Path_to_Records = os.path.abspath(os.path.join( os.getcwd(), os.pardir)) + "/Records"
os.chdir(Path_to_Records)
    
f = h5py.File('PaeruginosaStain442-2016-10-05-13-11-24.hdf5','r')


Path_to_Fred_Codes = os.path.abspath(os.path.join( os.getcwd(), os.pardir)) + "/Fred"
os.chdir(Path_to_Fred_Codes)
    
ks = f.keys()
#python3  for ks in f.keys():print(f[ks]) or list(f)
# list(f['DAQT7/DAC_Readings'])
len(f[ks[0]].values())
(f[ks[1]].values()[0]).shape
(f[ks[1]].values()[1]).shape
(f[ks[1]].values()[2]).shape

Details_Spectrometer_Name = f[ks[0]].attrs.keys()[0]
Details_Spectrometer_Content = f[ks[0]].attrs.values()[0]
print (f[ks[0]].attrs.keys()[0])
print (f[ks[0]].attrs.values()[0])
#or use this
print (Details_Spectrometer_Name.title())
print (Details_Spectrometer_Content.title())



Intensities = f[ks[1]].values()[0]
Spectrumeter = np.array(f[ks[1]].values()[1])
Wavelength = np.array(f[ks[1]].values()[2])

DAQ_Reading = f[ks[0]].values()[0]
DAQ_TimeInd = f[ks[0]].values()[1]
DAQ_TimeInd = DAQ_TimeInd - DAQ_TimeInd[0]
# example how to plot the signals

plt.figure()
plt.plot(DAQ_TimeInd, DAQ_Reading)


'''
for index,key in enumerate(ks[:]):
    print index, key
    data = np.array(f[key].values())
    plt.plot(data.ravel())
'''
plt.figure()
for I in range(Intensities.shape[1]):
    plt.plot(Wavelength[1:],Intensities[1:,I][:])
    plt.pause(0.1)
    #plt.clf()




'''
#https://www.linuxquestions.org/questions/programming-9/importing-h5-dataset-into-python-4175607632/
$ ipython3

In [1]: import h5py

In [2]: fp = h5py.File('SVM01.h5', 'r')

In [3]: list(fp.keys())
Out[3]: ['All_Data', 'Data_Products']

In [4]: list(fp['All_Data'].keys())
Out[4]: ['VIIRS-M1-SDR_All']

In [5]: list(fp['All_Data/VIIRS-M1-SDR_All'].keys())
Out[5]:
['ModeGran',
 'ModeScan',
 'NumberOfBadChecksums',
 'NumberOfDiscardedPkts',
 'NumberOfMissingPkts',
 'NumberOfScans',
 'PadByte1',
 'QF1_VIIRSMBANDSDR',
 'QF2_SCAN_SDR',
 'QF3_SCAN_RDR',
 'QF4_SCAN_SDR',
 'QF5_GRAN_BADDETECTOR',
 'Radiance',
 'RadianceFactors',
 'Reflectance',
 'ReflectanceFactors']

In [7]: data = fp['All_Data/VIIRS-M1-SDR_All/Radiance']

In [8]: print(len(data))
768

In [9]: print(data[42])
[20905 20716 20038 ..., 23681 23699 23699]

In [10]: type(data[42])
Out[10]: numpy.ndarray

In [11]: len(data[42])
Out[11]: 3200






#%% Relabel the labesl
import os
import cv2
import glob
import numpy as np
num_class = 6
#data_path ="/run/user/1001/gvfs/smb-share:server=hpc-fs.qut.edu.au,share=jonmoham/Python/DataForTraining/Cadava_2018-04-27/train/label"
data_path = '/home/yaqub/Documents/Robotics/Images/Contouring/Total'
os.chdir(data_path)
os.mkdir("label_12345")
os.mkdir("Femur")
os.mkdir("Tibia")
os.mkdir("ACL")
os.mkdir("MedialMeniscus")
os.mkdir("LateralMeniscus")
os.mkdir("Object")
train_label = sorted(glob.glob("label"+"/*.png"))
for i in range(len(train_label)):
    labels = cv2.imread(train_label[i], cv2.CV_8UC1)
    #x = np.zeros([384,768, num_class])
    Label_Tot=np.zeros((384,384,1))

    Label_Femur          = np.zeros((384,384,1))
    Label_Tibia          = np.zeros((384,384,1))
    Label_ACL            = np.zeros((384,384,1))
    #Label_Meniscus       = np.zeros((384,384,1))
    Label_MedialMeniscus = np.zeros((384,384,1))
    Label_LateralMiniscus= np.zeros((384,384,1))
    Label_Object         = np.zeros((384,384,1))

    for ii in range(384):
        for jj in range(384):
            if  labels[ii][jj] == 0:         #Nothing
                Label_Tot[ii,jj] = 0
            elif  labels[ii][jj] == 80:      #Femur
                Label_Tot[ii,jj] = 1
                Label_Femur[ii,jj] = 255
            elif labels[ii][jj] ==100:       #Tibia
                Label_Tot[ii,jj] = 2
                Label_Tibia[ii,jj] = 255
#            elif labels[i][jj] ==120:       #Patela
#                Label_Tot[ii,jj] = 2
#                Label_Patela[ii,jj] = 255
            elif labels[ii][jj] ==140:       #ACL
                Label_Tot[ii,jj] = 3
                Label_ACL[ii,jj] = 255
#            elif labels[i][jj] ==160:       #PCL
#                labels[i][jj] = 5
#                Label_PCL[ii,jj] = 255
#            elif labels[i][jj] ==180:       #PatelaTendon
#                labels[i][jj] = 6
#                Label_PatelaTendon[ii,jj] = 255
#            elif labels[i][jj] ==200:       #quadriceps_tendon
#                labels[i][jj] = 7
#                Label_QuadricepsTendon[ii,jj] = 255
            elif labels[ii][jj] ==220:       #Medial Meniscus
                Label_Tot[ii,jj] = 4
                Label_MedialMeniscus[ii,jj] = 255
            elif labels[ii][jj] ==240:       #Lateral Miniscus
                Label_Tot[ii,jj] = 5
                Label_LateralMiniscus[ii,jj] = 255
            elif labels[ii][jj] ==50:       #Object
                Label_Tot[ii,jj] = 6
                Label_Object[ii,jj] = 255
            else:
                print(train_label[i])
                print ("Error: Unkone label")
                print("ii:" + str(ii))
                print("jj" + str(jj))
                print('label:'+str(labels[ii][jj]))
                break
                break
                break
    cv2.imwrite("label_12345/"   +train_label[i][len("label/"):-9] + '.png', Label_Tot)
    cv2.imwrite("Femur/"         +train_label[i][len("label/"):-9] + '.png', Label_Femur)
    cv2.imwrite("Tibia/"         +train_label[i][len("label/"):-9] + '.png', Label_Tibia)
    cv2.imwrite("ACL/"           +train_label[i][len("label/"):-9] + '.png', Label_ACL)
    cv2.imwrite("MedialMeniscus/"+train_label[i][len("label/"):-9] + '.png', Label_MedialMeniscus)
    cv2.imwrite("LateralMeniscus/"+train_label[i][len("label/"):-9]+ '.png', Label_LateralMiniscus)
    cv2.imwrite("Object/"        +train_label[i][len("label/"):-9] + '.png', Label_Object)



#%%
import os
import cv2
import glob
import numpy as np
data_path ="~/Downloads/Cadava_27-4-2018/"
os.chdir(data_path)
Labels_L = sorted(glob.glob("*d.png"))
Labels_R = sorted(glob.glob("*R.png"))
for i in range(len(Labels_R)):
    Img_L = cv2.imread(Labels_R[i])
    Img_R = cv2.imread(Labels_R[i][:-6]+'.png')
    Img_R[:,384:384*2,:] = Img_L[:,384:384*2,:]
    cv2.imwrite(Labels_R[i][:-6]+'.png', Img_R)
    
    
    


#%% changes the intensity
import os
import cv2
import glob
import numpy as np
images = sorted(glob.glob("*.png"))
for i in range(len(images)):
    Img = cv2.imread(images[i])
    cv2.imwrite(images[i], Img*255)
    
    
#%% change the type of the image to the gray scale
import os
import cv2
import glob
import numpy as np
images = sorted(glob.glob("*mask.tif"))
for i in range(len(images)):
    Img = cv2.imread(images[i])
    gray_image = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(images[i], gray_image)


    
#%% changes the labesl
import os
import cv2
import glob
import numpy as np
images = sorted(glob.glob("*.png"))
for i in range(len(images)):
    Img = cv2.imread(images[i])
    cv2.imwrite(str(i)+'.png', Img)
    
    
#%% split the images
import os
import cv2
import glob
import numpy as np
images = sorted(glob.glob("*Cont.png"))
for i in range(len(images)):
    Img = cv2.imread(images[i])
    ImgR= Img[0:Img.shape[0] , Img.shape[1]/2: , :]
    ImgL= Img[0:Img.shape[0] , 0:Img.shape[1]/2, :]
    cv2.imwrite(images[i][:-4] + '_L.png', ImgL)
    
    ImgR[:,0:4,:] = 0
    
    #cv2.imwrite(images[i][:-4] + '.png', ImgR)
    

#%% downsample the images resolution
import os               
import cv2                        
import glob
import numpy as np
images = sorted(glob.glob("*.png"))
Img = cv2.imread(images[0])
r = 256.0 / Img.shape[1]
dim = (256, int(Img.shape[0] * r))
for i in range(len(images)):
    Img = cv2.imread(images[i])
    resized = cv2.resize(Img, dim, interpolation = cv2.INTER_AREA)
    cv2.imwrite(images[i], resized)
    
    
#%% blackens the L images on their left.
import os
import cv2
import glob
import numpy as np
images = sorted(glob.glob("*_L.png"))
for i in range(len(images)):
    Img = cv2.imread(images[i])
    Img[:,0:4,:] = 0
    cv2.imwrite(images[i], Img)    

#%% read 4 channel image
import cv2
im = cv2.imread("image.png", cv2.IMREAD_UNCHANGED)



import os
import cv2
import glob
import numpy as np
images = sorted(glob.glob("*_R.png"))
for i in range(len(images)):
    ImgR = cv2.imread(images[i])    
    ImgR[:,380:384,:] = 0
    cv2.imwrite(images[i], ImgR)


#%%
imgL = cv2.imread('Down_Left.png')
imgR = cv2.imread('Down_Right.png')
#imgL = imgL.astype('float32')
#imgR = imgR.astype('float32')
#imgL /= 255
#imgR /= 255
imgs_test2= imgs_test[0:2,:,:,:]
imgs_test2[1,:,:,0] = imgR[:,:,0]
imgs_test2[0,:,:,0] = imgL[:,:,0]


#%% Dice coeff
k=1
import os
import cv2
import glob
import numpy as np
images_GT = sorted(glob.glob("*_GT.png"))
images_Predic = sorted(glob.glob("*ic.png"))
Dice = np.zeros((100,1),np.float32)
for i in range(len(images_GT)):
    ImgGT = cv2.imread(images_GT[i]) 
    retval, ImgGT = cv2.threshold(ImgGT[:,:,0], 50, 255, cv2.THRESH_BINARY)
    ImgPr = cv2.imread(images_Predic[i]) 
    retval, ImgPr = cv2.threshold(ImgPr[:,:,0], 50, 255, cv2.THRESH_BINARY)
    
    ImgGT = ImgGT[:,:].astype('float32')
    ImgPr = ImgPr[:,:].astype('float32')
    ImgGT /= 255
    ImgPr /= 255
    
    seg = np.zeros((256,256), dtype='int')
    gt =  np.zeros((256,256), dtype='int')
    gt[:,:] = ImgGT[:,:]
    seg[:,:]= ImgPr[:,:]
    #dice = np.sum(ImgPr[images_GT==k])*2.0 / (np.sum(ImgPr) + np.sum(images_GT))
    dice = np.sum(seg[gt==k]==k)*2.0 / (np.sum(seg[seg==k]==k) + np.sum(gt[gt==k]==k))
    Dice[i] = dice
    print ('Dice similarity score is {}'.format(dice))
    
    

#%% Cropping the image
import os
import cv2
import glob
import numpy as np
images = sorted(glob.glob("*.png"))
for i in range(len(images)):
    Img = cv2.imread(images[i],-1)    
    Img2 = Img[30:285, 485:740]
    cv2.imwrite(images[i], Img2)





#%% Crop the mono images and then downsample them       
import cv2                        
import glob
import numpy as np
images = sorted(glob.glob("*.png"))
Img = cv2.imread(images[0])
for i in range(len(images)):
    Img = cv2.imread(images[i])
    ImgR = Img[5:675, 285:955,:]
    r = 384.0 / ImgR.shape[1]
    dim = (384, int(ImgR.shape[0] * r))
    resized = cv2.resize(ImgR, dim, interpolation = cv2.INTER_AREA)
    cv2.imwrite(images[i], resized)
    
    
import os               
import cv2                        
import glob
import numpy as np
images = sorted(glob.glob("*.png"))
Img = np.zeros((349,349,3))  
r = 256.0 / Img.shape[1]
dim = (256, int(Img.shape[0] * r))
for i in range(len(images)):
    Img = cv2.imread(images[i])
    #Img2 = Img[18:367, 18:367]              #for air
    Img2 = Img[18:367, 18:367]              #for water left [15:364, 24:373] and for the right [11:360, 5:354]
    resized = cv2.resize(Img2, dim, interpolation = cv2.INTER_AREA)
    cv2.imwrite(images[i], resized)


#%% break the stereo, crop and resize 
import os               
import cv2                        
import glob
import numpy as np
images = sorted(glob.glob("*.png"))
Img = np.zeros((349,349,3))  
r = 256.0 / Img.shape[1]
dim = (256, int(Img.shape[0] * r))
for i in range(len(images)):
    Img = cv2.imread(images[i])
    #Img2 = Img[18:367, 18:367]              #for air
    #for water left [24:373, 15:364]and for the right [5:354, 384+11:384+360]
    Img_L = Img[24:367, 21:364]
    Img_R = Img[24:367, 384+21:384+364]
    resized = cv2.resize(Img_L, dim, interpolation = cv2.INTER_AREA)
    cv2.imwrite('../left/'  + images[i], resized)
    resized = cv2.resize(Img_R, dim, interpolation = cv2.INTER_AREA)
    cv2.imwrite('../right/' + images[i], resized)
    
#%% reading and writing the text file and shuffling the lines
import numpy as np
Read_File = open("Readme.txt", "r")
Write_File = open('Names_Write.txt','w') 
#lines = text_file.readlines()
lines = Read_File.read().split('\n')
#print lines
print (len(lines))
Indexes = range(len(lines))
np.random.shuffle(Indexes)

for Index in Indexes:
    Write_File.write(lines[Index])
    Write_File.write('\n')
    
Read_File.close()
Write_File.close()

#%% reading and writing the text file and shuffling the lines for every 500
import numpy as np
Read_File = open("Camera_SLAM_Quatr.txt", "r")
Write_File = open('Camera_SLAM_Quatr_Rand.txt','w') 
lines = Read_File.read().split('\n')
#print lines
print (len(lines))
Sequence_length=250
Indexes=np.zeros((1,len(lines)-0), np.uint16)
for Index in range(0,len(lines)):
    Indexes[0,Index]=lines.index(lines[Index])  
    
Indexes2=Indexes[0,0:-1:Sequence_length]
np.random.shuffle(Indexes2)
#Write_File.write(lines[0])
#Write_File.write('\n')
#Write_File.write(lines[1])
#Write_File.write('\n')
#Write_File.write('\n')
counter=0
for Index in Indexes2:
    for i in range(Sequence_length):
        Write_File.write(lines[Index+i])
        Write_File.write('\n')
        
Read_File.close()
Write_File.close()


#%%
import os               
import cv2                        
import glob
import numpy as np
images = sorted(glob.glob("*.png"))
Img = np.zeros((349,349,3))  
r = 256.0 / Img.shape[1]
dim = (256, int(Img.shape[0] * r))
for i in range(len(images)):
    Img = cv2.imread(images[i])
    #Img2 = Img[18:367, 18:367]              #for air
    #for water left [24:373, 15:364]and for the right [5:354, 384+11:384+360]
    #Img_L = Img[24:373, 15:364]
    #Img_R = Img[20:369, 384+11:384+360]
    Img2 = Img[20:369, 11:360]  
    resized = cv2.resize(Img2, dim, interpolation = cv2.INTER_AREA)
    cv2.imwrite(images[i], resized)
    

#%% Copy into another folder
import numpy as np
import os
import shutil
Read_File = open("Names.txt", "r")
lines = Read_File.read().split('\n')
print len(lines)
Indexes = range(len(lines))
for Index in Indexes:
    shutil.copy2('Pairs/' + lines[Index] , 'Pairs2/' + lines[Index] )
Read_File.close()

#%% 3D scatter plot
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x =[1,2,3,4,5,6,7,8,9,10]
y =[5,6,2,3,13,4,1,2,4,8]
z =[2,3,3,3,5,7,9,11,9,10]
ax.scatter(x, y, z, c='r', marker='o')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()
    
#%%
import os               
import cv2                        
import glob
import numpy as np
images = sorted(glob.glob("*.png"))

for i in range(len(images)):
    images_path = 'Tests/'+images[i]    
    monodepth_simple.py --image_path images_path[i] --checkpoint_path /home/jonmoham/Python/Depth_Mono/temp_Stereo3/Stereo3_New_Rand_alpha_image_loss06/model-120000    --encoder resnet50 --input_height 256 --input_width 256



#%% Convert img numpy
r_mask = np.load('2018-11-20-13-32-56_00539_R_r_mask.npy')

l_mask = np.load('2018-11-20-13-32-56_00539_R_l_mask.npy')


r_disp = np.load('2018-11-20-13-32-56_00539_R_r_disp.npy')
#r_disp2 = r_disp-np.min(r_disp)
r_disp2 = np.round((r_disp*800))
r_disp2 = r_disp2.astype(np.uint8)

l_disp = np.load('2018-11-20-13-32-56_00539_R_l_disp.npy')
#l_disp2 = l_disp-np.min(l_disp)
l_disp2 = np.round((l_disp*800))
l_disp2 = l_disp2.astype(np.uint8)


Img = r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp
#Img2 = l_disp-np.min(Img)
Img2 = np.round((Img*800))
Img2 = Img2.astype(np.uint8)



Img = r_mask * ImgNP2_l + l_mask * ImgNP2_r + (1.0 - l_mask - r_mask) * ImgNP2_m
disp = np.load('2018-11-20-13-32-56_00139_L_disp.npy')
#disp2 = disp-np.min(disp)
disp2 = np.round((disp[0:2,:,:,0]*800))
disp2 = disp2.astype(np.uint8)


#%% 
m_disp = np.load('2018-11-20-13-32-56_02429_R_disp.npy')
#m_disp2 = m_disp-np.min(m_disp)
m_disp2 = np.round((m_disp*800))
m_disp2 = m_disp2.astype(np.uint8)

_, h, w, _ = m_disp.shape
l_disp = m_disp[0,:,:,0]
r_disp = np.fliplr(m_disp[1,:,:,0])
m_disp = 0.5 * (l_disp + r_disp)
l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
r_mask = np.fliplr(l_mask)

#%%
r_disp2 = np.round((r_disp*800))
r_disp2 = r_disp2.astype(np.uint8)

l_disp2 = np.round((l_disp*800))
l_disp2 = l_disp2.astype(np.uint8)
#%%
Img = r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp
#Img2 = Img-np.min(Img)
Img2 = np.round((Img*800))
Img2 = Img2.astype(np.uint8)
cv2.imshow('Middle',(Img2[:,:]))
cv2.imshow('right',(r_disp2[:,:]))
cv2.imshow('left' ,(l_disp2[:,:]))
cv2.waitKey(11100)

#%%
plt.hist(Img2.ravel()) ;
plt.show()



#%% 
images = sorted(glob.glob("*0_Cont.png"))
for i in range(len(images)):
    Img = cv2.imread(images[i])
    ImgR= Img[0:Img.shape[0] , Img.shape[1]/2: , :]
    ImgL= Img[0:Img.shape[0] , 0:Img.shape[1]/2, :]
    cv2.imwrite(images[i][:-8] + 'L_Cont.png', ImgL)

    
images = sorted(glob.glob("*0.png"))
for i in range(len(images)):
    Img = cv2.imread(images[i])
    ImgR= Img[0:Img.shape[0] , Img.shape[1]/2: , :]
    ImgL= Img[0:Img.shape[0] , 0:Img.shape[1]/2, :]
    cv2.imwrite(images[i][:-4] + '_L.png', ImgL)
    
    ImgR[:,0:4,:] = 0
    
    cv2.imwrite(images[i][:-4] + '_R.png', ImgR)
    
    
images = sorted(glob.glob("*Cont_R.png"))
for i in range(len(images)):
    Img = cv2.imread(images[i])
    ImgR= Img[0:Img.shape[0] , Img.shape[1]/2: , :]
    ImgL= Img[0:Img.shape[0] , 0:Img.shape[1]/2, :]
    
    ImgR[:,0:4,:] = 0
    
    cv2.imwrite(images[i][:-10] + 'R_Cont.png', ImgR) 
    
    
images = sorted(glob.glob("*_L.png"))
for i in range(len(images)):
    Img = cv2.imread(images[i])
    Img[:,379:384,:] = 0
    cv2.imwrite(images[i], Img)   

images = sorted(glob.glob("*_R.png"))
for i in range(len(images)):
    Img = cv2.imread(images[i])
    Img[:,0:5,:] = 0
    cv2.imwrite(images[i], Img)
    

images = sorted(glob.glob("*R_Cont.png"))
for i in range(len(images)):
    Img = cv2.imread(images[i])
    Img[:,0:5,:] = 0
    cv2.imwrite(images[i], Img)
    
images = sorted(glob.glob("*L_Cont.png"))
for i in range(len(images)):
    Img = cv2.imread(images[i])
    Img[:,379:384,:] = 0
    cv2.imwrite(images[i], Img)  
    
os.system('rm *0.png')
os.system('rm *Cont_L.png')
os.system('rm *Cont_R.png')
os.system('rm *0_Cont.png')
    
    
# %% Histogram equalization and Clahe (adaptive histogram equalization)
import cv2
import glob
images = sorted(glob.glob("*.png"))
for i in range(len(images)):
    imgBGR = cv2.imread(images[i])

    LAB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2LAB)
    LAB_planes = cv2.split(LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6,6))
    LAB_planes[0] = clahe.apply(LAB_planes[0])
    
    LAB = cv2.merge(LAB_planes)
    
    imgBGR_Clahe = cv2.cvtColor(LAB, cv2.COLOR_LAB2BGR)
    
    
    #equ = cv2.equalizeHist(img)
    #res = np.hstack((img,equ)) #stacking images side-by-side
    #cv2.imshow('Clahe', imgBGR_Clahe)
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    #cl1 = clahe.apply(img)
    #cv2.imshow('Orig', imgBGR)
    
    #cv2.waitKey(1)
    cv2.imwrite("../Clahe/"+images[i], imgBGR_Clahe)    
    
    
    
# %% Image to video
import numpy as np
import glob
import cv2
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
stereoframe = np.zeros((384,384*2,3), np.uint8)
fname = '90Degree'  
VideoRL = cv2.VideoWriter(fname[0:]+'.mp4', fourcc, 30.0, (384*2, 384))
imagesL = sorted(glob.glob("left/rgb/*.png"))
imagesR = sorted(glob.glob("right/rgb/*.png"))
for i in range(len(imagesL)):
    stereoframe[:,0:384,:] = cv2.imread(imagesL[i])
    stereoframe[:,384: ,:] = cv2.imread(imagesR[i])
    VideoRL.write(stereoframe[:,:,:])

    #cv2.imshow('Orig', imgBGR)
    
VideoRL.release()

    
    
    
import numpy as np
import glob
import cv2
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
stereoframe = np.zeros((576,720,3), np.uint8)
stereoframe2 = np.zeros((576,720,3), np.uint8)
fname = 'Video3'  
VideoRL = cv2.VideoWriter(fname[0:]+'.mp4', fourcc, 30.0, (720,576))
imagesL = sorted(glob.glob("rgb/*.png"))
for i in range(0, len(imagesL),4):
    stereoframe[:,:,:] = cv2.imread(imagesL[i])
    stereoframe2[:,:,2] = stereoframe[:,:,0]
    stereoframe2[:,:,1] = stereoframe[:,:,1]
    stereoframe2[:,:,0] = stereoframe[:,:,2]
    VideoRL.write(stereoframe2[:,:,:])
    #cv2.imwrite(imagesL[i],stereoframe2[:,:,:])
    #cv2.imshow('Orig', imgBGR)
VideoRL.release()
    

    
#% Clahe for a loop of folders
import cv2
import glob
import os

#CurrentDir='/run/user/1001/gvfs/smb-share:server=hpc-fs.qut.edu.au,share=jonmoham/DataForTraining/BlenderData'
CurrentDir='/home/jonmoham/DataForTraining/BlenderData'
os.chdir(CurrentDir)


Prefix = 'Movie_'
CurrentFoder = os.path.abspath(os.path.join( os.getcwd()))
for Folder_Index in range(60):
    
    Destin = Prefix  + str(Folder_Index+1) + '/DownSample'
    if not os.path.exists(CurrentDir + '/' + Destin + '/Clahe'):
        os.chdir(Destin)
        os.mkdir(CurrentDir + '/' + Destin + '/Clahe')
        #mkdir ../Clahe
        os.chdir(CurrentDir + '/' + Destin + '/Image')
        print(Destin)
        images = sorted(glob.glob("*.png"))
        for i in range(len(images)):
            imgBGR = cv2.imread(images[i])
        
            LAB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2LAB)
            LAB_planes = cv2.split(LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3,3))
            LAB_planes[0] = clahe.apply(LAB_planes[0])
            
            LAB = cv2.merge(LAB_planes)
            
            imgBGR_Clahe = cv2.cvtColor(LAB, cv2.COLOR_LAB2BGR)
            
            
            #equ = cv2.equalizeHist(img)
            #res = np.hstack((img,equ)) #stacking images side-by-side
            #cv2.imshow('Clahe', imgBGR_Clahe)
            #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            #cl1 = clahe.apply(img)
            #cv2.imshow('Orig', imgBGR)
            
            #cv2.waitKey(1)
            cv2.imwrite("../Clahe/"+images[i], imgBGR_Clahe[:,:,0:3])    
            
        os.chdir(CurrentFoder)
    
#%% printing the addresses of the images
Prefix = 'Movie_'
CurrentFoder = os.path.abspath(os.path.join( os.getcwd()))
with open('Text.txt', 'a') as f:
    for Folder_Index in range(60):
        Destin = Prefix + str(Folder_Index+1) + '/Clahe'
        os.chdir(Destin)
        #os.system('mkdir ../Clahe')
        #mkdir ../Clahe
        images = sorted(glob.glob("*.png"))
        for i in range(len(images)):
            imgBGR = cv2.imread(images[i])
            f.write(CurrentFoder + '/' + Destin+'/'+images[i])
            f.write('\n')
        os.chdir(CurrentFoder)



#%% loop though folders and conver the exr to Png
import cv2
import glob
import numpy as np
import os
CurrentDir='/home/jonmoham/DataForTraining/BlenderData'
os.chdir(CurrentDir)

Prefix = 'Movie_'
CurrentFoder = os.path.abspath(os.path.join( os.getcwd()))
for Folder_Index in range(60):
    Destin = Prefix + str(Folder_Index+1) + '/Depth_GPU'
    os.chdir(Destin)
    depths = sorted(glob.glob("*_R.exr"))
    for i in range(len(depths)):
        Temp_depth = cv2.imread(depths[i],-1)
        Temp_depth[:,:,:][Temp_depth[:,:,:]==np.inf]=0.0
        Temp_depth[:,:,:][Temp_depth[:,:,:]==10000000000.0]=0.0
        #Temp_depth[:,:,:] = Temp_depth[:,:,:]*1
        Temp_depth = Temp_depth[:,:,0]+10
        Temp_depth[:,:] = Temp_depth[:,:]*50
        Temp_depth[:,:][Temp_depth[:,:]==500]=65535
        
        cv2.imwrite(depths[i][5:-3] + 'png', Temp_depth.astype('uint16'))

        Temp_depth = cv2.imread(depths[i][:-6]+'_L.exr',-1)
        Temp_depth[:,:,:][Temp_depth[:,:,:]==np.inf]=0.0
        Temp_depth[:,:,:][Temp_depth[:,:,:]==10000000000.0]=0.0
        #Temp_depth[:,:,:] = Temp_depth[:,:,:]*1
        Temp_depth = Temp_depth[:,:,0]+10
        Temp_depth[:,:] = Temp_depth[:,:]*50
        Temp_depth[:,:][Temp_depth[:,:]==500]=65535
        cv2.imwrite(depths[i][5:-6] + '_L.png', Temp_depth.astype('uint16'))
        print(Folder_Index)
    os.chdir(CurrentFoder)

#%% loop though folders and downsample the images
import cv2
import glob
import numpy as np
import os
CurrentDir='/home/jonmoham/DataForTraining/BlenderData'
os.chdir(CurrentDir)
r = 256.0 / 384
dim = (256, int(384 * r))
Prefix = 'Movie_'
CurrentFoder = os.path.abspath(os.path.join( os.getcwd()))
for Folder_Index in range(60):
    #os.mkdir(Prefix + str(Folder_Index+1) + '/DownSample')
    #os.mkdir(Prefix + str(Folder_Index+1) + '/DownSample/Depth')
    Destin = Prefix + str(Folder_Index+1) + '/Depth'
    os.chdir(Destin)
    images= sorted(glob.glob("*.png"))
    for i in range(len(images)):
        Img = cv2.imread(images[i],-1)
        resized = cv2.resize(Img, dim, interpolation = cv2.INTER_AREA)
        cv2.imwrite('../DownSample/Depth/' + images[i], resized[:,:])
    print(Folder_Index)   
    os.chdir(CurrentFoder)


import cv2
import glob
import numpy as np
import os
Prefix = 'Movie_'
CurrentFoder = os.path.abspath(os.path.join( os.getcwd()))
for Folder_Index in range(60):
    os.mkdir(Prefix + str(Folder_Index+1) + '/Clahe3')
    Destin = Prefix + str(Folder_Index+1) + '/Clahe'
    os.chdir(Destin)
    images= sorted(glob.glob("*.png"))
    for i in range(len(images)):
        Img = cv2.imread(images[i], cv2.IMREAD_GRAYSCALE)
        cv2.imwrite('../Clahe2/'+images[i], Img[:,:])
       
    os.chdir(CurrentFoder)

#%% read CSV 
    python3
    import csv
    import os
    with open('List_Temp2.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        Disp_Index = 0
        for row in spamreader:
            Disp_Index = Disp_Index + 1
            print(' '.join(row))
            Line = row[0].split('/home/')
            
            SRC = 'disparity_' + str(Disp_Index)+'.png' 
            DEST = Line[1].split('/')[-1][:-6]+'D.png'
            os.rename(SRC,DEST)

import csv
import os  
with open('List_Temp2.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    Disp_Index = 0
    for row in spamreader:
        Disp_Index = Disp_Index + 1
        print(' '.join(row))
        Line = row[0].split('/home/')
        
        #SRC = str(Disp_Index)+'.png' 
        DEST = Line[1].split('/')[-1][:-6]+'.png'
        os.rename('cp DEST ../right')

#%% read text odometery file and parse it 
import os  
with open('0000000000.txt', newline='') as Odo:
    for cnt, line in enumerate(Odo):
        Coordinates=line.split(' ')
        print('Lat:',Coordinates[0])
        print('Lon:',Coordinates[1])
        print('Alt:',Coordinates[2])
        print('Rol:',Coordinates[3])
        print('Pit:',Coordinates[4])
        print('Yaw:',Coordinates[5])

#%% read log file for error
import os   
with open('loss_log.txt', newline='') as Error:
    next(Error) 
    for cnt, line in enumerate(Error): 
        Coordinates=line.split('pos_err:') 
        print(np.float(Coordinates[1][0:6])) 


#%%
import cv2
import glob
import numpy as np
import os
CurrentDir='/home/jonmoham/DataForTraining/BlenderData'
os.chdir(CurrentDir)
r = 256.0 / 384
dim = (256, int(384 * r))
Prefix = 'Movie_'
CurrentFoder = os.path.abspath(os.path.join( os.getcwd()))
for Folder_Index in range(1):
    #os.mkdir(Prefix + str(Folder_Index+1) + '/DownSample')
    os.mkdir(Prefix + str(Folder_Index+30) + '/DownSample/Image')
    Destin = Prefix + str(Folder_Index+30) + '/Image'
    os.chdir(Destin)
    images= sorted(glob.glob("*.png"))
    for i in range(len(images)):
        Img = cv2.imread(images[i],-1)
        resized = cv2.resize(Img, dim, interpolation = cv2.INTER_AREA)
        cv2.imwrite('../DownSample/Image/' + images[i], resized[:,:,:])
    print(Folder_Index)   
    os.chdir(CurrentFoder)


#%% parsing the .txt file
file = open("Text3.txt","w+")
counter = 0
for line in open('Text2.txt'):
    fields = line.split('/')
    
    if len(fields)==4:
        if fields[3][-6] =='R':
            print(fields[0]+'/'+fields[1]+ '/'+fields[2]+'/'+fields[3])
            #file.write(fields[0]+'/'+fields[1]+ '/'+fields[2]+'/'+fields[3])
            
            file.write(fields[0]+'/'+fields[1]+ '/'+'Depth/'+fields[-1][:4] + fields[-1][-7:-1]+'\n')
            counter = counter +1 
            print(counter)

file.close()





import cv2
import glob
import os

#CurrentDir='/run/user/1001/gvfs/smb-share:server=hpc-fs.qut.edu.au,share=jonmoham/DataForTraining/BlenderData'
CurrentDir='/home/jonmoham/DataForTraining/BlenderData'
os.chdir(CurrentDir)


Prefix = 'Movie_'
CurrentFoder = os.path.abspath(os.path.join( os.getcwd()))
for Folder_Index in range(1):
    
    Destin = Prefix  + str(Folder_Index+1) + '/DownSample'

    os.chdir(CurrentDir + '/' + Destin + '/Image')
    print(Destin)
    images = sorted(glob.glob("*.png"))
    for i in range(len(images)):
        imgBGR = cv2.imread(images[i])
        cv2.imwrite(images[i], imgBGR[:,:,0:3])        
    os.chdir(CurrentFoder)       
        
        
with open(List_test_Cada_Simul_Blend.csv, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    Disp_Index = 0
    os.chdir(args.output+'/disparities')
    for row in spamreader:
        print(' '.join(row))
        Line = row[0].split('/home/')
        SRC = 'disparity_' + str(Disp_Index)+'.png' 
        DEST = Line[1].split('/')[-1][:-4]+'D.png'
        os.rename(SRC,DEST)
        Disp_Index = Disp_Index + 1
                
        
        
        
        
        
import glob
import open3d as o3d
import numpy as np
import cv2
import os
import csv
#os.chdir('/home/jonmoham/Python/Real_timeStereo/Output_Stereo6')
Prefix='/home/jonmoham/DataForTraining/BlenderData/Movie_'
for Folder_Index in range(60):
     Destin= Prefix + str(Folder_Index+1) + '/Depth'
     os.chdir(Destin)
     depths = sorted(glob.glob("*.png"))
     Destin= Prefix + str(Folder_Index+1) + '/Image'
     images = sorted(glob.glob('*_Left_Col_L.png'))
     for I in range(len(depths)):
         color_raw = o3d.io.read_image(images[I])
         depth_raw = o3d.io.read_image(depths[I])
         rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)
         pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
         pcd.transform([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0,-1]])



     
     
     
     
#if not os.path.exists(Pointclouds):
#    os.mkdir('Pointclouds')

images = sorted(glob.glob('Image/*_Left_Col_L.png'))
#os.chdir('disparities/')
#with open('../List_test_Cada_Simul_Blend.csv', newline='') as csvfile:
#    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
#    Disp_Index = 0
for row in spamreader:
    #print(' '.join(row))
    Line = row[0].split('/home/')
    #SRC = 'disparity_' + str(Disp_Index)+'.png'
    print('/home/'+Line[1][:-1])
    print(depths[Disp_Index])
    color_raw = o3d.io.read_image('/home/'+Line[1][:-1])
    depth_raw = o3d.io.read_image(depths[Disp_Index])
    Temp = np.asarray(depth_raw)
    Temp = Temp.astype('float32')
    Temp = Temp/100 + 250
    np.asarray(depth_raw)[:,:]=Temp.astype('uint16')
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    pcd.transform([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0,-1]])
    #o3d.io.write_point_cloud(images[i][:-6]+'_P.pcd', pcd)
    if (Line[1].split('/')[3][:-2]=='Movie'):
        DEST= Line[1].split('/')[3]+'_'+Line[1].split('/')[-1][:-4]+'ply'
    else:
        DEST = Line[1].split('/')[-1][:-4]+'ply'
    o3d.io.write_point_cloud('Pointclouds/'+DEST, pcd)
    Disp_Index = Disp_Index + 1
        
        
        
        
        
        
import cv2
import glob
import numpy as np
import os
CurrentDir='/home/jonmoham/DataForTraining/BlenderData'
os.chdir(CurrentDir)

Prefix = 'Movie_Long_'
CurrentFoder = os.path.abspath(os.path.join( os.getcwd()))
for Folder_Index in range(60):
    Destin = Prefix + str(Folder_Index+1) + '/Depth_GPU'
    os.chdir(Destin)
    depths = sorted(glob.glob("*_R.exr"))
    for i in range(len(depths)):
        Temp_depth = cv2.imread(depths[i],-1)
        Temp_depth[:,:,:][Temp_depth[:,:,:]==np.inf]=0.0
        #Temp_depth[:,:,:][Temp_depth[:,:,:]==10000000000.0]=0.0
        #Temp_depth[:,:,:] = Temp_depth[:,:,:]*1
        #Temp_depth = Temp_depth[:,:,0]-4
        Temp_depth[:,:] = Temp_depth[:,:]*2
        #Temp_depth[:,:][Temp_depth[:,:]<0]=65535
        
        cv2.imwrite(depths[i][5:-3] + 'png', Temp_depth.astype('uint16'))

        Temp_depth = cv2.imread(depths[i][:-6]+'_L.exr',-1)
        Temp_depth[:,:,:][Temp_depth[:,:,:]==np.inf]=0.0
        #Temp_depth[:,:,:][Temp_depth[:,:,:]==10000000000.0]=0.0
        #Temp_depth[:,:,:] = Temp_depth[:,:,:]*1
        Temp_depth = Temp_depth[:,:,0]-4
        Temp_depth[:,:] = Temp_depth[:,:]*500
        Temp_depth[:,:][Temp_depth[:,:]<0]=65535
        cv2.imwrite(depths[i][5:-6] + '_L.png', Temp_depth.astype('uint16'))
        print(Folder_Index)
    os.chdir(CurrentFoder)
        
#%% convert the depth to disparity from the Blender data    
import cv2
import glob
import numpy as np
import os
CurrentDir='/home/jonmoham/Documents/DataForTraining/BlenderData/'
os.chdir(CurrentDir)
Prefix = 'Movie_Right'
CurrentFoder = os.path.abspath(os.path.join( os.getcwd()))
for Folder_Index in range(1):
    #Destin = CurrentDir +  Prefix + str(Folder_Index+1) + '/Depth_GPU'
    Destin = CurrentDir +  Prefix + '/Depth_GPU78'
    #os.mkdir(CurrentDir + Prefix + str(Folder_Index+1) + '/Disparity')
    os.chdir(Destin)
    depths = sorted(glob.glob("*_R.exr"))
    for i in range(len(depths)):
        Temp_depth = cv2.imread(depths[i],-1)
        #Temp_depth[:,:,:][Temp_depth[:,:,:]==np.inf]=0.0
        Disp = 20000.0/Temp_depth[:,:,0] + 1000
        cv2.imwrite(CurrentDir + Prefix + str(Folder_Index+1) + '/Disparity/' + depths[i][0:-3] + 'png', Disp.astype('uint16'))
        print(i)
    os.chdir(CurrentDir)    
    

import cv2 
import glob 
import numpy as np 
import os 
depths = sorted(glob.glob("*.png")) 
Max=0 
Min=40 
for I in range(len(depths)): 
    depth=cv2.imread(depths[I], -1) 
    print( depths[I])
    #depth[:,:,:][depth[:,:,:]==np.inf]=255
    if np.max(depth)==255: 
        print('Infinity!!!', depths[I])
        #cv2.imwrite('../depths[I][:-3]'+'png',-1)

    if np.max(depth) > Max: 
        Max=np.max(depth) 
        print('Max', Max) 
    if np.min(depth) < Min: 
        Min=np.min(depth) 
        print('Min', Min) 
    #depth[:,:,:][depth[:,:,:]==np.inf]=0.0 
    if np.min(depth)==0: 
        print('Zero', depths[I]) 


        
#%% read and plot KITTI odometry
import os  
import glob
import numpy as np
Odo_Dic={0:'Lat', 1:'Lon', 2:'Alt', 3:'Rol', 4:'Pit', 5:'Yaw'}
OdoFiles=sorted(glob.glob("/run/user/1000/gvfs/smb-share:server=hpc-fs.qut.edu.au,share=jonmoham/Python/monodepth2-master/kitti_data/2011_09_29/2011_09_29_drive_0071_sync/oxts/data/*.txt"))
Movement=np.zeros((len(OdoFiles), 6), dtype=np.float64)
for OdoFile_Index in range(len(OdoFiles)):
    with open(OdoFiles[OdoFile_Index], newline='') as Odo:
        for cnt, line in enumerate(Odo):
            Coordinates=line.split(' ')
            for Index in range(6):
                #print(Odo_Dic[Index], Coordinates[Index])
                Movement[OdoFile_Index][Index]=Coordinates[Index]
theta = Movement[:,0]
phi = Movement[:,1]
R = Movement[:,2]
roll = Movement[:,3]
pitch= Movement[:,4]
yaw  = Movement[:,5]
X = R * np.sin(phi) * np.cos(theta)
Y = R * np.sin(phi) * np.sin(theta)
Z = R * np.cos(phi)
#plt.scatter(X, Y, Z)
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.scatter3D(X, Y, Z, c=Z, cmap='hsv')
ax.quiver(X, Y, Z, roll, pitch, yaw, length=0.1, normalize=True)


#%% Blender grond truth
import os               
import cv2                        
import glob
import numpy as np
depths = sorted(glob.glob("*.exr"))
Img = cv2.imread(depths[0])
r = 256.0 / Img.shape[1]
dim = (256, int(Img.shape[0] * r))
for i in range(len(depths)):
    Temp_depth = cv2.imread(depths[i],-1)
    Temp_depth[:,:,:][Temp_depth[:,:,:]==np.inf]=255
    Temp_depth = cv2.cvtColor(Temp_depth, cv2.COLOR_BGR2GRAY)
    #Temp_depth[:,:,0] = Temp_depth[:,:,0]*1
    #Temp_depth[:,:,1] = Temp_depth[:,:,0]*1
    #Temp_depth[:,:,2] = Temp_depth[:,:,0]*1
    cv2.imwrite(depths[i][:-3] + 'png', Temp_depth.astype('uint8'))
    resized = cv2.resize(Temp_depth.astype('uint8'), dim, interpolation = cv2.INTER_AREA)
    print(depths[i])
    cv2.imwrite('../DownSampled/Depth_GPU/' + depths[i][:-3] + 'png', resized.astype('uint8'))


import os               
import cv2                        
import glob
import numpy as np
images = sorted(glob.glob("*.png"))
for i in range(len(images)):
    print(images[i])
    Img = cv2.imread(images[i],-1)
    if len(np.shape(Img))==3:
        Img = Img[:,:,0]
        cv2.imwrite(images[i], Img)


#%% RGB separation and masking and erosion

import os               
import cv2                        
import glob
import numpy as np
CurrentFoder = os.path.abspath(os.path.join( os.getcwd()))
os.chdir(CurrentFoder+'/Depth_GPU')
depths = sorted(glob.glob("*_R.exr"))
os.chdir(CurrentFoder+'/Image_Segmented')
images = sorted(glob.glob("*_R.png"))
os.chdir(CurrentFoder)
#for i in range(2):    
#for i in range(120):    
for i in range(len(images)):   
    img_number = images[i][:len(images[i])-12]
    #img   = cv2.imread(images[i],-1)	
    img = cv2.imread('Image_Segmented/'+ images[i],-1)
    print(depths[i][:-5]+'Right_R.png')	
    depth = cv2.imread('Depth_GPU/' + 'Image' + img_number + '_R.exr',-1)
    Temp_depth = depth[:,:,0]
    Temp_depth[:,:][Temp_depth[:,:]==np.inf]=0.0
    depth = Temp_depth.astype('uint8')
    R=np.zeros((256,256), np.uint8)
    G=np.zeros((256,256), np.uint8)
    B=np.zeros((256,256), np.uint8)
    Y=np.zeros((256,256), np.uint8)
    C=np.zeros((256,256), np.uint8)
    M=np.zeros((256,256), np.uint8)
    W=np.zeros((256,256), np.uint8)
    Gray=np.zeros((256,256), np.uint8)
    Masked=np.zeros((256,256,3), np.uint8)
    #ret,thresh1 = cv2.threshold(img,250,255,cv2.THRESH_TOZERO)
    ret,thresh1 = cv2.threshold(img,250,255,cv2.THRESH_BINARY)
    ret,threshNot = cv2.threshold(img,1,255,cv2.THRESH_BINARY)
    B[:,:]=threshNot[:,:,0]
    G[:,:]=threshNot[:,:,1]
    R[:,:]=threshNot[:,:,2]
    Y[:,:]=cv2.bitwise_and(R, G, mask=None)
    C[:,:]=cv2.bitwise_and(B, G, mask=None)
    M[:,:]=cv2.bitwise_and(R, B, mask=None)
    W[:,:]=cv2.bitwise_and(Y, B, mask=None)
    B[:,:]=cv2.bitwise_xor(B, C, mask=B)
    B[:,:]=cv2.bitwise_xor(B, M, mask=B)
    G[:,:]=cv2.bitwise_xor(G, C, mask=G)
    G[:,:]=cv2.bitwise_xor(G, Y, mask=G)
    R[:,:]=cv2.bitwise_xor(R, Y, mask=R)
    R[:,:]=cv2.bitwise_xor(R, M, mask=R)
    Y[:,:]=cv2.bitwise_xor(Y, W, mask=Y)
    mask_all= R+G+B+Y
    kernel = np.ones((3,3),np.uint8)
    Y = cv2.erode(Y,kernel,iterations = 1)
    G = cv2.erode(G,kernel,iterations = 1)
    R = cv2.erode(R,kernel,iterations = 1)
    B = cv2.erode(B,kernel,iterations = 1)
    mask_all=mask_all= R+G+B+Y
    Gray=Gray + (R/255)*50
    Gray=Gray + (G/255)*100
    Gray=Gray + (B/255)*150
    Gray=Gray + (Y/255)*200
    Gray=Gray.astype(np.uint8)
    depth=cv2.bitwise_and(mask_all, depth)
    Masked[:,:,0]=Gray
    Masked[:,:,1]=Gray
    Masked[:,:,2]=depth
    cv2.imwrite('Depth_GPU/' + 'Image' + img_number + '_R.png', Temp_depth.astype('uint8'))
    cv2.imwrite(CurrentFoder + "/Masked_ImageDepth/"+ img_number.rjust(5,'0') + '_R.png', Gray)
    cv2.imwrite(CurrentFoder + "/Masked_ImageDepth/"+ img_number.rjust(5,'0') + '_R_D.png', depth)
    cv2.imwrite(CurrentFoder + "/Masked_ImageDepth/"+ img_number.rjust(5,'0') + '_R_RGD.png', Masked)


#%% Convert the labels codes
  
import os               
import cv2                        
import glob
import numpy as np
depths = sorted(glob.glob("*_Cont.png"))
for i in range(len(depths)):    
    print(depths[i])	
    depth = cv2.imread(depths[i],-1)
    ret,Femur = cv2.threshold(depth,79,81,cv2.THRESH_BINARY)    
    ret,Tibia = cv2.threshold(depth,99,101,cv2.import cv2
import glob
import numpy as np
import os
CurrentDir='/home/jonmoham/DataForTraining/BlenderData'
os.chdir(CurrentDir)

Prefix = 'Movie_Long_'
CurrentFoder = os.path.abspath(os.path.join( os.getcwd()))
for Folder_Index in range(60):
    Destin = Prefix + str(Folder_Index+1) + '/Depth_GPU'
    os.chdir(Destin)
    depths = sorted(glob.glob("*_R.exr"))
    for i in range(len(depths)):
        Temp_depth = cv2.imread(depths[i],-1)
        Temp_depth[:,:,:][Temp_depth[:,:,:]==np.inf]=0.0
    ret,ACL   = cv2.threshold(depth,139,141,cv2.THRESH_BINARY)
    ret,Menisc= cv2.threshold(depth,219,221,cv2.THRESH_BINARY)
    Gray=np.zeros((256,256), np.uint8)
    Femur = (Femur[:,0:256]/255)*150
    Tibia = (Tibia[:,0:256]/255)*50
    Menisc= (Menics[:,0:256]/255)*200
    ACL   = (ACL[:,0:256]/255)*100
    Gray = Femur + Tibia + ACL + Menisc
    cv2.imwrite(depths[i][-15:-4]+'_D.png', Gray)



#%% convert the quatrenion to camer matrix
#http://fabiensanglard.net/doom3_documentation/37726-293748.pdf
import os  
import numpy as np
FolderName='/home/jonmoham/Python/tsdf-fusion-python-master/CustomData/CamWorld'
os.mkdir(FolderName)
with open('dataset_test_template.txt', newline='') as Odo:
    for cnt, line in enumerate(Odo):
        Coordinates=line.split(' ')
        Write_File = open(FolderName+'/'+Coordinates[0].split('/')[-3]+ '_' + Coordinates[0].split('/')[-1][:-4]+'.pose.txt', 'w')
        X=   np.float(Coordinates[1])
        Y=   np.float(Coordinates[2])
        Z=   np.float(Coordinates[3])
        W_O= np.float(Coordinates[4])
        X_O= np.float(Coordinates[5])
        Y_O= np.float(Coordinates[6])
        Z_O= np.float(Coordinates[7])
        World=np.zeros((4,4), np.float32)
        World[0][0]= 1 - 2*Y_O*Y_O - 2*Z_O*Z_O
        World[0][1]= 2*X_O*Y_O + 2*W_O*Z_O
        World[0][2]= 2*X_O*Z_O - 2*W_O*Y_O
        World[1][0]= 2*X_O*Y_O - 2*W_O*Z_O
        World[1][1]= 1 - 2*X_O*X_O - 2*Z_O*Z_O
        World[1][2]= 2*Y_O*Z_O + 2*W_O*X_O
        World[2][0]= 2*X_O*Z_O + 2*W_O*Y_O
        World[2][1]= 2*Y_O*Z_O - 2*W_O*X_O
        World[2][2]= 1 - 2*X_O*X_O - 2*Y_O*Y_O
        World[0][3]= X/1000
        World[1][3]= Y/1000
        World[2][3]= Z/1000
        World[3][3]= 1
        for i in range(4):
            #Write_File.write(str(World[i][0])+'e+00 '+str(World[i][1])+'e+00 '+str(World[i][2])+'e+00 '+ str(World[i][3])+'e+00 ' + '\n')
            Write_File.write(str(World[i][0])+' '+str(World[i][1])+' '+str(World[i][2])+' '+ str(World[i][3])+' ' + '\n')
        #if cnt==36:
            break
        Write_File.close()



import os  
import numpy as np
FolderName='.'
#os.mkdir(FolderName)
Write_File = open('data_train_NamesOnly.txt', 'w')

with open('dataset_train.txt', newline='') as Odo:
    for cnt, line in enumerate(Odo):
        cnt
        Coordinates=line.split(' ')
        Write_File.write(Coordinates[0].split(' ')[0]  + '\n')
        
    Write_File.close()

#%% Conversio of Kinect fusion pose to TSDF
import cv2
import glob
import numpy as np
import os
CurrentDir='/home/yaqub/Documents/Robotics/Codes/C-C++/OpenCVKinect/Pose'
os.chdir(CurrentDir)
Poses = sorted(glob.glob("*.txt"))
#Prefix = 'Movie_Long_'
camera_pose=np.zeros((4,4), np.float32)
for Folder_Index in range(len(Poses)):
    with open(Poses[Folder_Index], newline='\n') as Odo:
        f = open(Poses[Folder_Index][:-4]+'.pose2.txt', 'w')
        for cnt, line in enumerate(Odo):
            cnt
            Coordinates=line.split(',')
            print(np.float(Coordinates[0][1:]))
            if(cnt==3):
                camera_pose[0][3]=np.float(Coordinates[0][1:])
                camera_pose[1][3]=np.float(Coordinates[1][1:])
                camera_pose[2][3]=np.float(Coordinates[2][1:-2])
            else:
                camera_pose[cnt][0]=np.float(Coordinates[0][1:])
                camera_pose[cnt][1]=np.float(Coordinates[1][1:])
                camera_pose[cnt][2]=np.float(Coordinates[2][1:-2])
           
        f.write(str(camera_pose[0][0])[0:] + " ")
        f.write(str(camera_pose[0][1])[0:] + " ")
        f.write(str(camera_pose[0][2])[0:] + " ")
        f.write(str(camera_pose[0][3])[0:] + " \n")

        f.write(str(camera_pose[1][0])[0:] + " ")
        f.write(str(camera_pose[1][1])[0:] + " ")
        f.write(str(camera_pose[1][2])[0:] + " ")
        f.write(str(camera_pose[1][3])[0:] + " \n")
        f.write(str(camera_pose[2][0])[0:] + " ")
        f.write(str(camera_pose[2][1])[0:] + " ")
        f.write(str(camera_pose[2][2])[0:] + " ")
        f.write(str(camera_pose[2][3])[0:] + " \n")
        f.write(str(camera_pose[3][0])[0:] + " ")
        f.write(str(camera_pose[3][1])[0:] + " ")
        f.write(str(camera_pose[3][2])[0:] + " ")
        f.write(str(camera_pose[3][3])[0:] + " \n")

    f.close()



# remapping the segment labels
import os               
import cv2                        
import glob
import numpy as np
segments = sorted(glob.glob("*_Cont.png"))
for i in range(len(segments)):
    #img_name= depths[i][-10:-5]+'Right_R.png'
    segment   = cv2.imread(segments[i],-1)	
    #img   = cv2.imread('Image_Segmented/'+img_name,-1)
    #print(segments[i])	
    ret,Femur = cv2.threshold(segment,81,255,cv2.THRESH_TOZERO_INV)   #
    Femur=(Femur/80)*150
    ret,threshNot = cv2.threshold(segment,81,255,cv2.THRESH_TOZERO)   #
    
    ret,Tibia = cv2.threshold(threshNot,101,255,cv2.THRESH_TOZERO_INV)   #
    Tibia=(Tibia/100)*50
    ret,threshNot = cv2.threshold(threshNot,101,255,cv2.THRESH_TOZERO)   #
    
    ret,ACL = cv2.threshold(threshNot,141,255,cv2.THRESH_TOZERO_INV)   #
    ACL=(ACL/140)*100
    ret,threshNot = cv2.threshold(threshNot,141,255,cv2.THRESH_TOZERO)   #
    
    ret,Menisc = cv2.threshold(threshNot,221,255,cv2.THRESH_TOZERO_INV)   #
    Menisc=(Menisc/220)*200
    ret,threshNot = cv2.threshold(threshNot,221,255,cv2.THRESH_TOZERO)   #
    
    NewSeg=Femur+ACL+Tibia+Menisc
    cv2.imwrite(segments[i], NewSeg)
    

import os               
import cv2                        
import glob
import numpy as np
images = sorted(glob.glob("*.png"))
Img = cv2.imread(images[0])
r = 256.0 / 384
dim = (256, int(384 * r))
for i in range(len(images)):
    Img = cv2.imread(images[i])
    resized = cv2.resize(Img[0:384,0:384], dim, interpolation = cv2.INTER_AREA)
    cv2.imwrite(images[i][:-4]+'_L.png', resized)
    resized = cv2.resize(Img[0:384,404:], dim, interpolation = cv2.INTER_AREA)
    cv2.imwrite(images[i][:-4]+'_R.png', resized)



cam = bpy.data.objects['Camera']
SLAM_loc=[-10.528643, 8.768125, 3.436448]
SLAM_quat=[0.121505, 0.105289, 0.964742, -0.208384]
SLAM_loc=Vector(SLAM_loc)
SLAM_quat=Quaternion(SLAM_quat)
Coordinates=np.zeros((8),np.float32)
Coordinates[1:4]=SLAM_loc
Coordinates[4:8]=SLAM_quat
SLAM_matrix=quatr2world(Coordinates)
SLAM_matrix=Matrix(SLAM_matrix)
Blender_matrix=VSLAMMappingFromDSO2Blender(SLAM_loc, SLAM_quat)
cam.matrix_world=Blender_matrix



SLAM_loc=[-4.703865, 7.924488, 12.060865]
SLAM_quat=[-0.285065, 0.735782, 0.490916, -0.369275 ]


8.195742 6.320228 19.106552 0.004588 -0.325111 0.927060 0.186657

0.123074 -0.074312 0.978190 0.149918 

SLAM_loc=[6.960927, 9.406644, 17.793827]
SLAM_quat=[-0.245678, 0.655326, 0.674957, -0.233716  ]

SLAM_loc=[9.476293563842773, 13.061704635620117, 14.712431907653809]
SLAM_quat=[-0.282746317725, 0.157868662078, 0.936426302872, 0.135047118113 ]


import os               
import cv2                        
import glob
import numpy as np
images = sorted(glob.glob("*.png"))
Mean=np.zeros((256,256), np.float64)
for i in range(len(images)):
    Img = cv2.imread(images[i])
    Mean=Mean+Img[:,:,0]
    

# color histogram
import numpy as np
import cv2
from matplotlib import pyplot as plt
#img = cv.imread('home.jpg')
mask = np.zeros(img.shape[:2], np.uint8)
mask[100:300, 100:400] = 255
masked_img = cv2.bitwise_and(img,img,mask = mask)
# Calculate histogram with mask and without mask
# Check third argument for mask
hist_full = cv2.calcHist([img],[0],None,[256],[0,256])
hist_mask = cv2.calcHist([img],[0],mask,[256],[0,256])
plt.subplot(221), plt.imshow(img, 'gray')
plt.subplot(222), plt.imshow(mask,'gray')
plt.subplot(223), plt.imshow(masked_img, 'gray')
plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
plt.xlim([0,256])
plt.show()
    

#%% Thresholding for the 4 colors
# define the list of boundaries R G B Y  [B G R]
import numpy as np
import cv2
from matplotlib import pyplot as plt
boundaries = [
	([2, 2, 90], [70, 70, 200]),            #R1
	([50, 50, 200], [70, 70, 225]),          #R2
	([70, 70, 220], [125, 135, 255]),        #R3
	([2, 90, 2], [70, 200, 70]),            #G1
	([50, 200, 50], [70, 225, 70]),          #G2
	([70, 220, 70], [125, 255, 135]),        #G3
	([90, 2, 2], [200, 70, 70]),            #B1
	([200, 50, 50], [225, 70, 70]),          #B2
	([220, 70, 70], [255, 135, 125]),         #B3
	([1, 90, 90], [70, 200, 200]),         #Y1
	([50, 200, 200], [70, 225, 225]),        #Y2
	([70, 220, 220], [125, 255, 255])        #Y3
]

OffSet_R_Coeff= [1.45, 1.15, 1.] # right
OffSet_L_Coeff= [1.5 , 1.2, 1.] # left
img2=np.zeros((384,384,3), np.uint16)
img2[:]=img[:]

img2[:,:,0] = img2[:,:,0]*OffSet_R_Coeff[0]
img2[:,:,1] = img2[:,:,1]*OffSet_R_Coeff[1]
img2[:,:,2] = img2[:,:,2]*OffSet_R_Coeff[2]
img2[:,:,0][img2[:,:,0]>255]=255
img2[:,:,1][img2[:,:,1]>255]=255
img2[:,:,2][img2[:,:,2]>255]=255
img2 = img2.astype('uint8')

# loop over the boundaries
mask_total = np.zeros(img.shape[:2], np.uint8)
for (lower, upper) in boundaries:
	# create NumPy arrays from the boundaries
	lower = np.array(lower, dtype = "uint8")
	upper = np.array(upper, dtype = "uint8")
	img2 = cv2.GaussianBlur(img2, (19, 19), 0)
	# find the colors within the specified boundaries and apply
	# the mask
	mask = cv2.inRange(img2, lower, upper)
	#th3 = cv.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
	mask = cv2.erode(mask, None, iterations=1)
	mask = cv2.dilate(mask, None, iterations=2)
	mask = cv2.erode(mask, None, iterations=0)
	mask_total = mask_total + mask
output = cv2.bitwise_and(img2, img2, mask = mask_total)
# show the images
cv2.imshow("images", np.hstack([img2, output]))
cv2.waitKey(1000)
#cv2.destroyAllWindows()


blurred = cv2.GaussianBlur(img, (11, 11), 0)
hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
#mask = cv2.inRange(hsv, greenLower, greenUpper)
#mask = cv2.erode(mask, None, iterations=2)
#mask = cv2.dilate(mask, None, iterations=2)

#mask = cv2.inRange(img,lower_blue,upper_blue)
plt.imshow(mask)
plt.show()


for i,col in enumerate(color):
    histr = cv.calcHist([img],[i],mask,[256],[0,256])
    plt.subplot(221), plt.plot(histr,color = col), plt.xlim([0,256])
    plt.subplot(222), plt.imshow(masked_img,'gray')
plt.show()    
    




from pygtail import Pygtail
import sys
from os import path
import time

FileName='Logs.3.txt'
FileW=open(FileName, 'a'); 
Frame = 0
for line in Pygtail(FileName): 
    #sys.stdout.write(line) 
    Frame+=1
    print(str(Frame))
print('So far number of frames: ' + str(Frame))
#Frame = 0
while (True):
    #FileW.write(str(time.time())); 
    #FileW.write('\n'); FileW.flush();
    time.sleep(.01)   
    for line in Pygtail(FileName): 
       #sys.stdout.write(line) 
       if line[0:7] == '<<c4a52':
           print(line)
           print(time.time())
           #time.sleep(.1)
           Frame=Frame+1
           print(Frame)
FileW.close();

# record the time stamps of the NDI tracker
from pygtail import Pygtail
import sys
from os import path
import time
import argparse

class RecordTime():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        #self.initialized = False
    def getNames(self):
        self.parser.add_argument('--fileName', required=True, help='name of the CSV file where the coordinates will be recirded using the NDI software')
        self.parser.add_argument('--timeName', required=True, help='name of the file where the time stamps will be resorced')

#FileIn= 'Test8.csv'
#FileOut='Test8.txt'
FileIn = RecordTime.fileName
FileOut = RecordTime.timeName
if (path.exists(FileIn) or (path.exists(FileOut))):
    raise Exception("File in or out already exists. Try another file name.") 
else:
    FileW=open(FileIn, 'a'); 
    FileW2=open(FileOut, 'w')
Frame = 0
for line in Pygtail(FileIn): 
    #sys.stdout.write(line) 
    if (line[1:6] == ',Port'):
        FileW2.write(':Frame '); 
        FileW2.write('  Time: '); 
        FileW2.write('UnknownTime'); 
        FileW2.write('\n'); 
        FileW2.flush();
        Frame+=1
print(str(Frame) + ' frames do not have time staps')
#Frame = 0
while (True):
    time.sleep(.002)   
    for line in Pygtail(FileIn): 
        if (line[1:6] == ',Port'):
            #sys.stdout.write(line) 
            FileW2.write(':Frame '); 
            FileW2.write('  Time: '); 
            FileW2.write(str(time.time())); 
            FileW2.write('\n'); 
            FileW2.flush();
            Frame+=1
            print('Frame: ' + str(Frame))
FileW.close();
FileW2.close();




# Reading HDF5
f = h5py.File('Testt.hdf5', 'r')
Times = list(f.keys())[0]
Right = np.array(f[Times]['Right']) 
Left = np.array(f[Times]['Left'])
Left2=Left[Left!=0]


# create the pose files from the hdf5
import csv
import os  
import numpy as np
import h5py
import glob

def quatr2euler(Quaternion):         #w x y z 
    #roll = np.float64(Euler[0]) 
    #pitch= np.float64(Euler[1]) 
    #yaw  = np.float64(Euler[2]) 
    Euler = np.zeros((3), np.float64)  
    sinr_cosp = 2 * (Quaternion[0] * Quaternion[1] + Quaternion[2] * Quaternion[3]); 
    cosr_cosp = 1 - 2 * (Quaternion[1] * Quaternion[1] + Quaternion[2] * Quaternion[2]); 
    Euler[0] = np.arctan2(sinr_cosp, cosr_cosp); 
    sinp = 2 * (Quaternion[0] * Quaternion[2] - Quaternion[3] * Quaternion[1]); 
    if (np.abs(sinp) >= 1): 
        Euler[1] = np.copysign(np.pi / 2, sinp); # use 90 degrees if out of range 
    else: 
        Euler[1] = np.arcsin(sinp); 
    siny_cosp = 2 * (Quaternion[0] * Quaternion[3] + Quaternion[1] * Quaternion[2]); 
    cosy_cosp = 1 - 2 * (Quaternion[2] * Quaternion[2] + Quaternion[3] * Quaternion[3]); 
    Euler[2] = np.arctan2(siny_cosp, cosy_cosp); 
    return  Euler; #roll, pitch, yaw 


Cam_OffSet = 1 # becasue the first frame of the video does not have the corresponding time stamp
Frame_Lag = 9.2 # the more of this the more the NDI time goes ahead of image time
Time_Lag = (0.03*Frame_Lag)*(0.03/0.025)  # there is lag between the camera and the NDI
Drift_Lag = 2.2 # In creasing this lengthen the NDI time course
Drift = (0.03*Drift_Lag)*(0.03/0.025)/2500
Drift_Tot = 0
#Suffix = 'AM1330'
Suffix = glob.glob('AM1403.txt')[0][:-4]
CurrentFoder = os.path.abspath(os.path.join( os.getcwd()))
Cam_time_file = h5py.File(glob.glob('*.hdf5')[0], 'r')
#NDI_time_file = open(glob.glob('AM*.txt')[0], 'r')
NDI_time_file = open(glob.glob(Suffix+'.txt')[0], 'r')
Pose_file = glob.glob('*.csv')[0]
Times = list(Cam_time_file.keys())[0]
Cam_pose_file_Eul = open(glob.glob(Suffix+'.txt')[0][:-4]+'_Euler_pose20.txt', 'w')
Cam_pose_file = open(glob.glob(Suffix+'.txt')[0][:-4]+'_pose20.txt', 'w')

#Right = np.array(Cam_time_file[Times]['Right']) 
Left = np.array(Cam_time_file[Times]['Left'])
Cam_time= Left[Left!=0]
Cam_pose= np.zeros((len(Cam_time),7), np.float64) 

NDI_times = NDI_time_file.read().split('\n') 
World_stack = np.zeros((4,4), np.float32)
with open(Pose_file, newline='') as csvfile:
    poseReader = csv.reader(csvfile, delimiter=',', quotechar='|')
    Index_ref = 0
    next(poseReader)
    for row in poseReader:
        print(row[4])
        if (row[4]=='OK'):
            Index_ref += 1
            print(Index_ref)
    Index_ref = Index_ref -30
    NDI_time = np.zeros((Index_ref,1), np.float64)
    NDI_pose = np.zeros((Index_ref,7), np.float64)    
    Index = 0
with open(Pose_file, newline='') as csvfile:
    poseReader = csv.reader(csvfile, delimiter=',', quotechar='|')
    next(poseReader)
    for row in poseReader:
        if Index == Index_ref:
            break        
        if (row[4]=='OK'):
            NDI_time[Index] = float(NDI_times[Index][14:])
            #print(' '.join(row))
            Qw = float(row[5]) 
            Qx = float(row[6]) 
            Qy = float(row[7]) 
            Qz = float(row[8]) 
            Tx = float(row[9]) 
            Ty = float(row[10])
            Tz = float(row[11])
            NDI_pose[Index][0:7] = [Tx, Ty, Tz, Qw, Qx, Qy, Qz] 
            ##############################################################################################
            #then rotate the coordinate +90 -6 (for the sensor to cam offset) degree to align it with the computer vision coordinates (x rightward, y downward, z outward)
#            World_stack, _ = worldRotatZ(quatr2world(NDI_pose[Index][0:7].copy()), (+84)/57.29578)
#            NDI_pose[Index][3:7] = world2quatr(World_stack.copy()) 
#            NDI_pose[Index][0:3] = World_stack[0:3,3]
            ##############################################################################################
            Index += 1
#    for Index in range(Index_ref):
#        NDI_time[Index] = float(NDI_times[Index][14:])
Cam_pose_file.write('Frame Tx Ty Tz Qw Qx Qy Qz\n')   
Cam_pose_file_Eul.write('Frame Tx Ty Tz Wx Wy Wz\n')   
for Cam_Index in range(len(Cam_time)):
    NDI_time_err = np.zeros((Index_ref,1), np.float64)    
    print(Cam_Index)
    Drift_Tot = Drift_Tot + Drift
    print(Drift_Tot)
    for Err_Index in range(Index_ref):
        NDI_time_err[Err_Index] = np.abs(NDI_time[Err_Index] - Cam_time[Cam_Index] + Time_Lag - Drift_Tot)
    MinIndex = np.argmin((NDI_time_err))
    print(NDI_time_err[MinIndex])
    if (NDI_time_err[MinIndex] < 0.03):   
        Cam_pose[Cam_Index][:] = NDI_pose[MinIndex]
        Cam_pose_file.write(str(Cam_Index + Cam_OffSet) + ' ')
        Cam_pose_file.write(str(Cam_pose[Cam_Index][0]).rjust(5,'0') + ' ')
        Cam_pose_file.write(str(Cam_pose[Cam_Index][1]).rjust(5,'0') + ' ')    
        Cam_pose_file.write(str(Cam_pose[Cam_Index][2]).rjust(5,'0') + ' ')    
        Cam_pose_file.write(str(Cam_pose[Cam_Index][3]).rjust(5,'0') + ' ')    
        Cam_pose_file.write(str(Cam_pose[Cam_Index][4]).rjust(5,'0') + ' ')    
        Cam_pose_file.write(str(Cam_pose[Cam_Index][5]).rjust(5,'0') + ' ')    
        Cam_pose_file.write(str(Cam_pose[Cam_Index][6]).rjust(5,'0') + '\n')  
        
        EulerAngles = quatr2euler(Cam_pose[Cam_Index][3:7])
        Cam_pose_file_Eul.write(str(Cam_Index + Cam_OffSet) + ' ')
        Cam_pose_file_Eul.write(str(Cam_pose[Cam_Index][0]).rjust(5,'0') + ' ')
        Cam_pose_file_Eul.write(str(Cam_pose[Cam_Index][1]).rjust(5,'0') + ' ')    
        Cam_pose_file_Eul.write(str(Cam_pose[Cam_Index][2]).rjust(5,'0') + ' ')  
        Cam_pose_file_Eul.write('{:.7f}'.format(EulerAngles[0])  + ' ')
        Cam_pose_file_Eul.write('{:.7f}'.format(EulerAngles[1])  + ' ')
        Cam_pose_file_Eul.write('{:.7f}'.format(EulerAngles[2])  + '\n')
        
NDI_time_file.close()
Cam_pose_file.close()
Cam_pose_file_Eul.close()


# Merge the pose file and the image list
#import csv
import os  
import numpy as np
import h5py
import glob
CurrentFoder = os.path.abspath(os.path.join( os.getcwd()))
#Suffix = 'AM1413'

Name_file = open(glob.glob('Names2.txt')[0], 'r')
Name_list = Name_file.readlines()
Pose_file = open(glob.glob(Suffix+'_pose20.txt')[0], 'r')
Pose_list = Pose_file.readlines()
Pose_file_Eul = open(glob.glob(Suffix+'_Euler_pose20.txt')[0], 'r')
Pose_list_Eul = Pose_file_Eul.readlines()
NamePose_file = open((Suffix+'_pose2.txt')[:-9]+'NameAndPose20.txt', 'w')
NamePose_file_Eul = open((Suffix+'_Euler_pose2.txt')[:-9]+'NameAndPose20.txt', 'w')
NamePose_file.write('Air Knee Dataset \n') 
NamePose_file.write('ImageFile, Camera Position [X Y Z W P Q R] \n \n')
NamePose_file_Eul.write('Air Knee Dataset \n') 
NamePose_file_Eul.write('ImageFile, Camera Position [X Y Z Wx Wy Wz] \n \n')
Pose_vector = np.zeros((len(Pose_list), 6), np.float32)
Pose_vector_Eul = np.zeros((len(Pose_list_Eul), 3), np.float32)

for Index_Name in range(len(Name_list)):
    #Image_Num = int(Name_list[Index_Name][-17:-12]) 
    Image_Num = int(Name_list[Index_Name][0:-5]) 
    for Index_Pose in range(len(Pose_list)-1):
        Index_Pose += 1
        if (int(Pose_list[Index_Pose].split(' ')[0])==Image_Num):    
            Pose = Pose_list[Index_Pose].split(' ')
            Pose_Eul = Pose_list_Eul[Index_Pose].split(' ')
            Pose_vector[Index_Pose][0] = float(Pose[1])
            Pose_vector[Index_Pose][1] = float(Pose[2])
            Pose_vector[Index_Pose][2] = float(Pose[3])
            Pose_vector[Index_Pose][3] = float(Pose_Eul[4])
            Pose_vector[Index_Pose][4] = float(Pose_Eul[5])
            Pose_vector[Index_Pose][5] = float(Pose_Eul[6])
            #NamePose_file.write(str(Image_Num)+ ' ')
            NamePose_file.write(Name_list[Index_Name][:][:-1] + ' ')
            NamePose_file.write(Pose[1]+' '+Pose[2]+' '+Pose[3]+' '+Pose[4]+' '+Pose[5]+' '+Pose[6]+' '+Pose[7][:-1]+' \n')
            NamePose_file_Eul.write(Name_list[Index_Name][:][:-1] + ' ')
            NamePose_file_Eul.write(Pose_Eul[1]+' '+Pose_Eul[2]+' '+Pose_Eul[3]+' '+Pose_Eul[4]+' '+Pose_Eul[5]+' '+Pose_Eul[6][:-1]+' \n')

Name_file.close()
Pose_file.close()
Pose_file_Eul.close()
NamePose_file.close()
NamePose_file_Eul.close()

plt.plot(Pose_vector)
plt.show()

# Plot quaternion pos file
import os  
import numpy as np
import glob
from matplotlib import pyplot as plt
CurrentFoder = os.path.abspath(os.path.join( os.getcwd()))

#Pose_file = open(glob.glob('*_test.txt')[0], 'r')
Pose_file = open('/run/user/1000/gvfs/smb-share:server=hpc-fs.qut.edu.au,share=jonmoham/Python/tsdf-fusion-python-master/CustomData/2020-05-26-16-07-19_LR/pose2/2020-05-26-16-07-19_Deep.txt', 'r')
Pose_file = open('/home/yaqub/Documents/Robotics/Videos/Stereo6/3DPrintedKnee/Aligned_water/2020-09-18-13-30-12/AM1330_pose.txt', 'r')
Pose_list = Pose_file.readlines()
Pose_vector = np.zeros((len(Pose_list), 7), np.float32)

Offset = 1
for Index_Pose in range(len(Pose_list)-Offset):
    Index_Pose += Offset

    Pose = Pose_list[Index_Pose].split(' ')
    Pose_vector[Index_Pose][0] = float(Pose[1])
    Pose_vector[Index_Pose][1] = float(Pose[2])
    Pose_vector[Index_Pose][2] = float(Pose[3])
    Pose_vector[Index_Pose][3] = float(Pose[4])
    Pose_vector[Index_Pose][4] = float(Pose[5])
    Pose_vector[Index_Pose][5] = float(Pose[6])    
    Pose_vector[Index_Pose][6] = float(Pose[7])

Pose_file.close()

plt.plot(Pose_vector)
plt.show()

#%%
import os               
import cv2                        
import glob
import numpy as np
CurrentFoder = '/home/yaqub/Documents/Robotics/Videos/Stereo6/3DPrintedKnee/Good'
#os.chdir(CurrentFoder+'/Depth_GPU')
#depths = sorted(glob.glob("*_R.exr"))
os.chdir(CurrentFoder+'/Scratch/2020-05-26-16-07-19')
images = sorted(glob.glob("*_L_Cont.png"))
os.chdir(CurrentFoder)
#for i in range(2):    
#for i in range(120):    
r = 256.0 / 384
dim = (256, int(384 * r))
for i in range(len(images)):   
    img_number = images[i][:len(images[i])-12]
    #img   = cv2.imread(images[i],-1)	
    img2 = cv2.imread('Scratch/2020-05-26-16-07-19/'+ images[i],-1)
    print(images[i][:])	
    depth = cv2.imread('Scratch/2020-05-26-16-07-19/'+ images[i][:-10] + 'D.png',-1)
    #Temp_depth = depth[:,:,0]
    #Temp_depth[:,:][Temp_depth[:,:]==np.inf]=0.0
    #depth = Temp_depth.astype('uint8')
    img = cv2.resize(img2[:], dim, interpolation = cv2.INTER_AREA)
    depth=cv2.resize(depth, dim, interpolation = cv2.INTER_AREA)
    R=np.zeros((256,256), np.uint8)
    G=np.zeros((256,256), np.uint8)
    B=np.zeros((256,256), np.uint8)
    Y=np.zeros((256,256), np.uint8)
    C=np.zeros((256,256), np.uint8)
    M=np.zeros((256,256), np.uint8)
    W=np.zeros((256,256), np.uint8)
    Gray=np.zeros((256,256), np.uint8)
    Masked=np.zeros((256,256,3), np.uint8)
    #ret,thresh1 = cv2.threshold(img,250,255,cv2.THRESH_TOZERO)
    #ret,thresh1 = cv2.threshold(img,250,255,cv2.THRESH_BINARY)
    ret,threshNot = cv2.threshold(img,1,255,cv2.THRESH_BINARY)
    B[:,:]=threshNot[:,:,0]
    G[:,:]=threshNot[:,:,1]
    R[:,:]=threshNot[:,:,2]
    Y[:,:]=cv2.bitwise_and(R, G, mask=None)
    C[:,:]=cv2.bitwise_and(B, G, mask=None)
    M[:,:]=cv2.bitwise_and(R, B, mask=None)
    W[:,:]=cv2.bitwise_and(Y, B, mask=None)
    #B[:,:]=cv2.bitwise_xor(B, C, mask=B)
    #B[:,:]=cv2.bitwise_xor(B, M, mask=B)
    G[:,:]=cv2.bitwise_xor(G, C, mask=G)
    G[:,:]=cv2.bitwise_xor(G, Y, mask=G)
    R[:,:]=cv2.bitwise_xor(R, Y, mask=R)
    R[:,:]=cv2.bitwise_xor(R, M, mask=R)
    Y[:,:]=cv2.bitwise_xor(Y, W, mask=Y)
    mask_all= R+G+B+Y
    kernel = np.ones((3,3),np.uint8)
    Y = cv2.erode(Y,kernel,iterations = 1) # Menisc 200 in the Blender
    G = cv2.erode(G,kernel,iterations = 1) # Femur  150
    R = cv2.erode(R,kernel,iterations = 1) # ACL    100
    B = cv2.erode(B,kernel,iterations = 1) # Tibia  50
    mask_all=mask_all= R+G+B+Y
    Gray=Gray + (R/255)*100
    Gray=Gray + (G/255)*150
    Gray=Gray + (B/255)*50
    Gray=Gray + (Y/255)*200
    Gray=Gray.astype(np.uint8)
    #depth=cv2.bitwise_and(mask_all, depth)
    Masked[:,:,0]=Gray
    Masked[:,:,1]=Gray
    depth[:,:][Gray[:,:]==0]=0
    Masked[:,:,2]=depth
    #Masked[:,:,2]=depth
    #cv2.imwrite('Depth_GPU/' + 'Image' + img_number + '_R.png', Temp_depth.astype('uint8'))
    #cv2.imwrite(CurrentFoder + "/Masked_Image/"+ images[i], Gray)
    #cv2.imwrite(CurrentFoder + "/Masked_Image/"+ images[i][:-10] + '_L_D.png', depth)
    cv2.imwrite(CurrentFoder + "/Scratch/Masked_Image/"+ images[i][:-10] + 'L_RGD.png', Masked)


#%% rectify images
import csv
import os  
import numpy as np
import h5py
import glob

filename = '/home/yaqub/Documents/Robotics/Codes/Python/CalibrationData/Stereo6_Air2.hdf5'
f = h5py.File(filename, 'r')

# List all groups
print("Keys: %s" % f.keys())
#LeftCamAttr = list(f.keys())[0]
#RightCamAttr = list(f.keys())[1]
LeftCamAttr = list(f.keys())[2]
RightCamAttr = list(f.keys())[3]
mapL1 = np.array(f[LeftCamAttr]['One'])
mapL2 = np.array(f[LeftCamAttr]['Two'])
mapR1 = np.array(f[RightCamAttr]['One'])
mapR2 = np.array(f[RightCamAttr]['Two'])
images = sorted(glob.glob("*.png"))
#images = sorted(glob.glob("*_L_Cont.png"))
#os.chdir(CurrentFoder)
#for i in range(2):    
#for i in range(120):    
r = 256.0 / 384
dim = (256, int(384 * r))
for i in range(len(images)):
    img=cv2.imread(images[i], -1)
    grayL = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    undistorted_rectifiedL = cv2.remap(grayL, mapL1, mapL2, cv2.INTER_LINEAR);
    #undistorted_rectifiedR = cv2.remap(grayR, mapR1, mapR2, cv2.INTER_LINEAR);
    img[:,:,0] = cv2.remap(img[:,:,0], mapL1, mapL2, cv2.INTER_LINEAR);
    img[:,:,1] = cv2.remap(img[:,:,1], mapL1, mapL2, cv2.INTER_LINEAR);
    img[:,:,2] = cv2.remap(img[:,:,2], mapL1, mapL2, cv2.INTER_LINEAR);
    cv2.imwrite('../Test/' + images[i], img)




#%% conver the coordinates scan the coordinates rotations and flips
import numpy as np
#import cv2
import bpy, _cycles
import os
from mathutils import *
import sys
from mathutils import *
import math

Pose_File="/home/yaqub/Downloads/Temp/PoseComparison/2020-09-18-14-03-36_sub_filt.txt"
Pose_file = open(Pose_File, 'r')
Pose_list = Pose_file.readlines()
#NamePose_file = open("/home/yaqub/Downloads/test_145_SegDepth_New2.txt", 'w')
offset = 0
cam = bpy.data.objects['Camera']
CoordQuater = cam.rotation_quaternion
CoordTrans  = cam.location
#Name_Pose_file = open("/home/yaqub/Downloads/Test2.txt", 'w')
Angles = [math.pi/2, math.pi, -math.pi/2, 0]
Angles = np.array(Angles)
Axes =   [[1.0, 0.0 , 0.0], [0.0, 1.0 , 0.0], [0.0, 0.0 , 1.0], [0.0, 1.0 , 1.0], [1.0, 0.0 , 1.0], [1.0, 1.0 , 0.0], [1, 1, 1]] 
Axes = np.array(Axes)
i=0
for Axes_Index in range(len(Axes)):
    Axis = Axes[Axes_Index]
    for Rot_Index1 in range(3):
        Blender2DSO = Euler(Angles[Rot_Index1]*Axis , 'XYZ').to_matrix().to_4x4()
        for Rot_Index2 in range(3):
            mat_alignRotation = Euler(Angles[Rot_Index2]*Axis, 'XYZ').to_matrix().to_4x4()  
            #Name_Pose_file = open("/home/yaqub/Downloads/Name_Axis" + str(Axis[0])+ str(Axis[1])+ str(Axis[2])+'Rot_Index1'+str(Rot_Index1)+'Rot_Index2'+str(Rot_Index2)+ ".txt", 'w')
            Name_Pose_file = open(Pose_File[:-4] + str(i) + ".txt", 'w')
            print(i); i +=1
            for Index_Pose in range(len(Pose_list) - offset):
                Index_Pose = Index_Pose + offset
                Coordinates = Pose_list[Index_Pose].split(' ')
                CoordQuater[0] = np.float64(Coordinates[4])
                CoordQuater[1] = np.float64(Coordinates[5])
                CoordQuater[2] = np.float64(Coordinates[6])
                CoordQuater[3] = np.float64(Coordinates[7])
                CoordTrans [0] = np.float64(Coordinates[1])
                CoordTrans [1] = np.float64(Coordinates[2])
                CoordTrans [2] = np.float64(Coordinates[3])    
                #SLAMCorrdQuater = VSLAMMappingFromBlender2DSO(CoordTrans, CoordQuater)
                Trans = Matrix.Translation(CoordTrans)
                Rotat = CoordQuater.to_matrix().to_4x4()
                DSOPose = Blender2DSO * Trans * Rotat * mat_alignRotation
                #Name_Pose_file.write('/home/jonmoham/DataForTraining/BlenderData/Movie_Long_1/Masked_ImageDepth/' + Coordinates[0][0:-7]+'RGD.png ')
                #Name_Pose_file.write(Coordinates[0][0:-7]+'RGD.png ')
                Name_Pose_file.write(Coordinates[0]+' ')
#                Name_Pose_file.write(str(DSOPose.translation[0])+' ' + str(DSOPose.translation[1])+' ' + str(DSOPose.translation[2]))
#                Name_Pose_file.write(' ' + str(DSOPose.to_quaternion()[0])+' ' + str(DSOPose.to_quaternion()[1])+' ' + str(DSOPose.to_quaternion()[2])+' ' + str(DSOPose.to_quaternion()[3])+ ' \n') 
                Name_Pose_file.write(str("%04f " %DSOPose.translation[0])+ str("%04f " %DSOPose.translation[1])+ str("%04f " %DSOPose.translation[2]))
                Name_Pose_file.write(str("%04f " %DSOPose.to_quaternion()[0])+ str("%04f " %DSOPose.to_quaternion()[1]) + str("%04f " %DSOPose.to_quaternion()[2]) + str("%04f" %DSOPose.to_quaternion()[3])+ ' \n')                   
                  

Pose_file.close()

#            Trans = Matrix.Translation(CoordTrans[0:3])
#            Rotat = CoordQuater.to_matrix().to_4x4()
#            DSOPose = Blender2DSO * Trans * Rotat * mat_alignRotation
#            print(quatr2euler(world2quatr(np.array(DSOPose)[:3,:3]))*57.3) 

# Check the distance of a point with respect to a list
import os  
import numpy as np
CurrentFoder = os.path.abspath(os.path.join( os.getcwd()))


RefPose_file = open('dataset_train.txt', 'r')
RefPose_list = RefPose_file.readlines()
TargPose_file= open('test_5.txt', 'r')
TargPose_list= TargPose_file.readlines()
ErroPose_file= open('Error_5.txt', 'w')

os.chdir('/home/yaqub/Downloads/Temp')

TarPose_vector = np.zeros((len(TargPose_list), 7), np.float32)
RefPose_vector = np.zeros((len(RefPose_list), 7), np.float32)
ErrPose_vector = np.zeros((len(RefPose_list), 2), np.float32)

for Index_Pose in range(len(RefPose_list)):
    Pose = RefPose_list[Index_Pose].split(' ')
    RefPose_vector[Index_Pose][0] = float(Pose[1])
    RefPose_vector[Index_Pose][1] = float(Pose[2])
    RefPose_vector[Index_Pose][2] = float(Pose[3])
    RefPose_vector[Index_Pose][3] = float(Pose[4])
    RefPose_vector[Index_Pose][4] = float(Pose[5])
    RefPose_vector[Index_Pose][5] = float(Pose[6])            
    RefPose_vector[Index_Pose][6] = float(Pose[7]) 

for Index_Pose in range(len(TargPose_list)-1):
    Pose = TargPose_list[Index_Pose].split(' ')
    TarPose_vector[Index_Pose][0] = float(Pose[1])
    TarPose_vector[Index_Pose][1] = float(Pose[2])
    TarPose_vector[Index_Pose][2] = float(Pose[3])
    TarPose_vector[Index_Pose][3] = float(Pose[4])
    TarPose_vector[Index_Pose][4] = float(Pose[5])
    TarPose_vector[Index_Pose][5] = float(Pose[6])            
    TarPose_vector[Index_Pose][6] = float(Pose[7])     
    for Index_Pose2 in range(len(RefPose_list)): 
        ErrPose_vector[Index_Pose2][0] = np.linalg.norm(TarPose_vector[Index_Pose][0:3] -RefPose_vector[Index_Pose2][0:3]) 
        #ErrPose_vector[Index_Pose2][1] = np.linalg.norm(TarPose_vector[Index_Pose][3:] -RefPose_vector[Index_Pose2][3:]) 
        RefPose_vector[Index_Pose][4:] = -1* RefPose_vector[Index_Pose][4:]
        Stack = 57.296 * 2 *np.arccos(np.dot(TarPose_vector[Index_Pose][3:], RefPose_vector[Index_Pose2][3:]))
        ErrPose_vector[Index_Pose2][1] = np.min([np.abs(Stack), np.abs(360-Stack)])

    ErrPose_vector[:,0:2][ErrPose_vector[:, 0]> 20.] = 100
    MinIndex = np.argmin((ErrPose_vector[:, 1]))  
    ErroPose_file.write(TargPose_list[Index_Pose].split(' ')[0][-37:] + ' ' )    
    
    ErroPose_file.write(RefPose_list[MinIndex].split(' ')[0][-49:] + ' ' )    
    ErroPose_file.write(str(ErrPose_vector[MinIndex][0]) + ' ' +str(ErrPose_vector[MinIndex][1]) + ' \n' )    
    print(Index_Pose)
    
ErroPose_file.close()
TargPose_file.close()
RefPose_file.close()
 



import cv2
import glob
import numpy as np
import os
CurrentDir='/home/jonmoham/DataForTraining/BlenderData/'
os.chdir(CurrentDir)
Prefix = 'Movie_Long_'
#CurrentFoder = os.path.abspath(os.path.join( os.getcwd()))
for Folder_Index in range(1):
    Folder_Index=3
    Destin = CurrentDir +  Prefix + str(Folder_Index) + '/Image_GPU'
    #os.mkdir(CurrentDir + Prefix + str(Folder_Index+1) + '/Image_GPU')
    os.chdir(Destin)
    depths = sorted(glob.glob("*.png"))
    for i in range(len(depths)):
        Temp_depth = cv2.imread(depths[i],-1)
        #Temp_depth[:,:,:][Temp_depth[:,:,:]==np.inf]=0.0
        #Disp = 20000.0/Temp_depth[:,:,0] + 1000
        cv2.imwrite(CurrentDir + Prefix + str(Folder_Index) + '/Image_GPU/' + depths[i][0:-3] + 'png', Temp_depth[:,:,0:3].astype('uint8'))
        print(i/2)
    os.chdir(CurrentDir) 




import cv2
import glob
import numpy as np
import os
Images = sorted(glob.glob("*.png"))
for i in range(len(Images)):
#for i in range(25000):
   
    #NameL = 'Image' + str(i).rjust(5, '0') + '_L.png'
    #NameR = 'Image' + str(i).rjust(5, '0') + '_R.png' 
    #Image = cv2.imread(NameL,-1)
    #cv2.imwrite('../Disparity2/' + NameL , Image)
    Image = cv2.imread(Images[i],-1)
    #cv2.imwrite('../Disparity2/' + NameR , Image)
    #os.system('cp NameR ../Image_GPU2')    
    #os.system('cp NameR ../Image_GPU2')    
    #cv2.imwrite(Images[i][0:4].rjust(5,'0')+Images[i][4:] , Image[:,:,0:3].astype('uint8'))
    cv2.imwrite(Images[i] , Image[:,:,0:3])
    #cv2.imwrite(CurrentDir + Prefix + str(Folder_Index) + '/Image_GPU/' + depths[i][0:-3] + 'png', Temp_depth[:,:,0:3].astype('uint8'))
    print(Images[i])
#img_number.rjust(5,'0')



#
import os               
import cv2                        
import glob
import numpy as np
images = sorted(glob.glob("*disp.jpeg"))
Img = cv2.imread(images[0])
r = 256.0 / Img.shape[1]
dim = (256, int(Img.shape[0] * r))
for i in range(len(images)):
    Img = cv2.imread(images[i])
    Img[:100, 160:] = 0
    cv2.imwrite(images[i], Img)




images = sorted(glob.glob("image_03/data/*.png"))
Img = cv2.imread(images[0])
r = 256.0 / Img.shape[1]
dim = (256, int(Img.shape[0] * r))
for i in range(len(images)):
    Img = cv2.imread(images[i])
    resized = cv2.resize(Img, dim, interpolation = cv2.INTER_AREA)
    cv2.imwrite(images[i], resized)




def world2quatr(World):                # 
    tr = World[0][0] + World[1][1] + World[2][2]
    if (tr > 0):
        S = np.sqrt(tr+1.0) * 2# // S=4*qw
        qw = 0.25 * S;
        qx = (World[2][1] - World[1][2]) / S;
        qy = (World[0][2] - World[2][0]) / S; 
        qz = (World[1][0] - World[0][1]) / S;
    elif ((World[0][0] > World[1][1])and(World[0][0] > World[2][2])):
        S = np.sqrt(1.0 + World[0][0] - World[1][1] - World[2][2]) * 2; # S=4*qx
        qw = (World[2][1] - World[1][2]) / S;
        qx = 0.25 * S;
        qy = (World[0][1] + World[1][0]) / S; 
        qz = (World[0][2] + World[2][0]) / S; 
    elif (World[1][1] > World[2][2]):
        S = np.sqrt(1.0 + World[1][1] - World[0][0] - World[2][2]) * 2; # S=4*qy
        qw = (World[0][2] - World[2][0]) / S;
        qx = (World[0][1] + World[1][0]) / S;
        qy = 0.25 * S;
        qz = (World[1][2] + World[2][1]) / S;
    else:
        S = np.sqrt(1.0 + World[2][2] - World[0][0] - World[1][1]) * 2; # S=4*qz
        qw = (World[1][0] - World[0][1]) / S;
        qx = (World[0][2] + World[2][0]) / S;
        qy = (World[1][2] + World[2][1]) / S;
        qz = 0.25 * S;
    quaternion=np.array([qw, qx, qy, qz]) 
    return   quaternion 


#%% plot the translation for the estimated odometery pose
from matplotlib import pyplot as plt
import numpy as np
Poses_Orig= np.load('/run/user/1000/gvfs/smb-share:server=hpc-fs.qut.edu.au,share=jonmoham/Python/monodepth2-master/3DprintedKnee_Stereo_Mono1234_PredicMask/mdp/models/weights_14/poses_2020-05-26-16-07-19.npy')
Poses = Poses_Orig.copy()
Poses_Quat = np.zeros((len(Poses_Orig),1, 7), np.float32) 
#Poses[:]     = Poses_Orig[:]
for Index in range(len(Poses)-1):
    Poses[Index+1, 0, 3] = np.sum(np.squeeze(Poses_Orig[0:Index+1, 0, 3]))
    Poses[Index+1, 1, 3] = np.sum(np.squeeze(Poses_Orig[0:Index+1, 1, 3]))
    Poses[Index+1, 2, 3] = np.sum(np.squeeze(Poses_Orig[0:Index+1, 2, 3]))
    Poses_Quat[Index+1, 0, 0:3] = np.squeeze(Poses_Orig[Index+1, 0:3, 3])
    Quaternion_angles = np.squeeze(world2quatr(np.squeeze(Poses_Orig[Index+1,:,:])))
    Poses_Quat[Index+1, 0, 3] = Quaternion_angles[0]
    Poses_Quat[Index+1, 0, 4] = Quaternion_angles[1]
    Poses_Quat[Index+1, 0, 5] = Quaternion_angles[2]
    Poses_Quat[Index+1, 0, 6] = Quaternion_angles[3]
Poses_Quat2 = Poses_Quat.copy()
for Index in range(len(Poses)-1):
    #Poses_Quat2[Index+1, 0, 3] = np.sum(np.squeeze(Poses_Quat[:Index+1, 0, 3]))
    Poses_Quat2[Index+1, 0, 4] = np.sum(np.squeeze(Poses_Quat[:Index+1, 0, 4]))
    Poses_Quat2[Index+1, 0, 5] = np.sum(np.squeeze(Poses_Quat[:Index+1, 0, 5]))
    Poses_Quat2[Index+1, 0, 6] = np.sum(np.squeeze(Poses_Quat[:Index+1, 0, 6]))
    
    
plt.plot(Poses_Quat[1:,0,4:])
plt.show()
Quaternion = world2quatr(np.squeeze(Poses[0,:,:]))


#% Reconstruction of the global and rotations translation from local pose estimation
###############################################################################
Poses_GT_LOC2= Poses_GT_GLB.copy()
Poses_REC[0,:,:] = quatr2world(Poses_Quat[100,0,:])
for i in range(1, len(Poses_GT_GLB)):
    Poses_GT_LOC2[i-1,:,:] = (np.dot(np.linalg.inv(Poses_GT_GLB[i-1,:,:]), Poses_GT_GLB[i,:,:]))
for i in range(1, len(Poses_GT_GLB)):
    Poses_REC[i,:,:] = (np.dot((Poses_REC[i-1,:,:]), Poses_GT_LOC2[i-1,:,:]))
###############################################################################
Poses_REC2 = Poses_GT_GLB.copy()
Poses_REC2[0,:,:] = quatr2world(Poses_Quat[100,0,:])
for i in range(1, len(Poses_GT_GLB)):
    Poses_REC2[i,:,:] = (np.dot((Poses_REC2[i-1,:,:]),     Poses_GT_LOC[i-1,:,:]))


#%% load the estimated odo by the monodepth2 and then plot it 
Pose_File = '/run/user/1000/gvfs/smb-share:server=hpc-fs.qut.edu.au,share=jonmoham/Python/monodepth2-master/3DprintedKnee_Stereo_Mono0-11_All_k328/mdp/models/weights_19/poses_2020-05-26-16-07-192.npy'
Pose_File = '/run/user/1000/gvfs/smb-share:server=hpc-fs.qut.edu.au,share=jonmoham/Python/monodepth2-master/PreTrained/Mono_Stereo_640x192/poses.npy'
Poses_3D = np.load(Pose_File)
Poses_3D = Poses_3D.astype(np.float64)
Poses_3D_GLB = Poses_3D.copy()
Poses_3D_GLB[0,:,:] = np.eye(4)
for i in range(1, len(Poses_3D)):
    Poses_3D_GLB[i,:,:] =  (np.dot((Poses_3D[i-1,:,:]),     Poses_3D_GLB[i-1,:,:]))
plt.plot(np.squeeze(Poses_3D_GLB[1:, 0:3, 3]))
plt.show()



Split=['poses_2019-10-24-13-01-04_LR',  'poses_2020-09-18-14-03-36_sub_filt.npy']
Source='/run/user/1000/gvfs/smb-share:server=hpc-fs.qut.edu.au,share=jonmoham/Python/monodepth2-master/'

Pose_File= Source+'Cadava5_3DPrint_Sheep_filt05_Bch4_Lr5-5_pairs_PoseOnlyBoth_NymLay50/mdp/models/weights_39/poses_test.npy'#+Split[1]
Pose_File= Source+'Cadava5_3DPrint_Sheep_filt05_Bch4_Lr5-5_pairs_PoseOnlyBoth_NymLay50/mdp/models/weights_39/'+Split[0]

Pose_File='/home/joon/Documents/Code/Python/MonoDepth2Enodo/3Dprint_water_Self/mdp/models/weights_25/poses_2020-09-18-14-03-36.npy'
Poses_3D = np.load(Pose_File) 
 
Poses_3D = Poses_3D.astype(np.float64) 
#Poses_3D_GLB = Poses_3D.copy() 
No_ref_fram = 1 
FrameIndex = 1 #  
Poses_3D_GLB = np.zeros((int(np.shape(Poses_3D)[0]/No_ref_fram), 4, 4), np.float64) 
Poses_3D_GLB[0,:,:] = np.eye(4) 
Quat_GLB = np.zeros((int(np.shape(Poses_3D)[0]/No_ref_fram), 1, 4), np.float64) 
Euler_GLB = np.zeros((int(np.shape(Poses_3D)[0]/No_ref_fram), 1, 3), np.float64) 
ii = 1 
for i in range( FrameIndex , len(Poses_3D)-No_ref_fram, No_ref_fram): 
    Poses_3D_GLB[ii,:,:] =  np.dot(Poses_3D[i-1,:,:],  Poses_3D_GLB[ii-1,:,:]) 
    Quat_GLB[ii]= world2quatr(np.squeeze(Poses_3D_GLB[ii,:,:])).astype(np.float64) 
    Euler_GLB[ii]=world2euler(np.squeeze(Poses_3D_GLB[ii,:,:])).astype(np.float64) 
    ii += 1 
    #print (i) 
#plt.plot(np.squeeze(Poses_3D_GLB[1:, 0:3, 3])) 
Sign = 1  
if FrameIndex <= No_ref_fram/2: 
    Sign = -1 

trans_weight = 1/np.array([.5, .6, 1.])
fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(np.squeeze(Poses_3D_GLB[1:, 0:1, 3])*38*Sign*trans_weight[0] + 0) 
ax1.plot(np.squeeze(Poses_3D_GLB[1:, 1:2, 3])*38*Sign*trans_weight[1]  - 0) 
ax1.plot(np.squeeze(Poses_3D_GLB[1:, 2:3, 3])*38*1.1*Sign*trans_weight[2] + 0)  
ax1.legend(['X', 'Y', 'Z']) 
ax2.plot(np.squeeze(Euler_GLB[:-1,:])*-1)
plt.show()






Poses_3D_GLBTot = np.zeros((int(np.shape(Poses_3D)[0]/No_ref_fram), 4, 4), np.float64)
for FrameIndex in range(No_ref_fram):
     
    Poses_3D_GLB = np.zeros((int(np.shape(Poses_3D)[0]/No_ref_fram), 4, 4), np.float64)
    Poses_3D_GLB[0,:,:] = np.eye(4)
    ii = 1
    for i in range( FrameIndex , len(Poses_3D)-No_ref_fram, No_ref_fram):
        Poses_3D_GLB[ii,:,:] =  (np.dot((Poses_3D[i-1,:,:]),     Poses_3D_GLB[ii-1,:,:]))
        Quat_GLB[ii]= world2quatr(np.squeeze(Poses_3D_GLB[ii,:,:])).astype(np.float64)
        ii += 1
    
    if (FrameIndex != 0) and (FrameIndex <= No_ref_fram/2 ):
        Poses_3D_GLB = Poses_3D_GLB*-1
    
    Poses_3D_GLBTot = Poses_3D_GLBTot + Poses_3D_GLB

Poses_3D_GLBTot = Poses_3D_GLBTot/No_ref_fram
trans_weight = 1/np.array([.6, .5, 1.])
#trans_weight = 1/np.array([1, 1, 1.])
fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(np.squeeze(Poses_3D_GLB[1:, 0:1, 3])*18*Sign*trans_weight[0] + 0) 
ax1.plot(np.squeeze(Poses_3D_GLB[1:, 1:2, 3])*18*Sign*trans_weight[1]  - 0) 
ax1.plot(np.squeeze(Poses_3D_GLB[1:, 2:3, 3])*18*Sign*trans_weight[2] + 0)  
ax1.legend(['X', 'Y', 'Z']) 
ax2.plot(np.squeeze(Euler_GLB[:-1,:])*-1)
plt.show()
    

Euler_GLB22=np.squeeze(Euler_GLB[:,:])  
Euler_GLB22[:,1]=Euler_GLB22[:,1]*(-1) 

plt.plot(np.squeeze(Euler_GLB[:,:])*-1)
plt.legend(['X', 'Y', 'Z'])
plt.show()



#%% counter rotate the monodepth2 to match the NDI coordidate
Poses_3D_GLB2 = Poses_3D_GLB.copy()  



#%%
trans_weight = 1/np.array([.5, .6, 1.])
Poses_3D_GLB[:] = Poses_3D_GLB2.copy() 
for i in range(len(Poses_3D_GLB)): 
    Poses_3D_GLB[i,0:3,3] = Poses_3D_GLB[i,0:3,3]* trans_weight
    World_stack, _ = worldRotatZ(Poses_3D_GLB[i,:,:].copy(), (-84)/57.29578)
    Poses_3D_GLB[i,:,:] = World_stack.copy()
    Quat_GLB[i]= world2quatr(np.squeeze(Poses_3D_GLB[i,:,:])).astype(np.float64) 
Sign = 1  
if FrameIndex <= No_ref_fram/2: 
    Sign = -1 
    
Poses_3D_GLB[1:, 0:1, 3]=Poses_3D_GLB[1:, 2:3, 3]*14*Sign + 0
Poses_3D_GLB[1:, 1:2, 3] = Poses_3D_GLB[1:, 1:2, 3]*20*Sign  - 0
Poses_3D_GLB[1:, 2:3, 3]=Poses_3D_GLB[1:, 2:3, 3]*35*1.1*Sign + 0
plt.plot(np.squeeze(Poses_3D_GLB[1:, 0:1, 3])) 
plt.plot(np.squeeze(Poses_3D_GLB[1:, 1:2, 3])) 
plt.plot(np.squeeze(Poses_3D_GLB[1:, 2:3, 3]))  
plt.legend(['X', 'Y', 'Z']) 
plt.show()



plt.plot(np.squeeze(Quat_GLB[1:, 0, :] - Quat_GLB[1, 0, :]))
plt.legend(['W', 'X', 'Y', 'Z'])
plt.show()








#File_Quatr = open(Pose_File.split('/')[-1][:-3] + 'txt', 'w')
File_Quatr = open(Pose_File[:-3] + 'txt', 'w')
start_image = 95
File_Quatr = open('/home/yaqub/Downloads/MonoDeposes_2020-10-08-13-36-36' + '.txt', 'w')
for i in range(len(Poses_3D_GLB)):
    quaternion = world2quatr(np.squeeze(Poses_3D_GLB[i,:,:])).astype(np.float64)
    File_Quatr.write(str(i+start_image).rjust(5,'0') + '.png ')
    File_Quatr.write(str(Poses_3D_GLB[i,0,3]) + ' ')
    File_Quatr.write(str(Poses_3D_GLB[i,1,3]) + ' ')
    File_Quatr.write(str(Poses_3D_GLB[i,2,3]) + ' ')
    File_Quatr.write(str(quaternion[0]) + ' ')
    File_Quatr.write(str(quaternion[1]) + ' ')
    File_Quatr.write(str(quaternion[2]) + ' ')
    File_Quatr.write(str(quaternion[3]) + '\n')
File_Quatr.close()






#%% convert a file from quaternion to world matrix

import os               
import glob
import numpy as np
import math
from matplotlib import pyplot as plt
#CurrentDir='/home/yaqub/Documents/Robotics/Videos/Stereo6/Sheep/2020-10-08-13-36-36'
CurrentDir='/home/yaqub/Documents/Robotics/Videos/Stereo6/3DPrintedKnee/Aligned_water/2020-09-18-14-03-36'
os.chdir(CurrentDir)
#Pose_file =  open(glob.glob('PM13:36_pose20.txt')[0], 'r')
Pose_file =  open(glob.glob('*AM1403_pose20.txt')[0], 'r')
Pose_Lines = Pose_file.readlines()
Pose_Quatr = np.zeros((len(Pose_Lines), 7), np.float64)
Pose_World = np.zeros((len(Pose_Lines), 4, 4), np.float64)
Pose_Euler = np.zeros((len(Pose_Lines), 6), np.float64)
for Index_Pose in range(len(Pose_Lines)-1):
    Index_Pose += 1
    Pose = Pose_Lines[Index_Pose].split(' ')
    Pose_Quatr[Index_Pose][0] = float(Pose[1])
    Pose_Quatr[Index_Pose][1] = float(Pose[2])
    Pose_Quatr[Index_Pose][2] = float(Pose[3])
    Pose_Quatr[Index_Pose][3] = float(Pose[4])
    Pose_Quatr[Index_Pose][4] = float(Pose[5])
    Pose_Quatr[Index_Pose][5] = float(Pose[6])            
    Pose_Quatr[Index_Pose][6] = float(Pose[7]) 
    Pose_World[Index_Pose, :, :] = quatr2world(Pose_Quatr[Index_Pose])
    Pose_Euler[Index_Pose,0:3] = Pose_Quatr[Index_Pose,0:3]
    Pose_Euler[Index_Pose,3:] = world2euler(np.squeeze(Pose_World[Index_Pose,:,:])) *57.296
Translation = np.zeros((len(Pose_Lines), 3), np.float64) 
Translation[:,0] = np.squeeze(Pose_World[:, 0:3, 3])[:,0] - np.squeeze(Pose_World[:, 0:3, 3])[100,0] #np.mean(np.squeeze(Pose_World[:, 0:3, 3])[:,0])
Translation[:,1] = np.squeeze(Pose_World[:, 0:3, 3])[:,1] - np.squeeze(Pose_World[:, 0:3, 3])[100,1] #np.mean(np.squeeze(Pose_World[:, 0:3, 3])[:,1])
Translation[:,2] = np.squeeze(Pose_World[:, 0:3, 3])[:,2] - np.squeeze(Pose_World[:, 0:3, 3])[100,2] #np.mean(np.squeeze(Pose_World[:, 0:3, 3])[:,2])

#plt.plot(Translation[1:,:1])
#plt.plot(Translation[1:,0:3]*(1))36-36
fig, (ax1, ax2) = plt.subplots(2)
#fig.suptitle('Vertically stacked subplots')

ax1.plot(np.squeeze(Translation[1:, 0]) - 0.0) 
ax1.plot(np.squeeze(Translation[1:, 1]) + 0) 
ax1.plot(np.squeeze(Translation[1:, 2]) + 0) 
ax2.plot(np.squeeze(Pose_Euler[1:, 3:] - Pose_Euler[1, 3:]))
plt.legend(['X', 'Y', 'Z'])
plt.show()




plt.plot(np.squeeze(Pose_Quatr[1:, 3:] - Pose_Quatr[1, 3:]))
plt.legend(['W', 'X', 'Y', 'Z'])
plt.show()


plt.plot(np.squeeze(Pose_Euler[1:, 3:] - Pose_Euler[1, 3:]))
plt.legend(['X', 'Y', 'Z'])
plt.show()

def quatr2world(Coordinates):         #
    X=   np.float64(Coordinates[0])
    Y=   np.float64(Coordinates[1])
    Z=   np.float64(Coordinates[2])
    W_O= np.float64(Coordinates[3])
    X_O= np.float64(Coordinates[4])
    Y_O= np.float64(Coordinates[5])
    Z_O= np.float64(Coordinates[6])
    World=np.zeros((4,4), np.float64)
    World[0][0]= 1 - 2*Y_O*Y_O - 2*Z_O*Z_O
    World[0][1]= 2*X_O*Y_O + 2*W_O*Z_O
    World[0][2]= 2*X_O*Z_O - 2*W_O*Y_O
    World[1][0]= 2*X_O*Y_O - 2*W_O*Z_O
    World[1][1]= 1 - 2*X_O*X_O - 2*Z_O*Z_O
    World[1][2]= 2*Y_O*Z_O + 2*W_O*X_O
    World[2][0]= 2*X_O*Z_O + 2*W_O*Y_O
    World[2][1]= 2*Y_O*Z_O - 2*W_O*X_O
    World[2][2]= 1 - 2*X_O*X_O - 2*Y_O*Y_O
    World[0][3]= X/1
    World[1][3]= Y/1
    World[2][3]= Z/1
    World[3][3]= 1
    World[0:3, 0:3] = np.transpose(World[0:3,0:3])
    return World


def world2euler(World):   # best so far.
    import sys
    tol = sys.float_info.epsilon * 10
    Euler = np.zeros((3), np.float64)
    #https://www.meccanismocomplesso.org/en/3d-rotations-and-euler-angles-in-python/  
    if abs(World.item(0,0))< tol and abs(World.item(1,0)) < tol:
       Euler[2] = 0
       Euler[1] = math.atan2(-World.item(2,0), World.item(0,0))
       Euler[0] = math.atan2(-World.item(1,2), World.item(1,1))
    else:   
       Euler[2] = math.atan2(World.item(1,0),World.item(0,0))
       sp = math.sin(Euler[0])
       cp = math.cos(Euler[0])
       Euler[1] = math.atan2(-World.item(2,0),cp*World.item(0,0)+sp*World.item(1,0))
       Euler[0] = math.atan2(sp*World.item(0,2)-cp*World.item(1,2),cp*World.item(1,1)-sp*World.item(0,1))

    return Euler

#% filter the singal
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

fs = 1000  # Sampling frequency
# Generate the time vector properly
t = np.arange(1000) / fs
#signala = np.sin(2*np.pi*100*t) # with frequency of 100
#plt.plot(t, signala, label='a')
#signalb = np.sin(2*np.pi*20*t) # frequency 20
#plt.plot(t, signalb, label='b')
#
#signalc = signala + signalb
#plt.plot(t, signalc, label='c')


t = np.arange(1000) / fs
fc = 50  # Cut-off frequency of the filter
w = fc / (fs / 2) # Normalize the frequency
b, a = signal.butter(5, w, 'low')
output = signal.filtfilt(b, a, signalc)
plt.plot(t, output, label='filtered')
plt.legend()
plt.show()



#%% Creating the file index like the kitty style
import os               
import cv2                        
import glob
import numpy as np
CurrentDir='/home/jonmoham/DataForTraining/Cadavar_2019-10-24/kitti_style_cropped'
os.chdir(CurrentDir)
Names =  open('Names_LR.txt', 'w')

Directories = sorted(glob.glob("*LR"))
for index_Dir in range(len(Directories)):
    os.chdir(Directories[index_Dir]+'/image_02/data')
    images = sorted(glob.glob("*.png"))
    index = 3
    while (index < (len(images)-3)):
        if  (int(images[index-3][0:5])+1 == int(images[index-2][0:5]) and 
                 int(images[index-2][0:5])+1 == int(images[index-1][0:5]) and
                 int(images[index-1][0:5])+1 == int(images[index+0][0:5]) and
                 int(images[index+0][0:5])+1 == int(images[index+1][0:5]) and
                 int(images[index+1][0:5])+1 == int(images[index+2][0:5]) and
                 int(images[index+2][0:5])+1 == int(images[index+3][0:5])):
            Names.write(Directories[index_Dir]+'/')
            Names.write(images[index+0][0:5] + '\n')
            print(index)
#        else:
#            if (int(images[index][0:5]) < int(images[-4][0:5])): 
#                index += 6
#                print('safdasfdasfdasdfasdfsa' + str(index))
        index += 1

    os.chdir(CurrentDir)
Names.close()
    

import os 
import cv2
import numpy as np
import glob
CurrentDir='/home/yaqub/Documents/Robotics/Videos/Stereo6/Cadava_24-10-2019'
os.chdir(CurrentDir)

TargPose_file= open('Names_Stereo.txt', 'r')
TargPose_list=TargPose_file.readlines()
for Index_Pose in range(len(TargPose_list)-1):
    Index_Pose +=16500
    img=cv2.imread(TargPose_list[Index_Pose][:-1],-1)
    cv2.imwrite(TargPose_list[Index_Pose][0:-12]+'Cropped'+TargPose_list[Index_Pose][86:-1], img)
    


#%% Euler to world
    
import csv
import os  
import numpy as np
import h5py
import glob
from scipy.io import savemat
from scipy.io import loadmat

x = loadmat('/home/yaqub/Documents/MATLAB/WorkSpace/RWHE-Calib-master/Datasets/kuka_3/RobotPoses.mat' )
CurrentDir='/home/yaqub/Documents/Robotics/Images/Calibration/Stereo6/Water4_Pose/Left_copy'
os.chdir(CurrentDir)
Euler = np.zeros((1,6), np.float64)    
Pose_files = glob.glob('*.csv')
Index = 0
Flatten_world = np.zeros((len(Pose_files), 16), np.float64)
for Pose_file_Index in Pose_files:    
    print(Pose_file_Index)
    img = cv2.imread('../Left/' + Pose_file_Index[:-4] + '-L.png')
    #Cam_pose_Mat = open(Pose_file_Index[:-8]+'.txt', 'w')
    Cam_pose_Mat = open('Pose_'+str(Index+1).rjust(3,'0')+'.txt', 'w')
    with open(Pose_file_Index, newline='') as csvfile:
        poseReader = csv.reader(csvfile, delimiter=',', quotechar='|')
        rows=[r for r in poseReader]
    #print(rows[-1])
    row = rows[-2]
    print(row)
    Qw = float(row[-9]) 
    Qx = float(row[-8])    #roll
    Qy = float(row[-7])    #pitch
    Qz = float(row[-6]) 
    Tx = float(row[-5])/1000 
    Ty = float(row[-4])/1000 
    Tz = float(row[-3])/1000 
#    Euler[0:6] = np.squeeze([Rx, Ry, Rz, Tx, Ty, Tz]) 
#    Quater = euler2quatr(Euler[0,0:3]/57.29577951308232)
    Quater = np.squeeze([Tx, Ty, Tz, Qw, Qx, Qy, Qz])
    
    World = quatr2world(Quater)
#    World = quatr2world(np.concatenate([Euler[0,3:], Quater]))
    #print(world)
    for Inde_row in range(4):
        for Inde_col in range(4):
            Cam_pose_Mat.write(str(World[Inde_row, Inde_col]) + ' ')
        Cam_pose_Mat.write('\n')
    
    Cam_pose_Mat.close()
    cv2.imwrite('../Image/Image_'+str(Index+1).rjust(3,'0')+'.png', img)
    Flatten_world[Index,:] = World.reshape(16)
    Index +=1
x['handposes'] = Flatten_world
savemat("RobotPoses.mat", x)

#%% 3D stacked hist

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

Images = {}

img1 = cv2.imread('/home/yaqub/Downloads/bone.png' ,-1)
img2 = cv2.imread('/home/yaqub/Downloads/tissue.png' ,-1)

Images[0] = img1
Images[1] = img2



nbins = 50
for Image_Index in range(len(Images)):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    color_index = 0
    for c, z in zip(['r', 'g', 'b'], [ 20, 10, 0]):
    #    ys = np.random.normal(loc=10, scale=10, size=2000)
        ys = Images[Image_Index][:,:,color_index]
    
        hist, bins = np.histogram(ys, bins=nbins)
        xs = (bins[:-1] + bins[1:])/2
    
        ax.bar(xs, hist, zs=z, zdir='y', color=c, ec=c, alpha=0.8)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    

plt.show()


#%% rotating the coordinates to align the camera with the sensor and the refernce tracker
#                                                     z                                z
#                                                    /                                /
# for the tracker the original coordinate is  y_____/                                /_____x
#                                                   |                                |
#                                                   |             and for camera is  | 
#                                                   |                                |
#                                                   x                                 y
#                                               


import os               
import glob
import numpy as np
from matplotlib import pyplot as plt
#CurrentDir='/home/yaqub/Documents/Robotics/Videos/Stereo6/3DPrintedKnee/Good/21'
CurrentDir='/home/yaqub/Documents/Robotics/Videos/Stereo6/3DPrintedKnee/Water/Record5/2020-08-12-10-39-17_LR'
os.chdir(CurrentDir)
Pose_file =  open(glob.glob('*_pose.txt')[0], 'r')
Pose_Lines = Pose_file.readlines()
Pose_Quatr = np.zeros((len(Pose_Lines), 7), np.float32)
Pose_World = np.zeros((len(Pose_Lines), 4, 4), np.float32)
Pose_World_aligned = np.zeros((len(Pose_Lines), 4, 4), np.float32)
Pose_World_Stack = np.zeros(4, 4), np.float32)
for Index_Pose in range(len(Pose_Lines)-1):
    Index_Pose += 1
    Pose = Pose_Lines[Index_Pose].split(' ')
    Pose_Quatr[Index_Pose][0] = float(Pose[1])
    Pose_Quatr[Index_Pose][1] = float(Pose[2])
    Pose_Quatr[Index_Pose][2] = float(Pose[3])
    Pose_Quatr[Index_Pose][3] = float(Pose[4])
    Pose_Quatr[Index_Pose][4] = float(Pose[5])
    Pose_Quatr[Index_Pose][5] = float(Pose[6])            
    Pose_Quatr[Index_Pose][6] = float(Pose[7]) 
    Pose_World[Index_Pose, :, :] = quatr2world(Pose_Quatr[Index_Pose])
    
    Pose_World_Stack[:] = worldMirror(Pose_World_Stack.copy(), [0,0,1])    
    Pose_World_Stack[:] , _ = worldRotatZ(Pose_World_Stack.copy(), 26+90)
    
    FirstFrame_Eul = quatr2euler(world2quatr(Pose_World_Stack.copy()))*57.29578

    
    Pose_World_Stack[:], _ = worldRotatX(Pose_World_Stack.copy(), FirstFrame_Eul[0])
    Pose_World_Stack[:], _ = worldRotatY(Pose_World_Stack.copy(), FirstFrame_Eul[1])
    Pose_World_Stack[:], _ = worldRotatZ(Pose_World_Stack.copy(), FirstFrame_Eul[2])
    SecondFrame[:]
    
    
    
    
plt.plot(np.squeeze(Pose_World[:, 0:3, 3]))
plt.show()


World4  = np.zeros((4,4), np.float64)
World19 = np.zeros((4,4), np.float64)
World78 = np.zeros((4,4), np.float64)

World4[:] = [[-0.05949878098432004, -0.06784920767775998, 0.99591979139104, 0.08110099999999999],
       [0.9965296393385601, -0.06221578516930004, 0.055296694461980045, -0.06506],
       [0.05821016909984, 0.99575375044042, 0.07131559383101993, -0.23588900000000002],
       [0.0, 0.0, 0.0, 1.0]]

World19[:] = [[-0.0533493221647402, -0.07506949543326003, 0.99575015268642, 0.081149],
        [0.99857425751898, -0.005810255301700096, 0.05306261858074013, -0.06311800000000001],
        [0.0018022042200600552, 0.99716135109902, 0.07527246798140003, -0.236654],
        [0.0, 0.0, 0.0, 1.0]]

World78[:] = [[-0.05173075195236021, -0.07248116513303993, 0.9960273545013201, 0.080295],
        [0.99863056527892, -0.011554823740200115, 0.051025071873840144, -0.058631],
        [0.007810525247760047, 0.9973028850705201, 0.0729796029221999, -0.238758],
        [0.0, 0.0, 0.0, 1.0]] 




World4_M = worldMirror(World4.copy(), [0,0,1])    
World19_M = worldMirror(World19.copy(), [0,0,1])
World78_M = worldMirror(World78.copy(), [0,0,1])

World4_M_26 , _ = worldRotatZ(World4_M.copy(), 26)
World19_M_26, _ = worldRotatZ(World19_M.copy(), 26)
World78_M_26, _ = worldRotatZ(World78_M.copy(), 26)



World4_ref2cam, _ = worldRotatZ(World4_M_26.copy(), 90)
World19_ref2cam, _ = worldRotatZ(World19_M_26.copy(), 90)
World78_ref2cam, _ = worldRotatZ(World78_M_26.copy(), 90)

FirstFrame_Eul = quatr2euler(world2quatr(World4_ref2cam))*57.29578

SecondFrame = np.zeros((4,4), np.float64)
SecondFrame[:] = World4_ref2cam[:]
SecondFrame[:], _ = worldRotatX(SecondFrame.copy(), -FirstFrame_Eul[0])
SecondFrame[:]
SecondFrame[:], _ = worldRotatY(SecondFrame.copy(), -FirstFrame_Eul[1])
SecondFrame[:]
SecondFrame[:], _ = worldRotatZ(SecondFrame.copy(), -FirstFrame_Eul[2])
SecondFrame[:]

#%% convert NDI coordinate real time

os.system('rm -rf  Test.csv')
os.system('rm -rf  Test.txt')
os.system('rm -rf  Test.csv.offset')

FileIn= 'Test.csv'
FileOut='Test.txt'
#FileIn = args.fileName+'.csv'
#FileOut = args.fileName+'.txt'
FileW=open(FileIn, 'a'); 
FileW2=open(FileOut, 'w')
Frame = 0
Frame2 = 0

Pose_Quatr = np.zeros(7, np.float32)
Pose_World_Stack = np.zeros((4, 4), np.float32)


Frame_Total = 0
sys.stdout.write('Waiting for the NDI tracker to start the recording ... \n')
while (True):
    time.sleep(.001)  
    Frame = 0 
    for line in Pygtail(FileIn): 
        if (line[1:6] == ',Port') and (len(line)>151) and(len(line)<160):
            Frame+=1
            Frame2+=1
            if Frame2 == 5:
                Line = line.split(',')
                #print('Tx ' + Line[-5] + '    Ty ' + Line[-4] + '    Tz ' + Line[-3])
                Frame2 = 0
                Pose_Quatr[0] = float(Line[-5])
                Pose_Quatr[1] = float(Line[-4])
                Pose_Quatr[2] = float(Line[-3])
                Pose_Quatr[6] = float(Line[-6])
                Pose_Quatr[5] = float(Line[-7])
                Pose_Quatr[4] = float(Line[-8])
                Pose_Quatr[3] = float(Line[-9])
                print('Pose_Quatr[0:3]')
                print(Pose_Quatr[0:3])

                
                Pose_World_Stack[:] = quatr2world(Pose_Quatr[:])
                
                #Pose_World_Stack[:] = worldMirror(Pose_World_Stack.copy(), [0,0,1])    
                Pose_World_Stack[:] , _ = worldRotatZ(Pose_World_Stack.copy(), ( 90)/57.29578)
                
                FirstFrame_Eul = quatr2euler(world2quatr(Pose_World_Stack.copy()))
                print('FirstFrame_Eul:')
                print(np.round(FirstFrame_Eul*57.29)) 
                
#                Pose_World_Stack[:], _ = worldRotatX(Pose_World_Stack.copy(), FirstFrame_Eul[0])
#                Pose_World_Stack[:], _ = worldRotatY(Pose_World_Stack.copy(), FirstFrame_Eul[1])
#                Pose_World_Stack[:], _ = worldRotatZ(Pose_World_Stack.copy(), FirstFrame_Eul[2])
                print('Pose_World_Stack[0:3,3]:')
                print(Pose_World_Stack[0:3,3])
                print('############################################################3\n')


    timeStack = time.time()
    for i in range(Frame):
        #sys.stdout.write(line) 
        FileW2.write(':Frame '); 
        FileW2.write('  Time: '); 
        FileW2.write(str(timeStack-(Frame-i)*0.025)); 
        FileW2.write('\n'); 
        FileW2.flush();
        Frame_Total+=1
        #sys.stdout.write("\r" + 'Current frame: ' + str(Frame_Total) + '. Press Ctrl+C to terminate after recording.')
FileW.close();
FileW2.close();


# Read txt and manipulate it


#% 
from scipy import signal
import os               
import glob
import numpy as np
import math
from matplotlib import pyplot as plt
#CurrentDir='/home/yaqub/Documents/Robotics/Videos/Stereo6/3DPrintedKnee/Good/21'
CurrentDir='/home/yaqub/Documents/Robotics/Videos/Stereo6/Sheep/2020-10-08-13-36-36'
os.chdir(CurrentDir)
Prefix = 'PM13:36'
Pose_file_Q = open(Prefix+'_pose_Manual.txt')[0], 'r')
Pose_file_E = open(Prefix+'_Euler_pose_Manual.txt')[0], 'r')
Pose_Lines_Q = Pose_file_Q.readlines()
Pose_Lines_E = Pose_file_E.readlines()
Pose_Quatr = np.zeros((len(Pose_Lines_Q), 7), np.float64)
#Pose_World = np.zeros((len(Pose_Lines), 4, 4), np.float64)
Pose_Euler = np.zeros((len(Pose_Lines_E), 6), np.float64)
for Index_Pose in range(len(Pose_Lines_Q)-1):
    Index_Pose += 1
    Pose_Q = Pose_Lines[Index_Pose_Q].split(' ')
    Pose_Quatr[Index_Pose][0] = float(Pose[1])
    Pose_Quatr[Index_Pose][1] = float(Pose[2])
    Pose_Quatr[Index_Pose][2] = float(Pose[3])
    Pose_Quatr[Index_Pose][3] = float(Pose[4])
    Pose_Quatr[Index_Pose][4] = float(Pose[5])
    Pose_Quatr[Index_Pose][5] = float(Pose[6])            
    Pose_Quatr[Index_Pose][6] = float(Pose[7]) 
    
    Pose_E = Pose_Lines_E[Index_Pose].split(' ')
    Pose_Quatr[Index_Pose][0] = float(Pose[1])
    Pose_Quatr[Index_Pose][1] = float(Pose[2])
    Pose_Quatr[Index_Pose][2] = float(Pose[3])
    Pose_Quatr[Index_Pose][3] = float(Pose[4])
    Pose_Quatr[Index_Pose][4] = float(Pose[5])
    Pose_Quatr[Index_Pose][5] = float(Pose[6])            
    Pose_Quatr[Index_Pose][6] = float(Pose[7])
    
    
    Pose_Euler[Index_Pose,0:3] = Pose_Quatr[Index_Pose,0:3]
    Pose_Euler[Index_Pose,3:] = world2euler(np.squeeze(Pose_World[Index_Pose,:,:])) *57.296
Translation = np.zeros((len(Pose_Lines), 3), np.float64) 
Translation[:,0] = np.squeeze(Pose_World[:, 0:3, 3])[:,0] - np.squeeze(Pose_World[:, 0:3, 3])[100,0] #np.mean(np.squeeze(Pose_World[:, 0:3, 3])[:,0])
Translation[:,1] = np.squeeze(Pose_World[:, 0:3, 3])[:,1] - np.squeeze(Pose_World[:, 0:3, 3])[100,1] #np.mean(np.squeeze(Pose_World[:, 0:3, 3])[:,1])
Translation[:,2] = np.squeeze(Pose_World[:, 0:3, 3])[:,2] - np.squeeze(Pose_World[:, 0:3, 3])[100,2] #np.mean(np.squeeze(Pose_World[:, 0:3, 3])[:,2])



from scipy import signal
import os               
import glob
import numpy as np
import math
from matplotlib import pyplot as plt
#CurrentDir='/home/yaqub/Documents/Robotics/Videos/Stereo6/3DPrintedKnee/Good/21'
CurrentDir='/home/yaqub/Documents/Robotics/Videos/Stereo6/'
os.chdir(CurrentDir)
Prefix = '2020-09-18-14-03-36_sub'
#Pose_file_Q = open(Prefix+'.txt')[0], 'r')
Pose_file_E = open(Prefix+'.txt', 'r')
Pose_file_E_W = open(Prefix+'_filtered.txt', 'w')

#Pose_Lines_Q = Pose_file_Q.readlines()
Pose_Lines_E = Pose_file_E.readlines()
#Pose_Quatr = np.zeros((len(Pose_Lines_Q), 7), np.float64)
#Pose_World = np.zeros((len(Pose_Lines), 4, 4), np.float64)
Addresses = []
Images    = []
Pose_Euler = np.zeros((len(Pose_Lines_E), 6), np.float64)
for Index_Pose in range(len(Pose_Lines_E)):
    Index_Pose += 0  
    Pose_E = Pose_Lines_E[Index_Pose].split(' ')
    Image = Pose_E[0] 
    Pose_Euler[Index_Pose][0] = float(Pose_E[3])
    Pose_Euler[Index_Pose][1] = float(Pose_E[4])
    Pose_Euler[Index_Pose][2] = float(Pose_E[5])
    Pose_Euler[Index_Pose][3] = float(Pose_E[6])
    Pose_Euler[Index_Pose][4] = float(Pose_E[7])
    Pose_Euler[Index_Pose][5] = float(Pose_E[8])            
    Addresses.append(Pose_E[0])
    Images.append(Pose_E[1])
 
Pose_Euler_Orig = Pose_Euler.copy()
fs = 1000  # Sampling frequency
# Generate the time vector properly
t = np.arange(Index_Pose+1) / fs
fc = 50  # Cut-off frequency of the filter
w = fc / (fs / 2) # Normalize the frequency
b, a = signal.butter(5, w, 'low')
#output = signal.filtfilt(b, a, signalc)
Filterd = signal.filtfilt(b, a, Pose_Euler[:,3])     
Pose_Euler[:,3] = Filterd
Filterd = signal.filtfilt(b, a, Pose_Euler[:,4])     
Pose_Euler[:,4] = Filterd
Filterd = signal.filtfilt(b, a, Pose_Euler[:,5])     
Pose_Euler[:,5] = Filterd
    
for Index_Pose in range(Index_Pose+1):    
    
    Pose_file_E_W.write(Addresses[Index_Pose] + ' ' + Images[Index_Pose] + ' l ')    
    Pose_file_E_W.write(str(Pose_Euler[Index_Pose][0]) + ' ')    
    Pose_file_E_W.write(str(Pose_Euler[Index_Pose][1]) + ' ')    
    Pose_file_E_W.write(str(Pose_Euler[Index_Pose][2]) + ' ')    
    Pose_file_E_W.write(str(Pose_Euler[Index_Pose][3]) + ' ')    
    Pose_file_E_W.write(str(Pose_Euler[Index_Pose][4]) + ' ')    
    Pose_file_E_W.write(str(Pose_Euler[Index_Pose][5]) + '\n') 

Pose_file_E_W.close()


plt.plot(Pose_Euler[:,3:]*57) 
plt.plot(Pose_Euler_Orig[:,3:]*57) 
plt.legend()
plt.show()


# Jason reading
import json 
with open('opt.json') as f: 
    Data = jason.load(f)

#%% read coordinate filter them and plot them
    
from scipy import signal
import os               
import glob
import numpy as np
import math
from matplotlib import pyplot as plt
#CurrentDir='/home/yaqub/Documents/Robotics/Videos/Stereo6/3DPrintedKnee/Good/21'
#File='/run/user/1000/gvfs/smb-share:server=hpc-fs.qut.edu.au,share=jonmoham/Python/tsdf-fusion-python-master/CustomData/2020-09-18-14-03-36/pose/2020-09-18-14-03-36_sub_filt20.txt'    
File='/home/yaqub/Downloads/Temp/train_files.txt'
Pose_file_E = open(File, 'r')
Pose_file_E_W = open('filtered2.txt', 'w')
Pose_Lines_E = Pose_file_E.readlines()
Pose_quatr = np.zeros((len(Pose_Lines_E), 7), np.float64)
for Index_Pose in range(len(Pose_Lines_E)):  
    Index_Pose += 0  
    Pose_E = Pose_Lines_E[Index_Pose].split(' ')
    Image = Pose_E[0] 
    Pose_quatr[Index_Pose][0] = float(Pose_E[3]) #- Pose_quatr[0][0]
    Pose_quatr[Index_Pose][1] = float(Pose_E[4]) #- Pose_quatr[0][1]
    Pose_quatr[Index_Pose][2] = float(Pose_E[5]) #- Pose_quatr[0][2]
    Pose_quatr[Index_Pose][3] = float(Pose_E[6]) #- Pose_quatr[0][3]
    Pose_quatr[Index_Pose][4] = float(Pose_E[7]) #- Pose_quatr[0][4]
    Pose_quatr[Index_Pose][5] = float(Pose_E[8]) #- Pose_quatr[0][5]            
    Pose_quatr[Index_Pose][6] = float(Pose_E[9][:-2]) #- Pose_quatr[0][6] 

Pose_quatr2 = Pose_quatr.copy()


fs = 1000  # Sampling frequency
# Generate the time vector properly
t = np.arange(Index_Pose+1) / fs
fc = 100  # Cut-off frequency of the filter
w = fc / (fs / 2) # Normalize the frequency
b, a = signal.butter(5, w, 'low')
#output = signal.filtfilt(b, a, signalc)
Filterd = signal.filtfilt(b, a, Pose_quatr[:,3])     
Pose_quatr[:,3] = Filterd
Filterd = signal.filtfilt(b, a, Pose_quatr[:,4])     
Pose_quatr[:,4] = Filterd
Filterd = signal.filtfilt(b, a, Pose_quatr[:,5])     
Pose_quatr[:,5] = Filterd
Filterd = signal.filtfilt(b, a, Pose_quatr[:,6])     
Pose_quatr[:,6] = Filterd


    
    
for Index_Pose in range(len(Pose_Lines_E)):  
    Pose_E = Pose_Lines_E[Index_Pose].split(' ')
    Image = Pose_E[0] 
    
    World_stack, _ = worldRotatZ(quatr2world(Pose_quatr[Index_Pose][0:7].copy()), (-84)/57.29578)
    Pose_quatr[Index_Pose][3:7] = world2quatr(World_stack.copy()) 
    Pose_quatr[Index_Pose][0:3] = World_stack[0:3,3]
    Pose_file_E_W.write(Pose_E[0] + ' ' + Pose_E[1] + ' l ')    
    Pose_file_E_W.write(str("%03f" %Pose_quatr[Index_Pose][0]) + ' ')    
    Pose_file_E_W.write(str("%03f" %Pose_quatr[Index_Pose][1]) + ' ')    
    Pose_file_E_W.write(str("%03f" %Pose_quatr[Index_Pose][2]) + ' ')    
    Pose_file_E_W.write(str("%06f" %Pose_quatr[Index_Pose][3]) + ' ')    
    Pose_file_E_W.write(str("%06f" %Pose_quatr[Index_Pose][4]) + ' ')    
    Pose_file_E_W.write(str("%06f" %Pose_quatr[Index_Pose][5]) + ' ') 
    Pose_file_E_W.write(str("%06f" %Pose_quatr[Index_Pose][6]) + '\n') 
    
Pose_file_E_W.close()

#plt.plot(Pose_quatr[1:-1,3:])
#plt.legend(['X', 'Y', 'Z']) 
#plt.show()





plt.plot(Pose_quatr[1:-1,3:])
plt.plot(Pose_quatr2[1:-1,3:])
#plt.legend(['X', 'Y', 'Z']) 
plt.show()




File='/home/yaqub/Downloads/Temp/VALVAL.txt'
Pose_file_E = open(File, 'r')
Pose_file_E_W = open('filteredEulVal.txt', 'w')
Pose_Lines_E = Pose_file_E.readlines()
Pose_quatr = np.zeros((len(Pose_Lines_E), 7), np.float64)
Pose_Euler = np.zeros((len(Pose_Lines_E), 3), np.float64)
for Index_Pose in range(len(Pose_Lines_E)):  
    Index_Pose += 0  
    Pose_E = Pose_Lines_E[Index_Pose].split(' ')
    Image = Pose_E[0] 
    Pose_quatr[Index_Pose][0] = float(Pose_E[3]) #- Pose_quatr[0][0]
    Pose_quatr[Index_Pose][1] = float(Pose_E[4]) #- Pose_quatr[0][1]
    Pose_quatr[Index_Pose][2] = float(Pose_E[5]) #- Pose_quatr[0][2]
    Pose_quatr[Index_Pose][3] = float(Pose_E[6]) #- Pose_quatr[0][3]
    Pose_quatr[Index_Pose][4] = float(Pose_E[7]) #- Pose_quatr[0][4]
    Pose_quatr[Index_Pose][5] = float(Pose_E[8]) #- Pose_quatr[0][5]            
    Pose_quatr[Index_Pose][6] = float(Pose_E[9][:-2]) #- Pose_quatr[0][6] 
    
    Pose_Euler[Index_Pose] = world2euler(quatr2world(Pose_quatr[Index_Pose][0:7].copy()))

    Pose_file_E_W.write(Pose_E[0] + ' ' + Pose_E[1] + ' l ')    
    Pose_file_E_W.write(str("%03f" %Pose_quatr[Index_Pose][0]) + ' ')    
    Pose_file_E_W.write(str("%03f" %Pose_quatr[Index_Pose][1]) + ' ')    
    Pose_file_E_W.write(str("%03f" %Pose_quatr[Index_Pose][2]) + ' ')    
    Pose_file_E_W.write(str("%03f" %Pose_Euler[Index_Pose][0]) + ' ')    
    Pose_file_E_W.write(str("%03f" %Pose_Euler[Index_Pose][1]) + ' ')       
    Pose_file_E_W.write(str("%03f" %Pose_Euler[Index_Pose][2]) + '\n') 

Pose_file_E_W.close()





File='/run/user/1000/gvfs/smb-share:server=hpc-fs.qut.edu.au,share=jonmoham/Python/monodepth2-master/splits/3DPrint_Sheep_quat_NDI_filt/val_files.txt'
Pose_file_E = open(File, 'r')
Pose_Lines_E = Pose_file_E.readlines()
Pose_quatr = np.zeros((len(Pose_Lines_E), 7), np.float64)
for Index_Pose in range(len(Pose_Lines_E)):  
    Index_Pose += 0  
    Pose_E = Pose_Lines_E[Index_Pose].split(' ')
    Image = Pose_E[0] 
    Pose_quatr[Index_Pose][0] = float(Pose_E[3]) - Pose_quatr[0][0]
    Pose_quatr[Index_Pose][1] = float(Pose_E[4]) - Pose_quatr[0][1]
    Pose_quatr[Index_Pose][2] = float(Pose_E[5]) - Pose_quatr[0][2]
    Pose_quatr[Index_Pose][3] = float(Pose_E[6]) - Pose_quatr[0][3]
    Pose_quatr[Index_Pose][4] = float(Pose_E[7]) - Pose_quatr[0][4]
    Pose_quatr[Index_Pose][5] = float(Pose_E[8]) - Pose_quatr[0][5]            
    Pose_quatr[Index_Pose][6] = float(Pose_E[9][:-2]) - Pose_quatr[0][5] 
fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(np.squeeze(Pose_quatr[1:, 0:1])) 
ax1.plot(np.squeeze(Pose_quatr[1:, 1:2])) 
ax1.plot(np.squeeze(Pose_quatr[1:, 2:3]))  
ax1.legend(['X', 'Y', 'Z']) 
ax2.plot(np.squeeze(Pose_quatr[1:, 3:]))
plt.show()

#%%

Networks=[
'3Dprint_waterPose_Mono0-4-224_pairs_Bch20_Lr2-4_Full_OK_GoodDepth_25/mdp/models/weights_25/',
'3DPrint_Sheep_NDI_filt_02_Bch12_Lr5-5_2Loss_pairs_UsePose_Smoo.02_NymLay50_Pre/mdp/models/weights_02/',
'3DPrint_Sheep_NDI_filt_02_Bch12_Lr5-5_2Loss_pairs_UsePose_Smoo.02_NymLay50_Pre_07/mdp/models/weights_07/',
'3DPrint_Sheep_NDI_filt_02_Bch14_Lr5-5_2Loss_pairs_NoPose_Smoo.02_NymLay50/mdp/models/weights_59/',
'Cadava5_3DPrint_Sheep_filt0-2-112_Bch8_Lr5-5_pairs_UsePoseSup_NymLay50_MinDep.05Max200/mdp/models/weights_39/',
'Cadava5_3DPrint_Sheep_filt0-2-112_Bch10_Lr5-5_pairs_NoPoseSup_NymLay50_MinDep.05Max200/mdp/models/weights_32/',
'Cadava5_3DPrint_Sheep_filt_024_Bch14_Lr5-5_2Loss_pairs_NoPose_Smoo.01_NymLay50/mdp/models/weights_59/',
'Cadava5_3DPrint_Sheep_filt_024_Bch14_Lr5-5_2Loss_pairs_NoPose_Smoo.005_NymLay50/mdp/models/weights_59/',
'3DPrint_Sheep_NDI_filt_02_Bch14_Lr5-5_2Loss_pairs_UsePose_Smoo.005_NymLay50/mdp/models/weights_59/',
'Cadava5_3DPrint_Sheep_filt_02_Bch10_Lr5-5_2Loss_pairs_NoPose_Smoo.03_NymLay50/mdp/models/weights_59/',
'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch18_Lr5-5_2Loss_all_AngTranLoss_NymLay50_Pre/mdp/models/weights_19/',
'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch10_Lr5-5_all_NoPoseSup_NymLay50_MinDep.05Max200/mdp/models/weights_11/',
'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch8_Lr5-5_all_NoPoseSup_NymLay50_MinDep.05Max200/mdp/models/weights_14/',
'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch8_Lr5-5_pairs_PoseSup_Mask_NymLay50_Pre/mdp/models/weights_03/',
'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch4_Lr5-5_2Loss_all_AngTranLoss_NymLay50_Pre/mdp/models/weights_11/',
'Cadava5_3DPrint_Sheep_filt0-2-112_Bch10_Lr5-5_pairs_NoPoseSup_NymLay50_MinDep.05Max200/mdp/models/weights_32/',
'Cadava5_3DPrint_Sheep_filt0-2-112_Bch10_Lr5-5_all_NoPoseSup_NymLay50_MinDep.05Max200/mdp/models/weights_37/',
'Cadava5_3DPrint_Sheep_filt0-2-112_Bch8_Lr5-5_pairs_UsePoseSup_NymLay50_MinDep.05Max200_Pre/mdp/models/weights_35/',
'3Dprint_waterPose_Mono0-3-2-1123_pairs_Bch20_Lr3-5_Full_OKDepth/mdp/models/weights_30/',
'3Dprint_waterPose_Mono0-4-224_pairs_Bch20_Lr2-4_Full_OK_GoodDepth/mdp/models/weights_35/',
'3Dprint_waterPose_Mono03_pairs_Bch5_Lr5-5_Good/mdp/models/weights_50/',
'3Dprint_water_SheeSub0-5-4-3-2-112345_Bch22_Lr1-4_2Loss_all_NoPoseLoss2/mdp/models/weights_37/',
'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch18_Lr1-4_2Loss_all_NoPoseLoss/mdp/models/weights_13/',
'Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch18_Lr1-4_2Loss_all_AngTranLoss_Pre/mdp/models/weights_12/'
]
Networks=['3DPrint_Sheep_NDI_filt_02_Bch14_Lr5-5_2Loss_pairs_NoPose_Smoo.02_NymLay50_5/mdp/models/weights_35/']
for Network_Ind in range(len(Networks)):
    print(Networks[Network_Ind])
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4)
    depth1 = cv2.imread('/home/yaqub/Downloads/Temp/2019-10-24-13-02-46_LR/'+Networks[Network_Ind][:-23]+'/00144_depth.jpeg', -1)
    depth2 = cv2.imread('/home/yaqub/Downloads/Temp/2019-10-24-13-02-46_LR/'+Networks[Network_Ind][:-23]+'/01975_depth.jpeg', -1)
    depth3 = cv2.imread('/home/yaqub/Downloads/Temp/2019-10-24-13-02-46_LR/'+Networks[Network_Ind][:-23]+'/03193_depth.jpeg', -1)
    depth4 = cv2.imread('/home/yaqub/Downloads/Temp/2019-10-24-13-02-46_LR/'+Networks[Network_Ind][:-23]+'/03400_depth.jpeg', -1)
    ax1.imshow(depth1)
    ax2.imshow(depth2)
    ax3.imshow(depth3)
    ax4.imshow(depth4)
    fig.suptitle(Networks[Network_Ind][:-23])


depth1 = cv2.imread('/home/yaqub/Downloads/Temp/2019-10-24-13-02-46_LR/image_02/data/00144.png', -1)
depth2 = cv2.imread('/home/yaqub/Downloads/Temp/2019-10-24-13-02-46_LR/image_02/data/01975.png', -1)
depth3 = cv2.imread('/home/yaqub/Downloads/Temp/2019-10-24-13-02-46_LR/image_02/data/03193.png', -1)
depth4 = cv2.imread('/home/yaqub/Downloads/Temp/2019-10-24-13-02-46_LR/image_02/data/03400.png', -1)
ax1.imshow(depth1)
ax2.imshow(depth2)
ax3.imshow(depth3)
ax4.imshow(depth4)    

plt.show()


'/3DPrint_Sheep_NDI_filt_02_Bch12_Lr5-5_2Loss_pairs_NoPose_Smoo.005_NymLay50/',
'/3DPrint_Sheep_NDI_filt_02_Bch12_Lr5-5_2Loss_pairs_UsePose_Smoo.02_NymLay50_Pre/',
'/3DPrint_Sheep_NDI_filt_02_Bch14_Lr5-5_2Loss_pairs_UsePose_Smoo.005_NymLay50/',
'/Cadava5_3DPrint_Sheep_filt0-2-112_Bch8_Lr5-5_pairs_UsePoseSup_NymLay50_MinDep.05Max200/'

'/Cadava5_3DPrint_Sheep_filt0-2-112_Bch8_Lr5-5_pairs_UsePoseSup_NymLay50_MinDep.05Max200_Pre
'/Cadava5_3DPrint_Sheep_filt0-2-112_Bch10_Lr5-5_all_NoPoseSup_NymLay50_MinDep.05Max200
'/Cadava5_3DPrint_Sheep_filt0-2-112_Bch10_Lr5-5_pairs_NoPoseSup_NymLay50_MinDep.05Max200
'/Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch4_Lr5-5_2Loss_all_AngTranLoss_NymLay50_Pre
/Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch8_Lr5-5_all_NoPoseSup_NymLay50_MinDep.05Max200
/Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch8_Lr5-5_pairs_PoseSup_Mask_NymLay50_Pre
/Cadava5_3DPrint_Sheep_filt0-5-4-3-2-112345_Bch10_Lr5-5_all_NoPoseSup_NymLay50_MinDep.05Max200




import numpy as np
import glob
import cv2
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

stereoframe = np.zeros((384,384*2, 3), np.uint8)
#%%    
VideoRL = cv2.VideoWriter('Art_Depth.mp4', fourcc, 30.0, (384*2, 384))

CurrentDir='/home/yaqub/Downloads/Temp/Depth/'
os.chdir(CurrentDir) 
images= sorted(glob.glob("image_02/data/*.png")) 
for i in (images): 
    Img = cv2.imread(i,-1) 
    #resized = cv2.resize(Img, dim, interpolation = cv2.INTER_AREA) 
    #cv2.imwrite(str(i).rjust(5,'0')+'.png', Img) 
    stereoframe[:,:384,:] = Img[:,:,:]
    Img = cv2.imread('image_03' + i[8:],-1) 
    stereoframe[:,384:,:] = Img[:,:,:]
    VideoRL.write(stereoframe)      
VideoRL.release()

imagesL = sorted(glob.glob("rgb/*.png"))
for i in range(0, len(imagesL),4):
    stereoframe[:,:,:] = cv2.imread(imagesL[i])
    stereoframe2[:,:,2] = stereoframe[:,:,0]
    stereoframe2[:,:,1] = stereoframe[:,:,1]
    stereoframe2[:,:,0] = stereoframe[:,:,2]
    VideoRL.write(stereoframe2[:,:,:])
    #cv2.imwrite(imagesL[i],stereoframe2[:,:,:])
    #cv2.imshow('Orig', imgBGR)
VideoRL.release()









def euler2world(theta) :
    #https://www.learnopencv.com/rotation-matrix-to-euler-angles/
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]], np.float64)                  
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]], np.float64)              
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]], np.float64)                                 
    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R








from scipy import signal
import os               
import glob
import numpy as np
import math
from matplotlib import pyplot as plt
#CurrentDir='/home/yaqub/Documents/Robotics/Videos/Stereo6/3DPrintedKnee/Good/21'
#File='/run/user/1000/gvfs/smb-share:server=hpc-fs.qut.edu.au,share=jonmoham/Python/tsdf-fusion-python-master/CustomData/2020-09-18-14-03-36/pose/2020-09-18-14-03-36_sub_filt20.txt'    
File='/home/joon/Downloads/Tem/AM1403_Euler_driftCorrec_NameAndPose_invert.txt'
Pose_file_E = open(File, 'r')
#Pose_file_E_W = open('filtered2.txt', 'w')
Pose_Lines_E = Pose_file_E.readlines()
#Pose_E = np.zeros((len(Pose_Lines_E), 6), np.float64)
Pose_E_Orig = np.zeros((len(Pose_Lines_E), 6), np.float64)
Pose_E_2    = np.zeros((len(Pose_Lines_E), 6), np.float64)
Indexes     = np.zeros((len(Pose_Lines_E), 1), np.uint16)
Pose_Q      = np.zeros((1, 7), np.float64)
Pose_Q_2    = np.zeros((1, 4), np.float64)
Pose_W      = np.zeros((4, 4), np.float64)
Start = 3
for Index_Pose in range(len(Pose_Lines_E)):  
    Pose_E_temp = Pose_Lines_E[Index_Pose].split(' ')
    #Indexes[Index_Pose] = int(Pose_E_temp[0][1:5])
    Indexes[Index_Pose] = int(Pose_E_temp[1][1:5])
    Pose_E_Orig[Index_Pose][0] = float(Pose_E_temp[Start]) #- Pose_quatr[0][0]
    Pose_E_Orig[Index_Pose][1] = float(Pose_E_temp[Start+1]) #- Pose_quatr[0][1]
    Pose_E_Orig[Index_Pose][2] = float(Pose_E_temp[Start+2]) #- Pose_quatr[0][2]
    Pose_E_Orig[Index_Pose][3] = float(Pose_E_temp[Start+3]) #- Pose_quatr[0][3]
    Pose_E_Orig[Index_Pose][4] = float(Pose_E_temp[Start+4]) #- Pose_quatr[0][4]
    Pose_E_Orig[Index_Pose][5] = float(Pose_E_temp[Start+5]) #- Pose_quatr[0][5]            

    Pose_Q[0,0] = float(Pose_E_Orig[Index_Pose][0])
    Pose_Q[0,1] = float(Pose_E_Orig[Index_Pose][1])
    Pose_Q[0,2] = float(Pose_E_Orig[Index_Pose][2])
   
    Pose_Q[0,3:7]  = euler2quatr(Pose_E_Orig[Index_Pose][3:6]/1)
    
    Pose_Q_2 = quatMirror(Pose_Q, [0,1,0])
    #Pose_E_2[Index_Pose,3:6]= quatr2euler(Pose_Q_2[3:])       
    #Pose_E_2[Index_Pose,0:3]= Pose_Q_2[0:3]    
    Pose_W, _ = worldRotatZ(quatr2world(Pose_Q_2[0:7].copy()), (90)/57.29578)
    Pose_Q_2= world2quatr(Pose_W.copy()) 	
    Pose_E_2[Index_Pose,3:]  = quatr2euler(Pose_Q_2)
    Pose_E_2[Index_Pose,0:3] = Pose_W[0:3,3]
    
#plt.plot(Indexes, Pose_E_Orig[:,3:] - np.mean(Pose_E_Orig[2000:2001,3:],0))
plt.plot(Indexes[:], Pose_E_Orig[:,:3] - np.mean(Pose_E_Orig[0:2,:3],0))
plt.show()


Drift = 8
sigLength = len(Pose_E_Orig[:,0])
Pad = np.zeros((Drift,6), np.float32)
Padded = np.concatenate((Pad, Pose_E_Orig), axis=0)
totalLength = sigLength - int(0.00083*sigLength) + Drift 
Padded_stretched = signal.resample(Padded, totalLength)


#plt.plot(Indexes[:], Padded_stretched[0:sigLength,:])
plt.plot(Indexes[:], Padded_stretched[3:,:3] - np.mean(Padded_stretched[Drift:Drift+2,:3],0))
plt.show()

Padded_stretched = Padded_stretched[3:,:]



Pose_file_E_W = open('AM1338_Euler_driftCorrec_NameAndPose.txt', 'w')
for Index_Pose in range(len(Padded_stretched[:-1,:])):  
    Index_Pose += 0  

    #Pose_file_E_W.write(str(Index_Pose).rjust(5,'0') +  ' l ')
    Pose_file_E_W.write(str(Indexes[Index_Pose][0]).rjust(5,'0') +  ' l ')    
    Pose_file_E_W.write(str("%03f" %Padded_stretched[Index_Pose][0]) + ' ')    
    Pose_file_E_W.write(str("%03f" %Padded_stretched[Index_Pose][1]) + ' ')    
    Pose_file_E_W.write(str("%03f" %Padded_stretched[Index_Pose][2]) + ' ')    
    Pose_file_E_W.write(str("%03f" %Padded_stretched[Index_Pose][3]) + ' ')    
    Pose_file_E_W.write(str("%03f" %Padded_stretched[Index_Pose][4]) + ' ')       
    Pose_file_E_W.write(str("%03f" %Padded_stretched[Index_Pose][5]) + '\n') 

Pose_file_E_W.close()







Pose_File='/home/joon/Documents/Code/Python/MonoDepth2Enodo/WithMag_Inv_MulSclFul_Pre/mdp/models/weights_39/poses_2020-09-18-14-03-36.npy'
Poses_3D = np.load(Pose_File) 
 
Poses_3D = Poses_3D.astype(np.float64) 
#Poses_3D_GLB = Poses_3D.copy() 
No_ref_fram = 1 
FrameIndex = 1 #  
Poses_3D_GLB = np.zeros((int(np.shape(Poses_3D)[0]/No_ref_fram), 4, 4), np.float64) 
Poses_3D_GLB[0,:,:] = np.eye(4) 
Quat_GLB = np.zeros((int(np.shape(Poses_3D)[0]/No_ref_fram), 1, 4), np.float64) 
Euler_GLB = np.zeros((int(np.shape(Poses_3D)[0]/No_ref_fram), 1, 3), np.float64) 
ii = 1 
for i in range( FrameIndex , len(Poses_3D)-No_ref_fram, No_ref_fram): 
    Poses_3D_GLB[ii,:,:] =  np.dot(Poses_3D[i-1,:,:],  Poses_3D_GLB[ii-1,:,:]) 
    Quat_GLB[ii]= world2quatr(np.squeeze(Poses_3D_GLB[ii,:,:])).astype(np.float64) 
    Euler_GLB[ii]=world2euler(np.squeeze(Poses_3D_GLB[ii,:,:])).astype(np.float64) 
    ii += 1 
    #print (i) 
#plt.plot(np.squeeze(Poses_3D_GLB[1:, 0:3, 3])) 
trans_weight = 1/np.array([1., 1., 1.])
fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(np.squeeze(Poses_3D_GLB[1:, 0:1, 3])*trans_weight[0]) 
ax1.plot(np.squeeze(Poses_3D_GLB[1:, 1:2, 3])*trans_weight[1]) 
ax1.plot(np.squeeze(Poses_3D_GLB[1:, 2:3, 3])*trans_weight[2])  
ax1.legend(['X', 'Y', 'Z']) 
ax2.plot(np.squeeze(Euler_GLB[:-1,:]))
plt.show()





plt.plot(np.squeeze(Poses_3D_GLB[1:, 0:3, 3]) - np.mean(np.squeeze(Poses_3D_GLB[1935:1937, 0:3, 3]),0))
plt.show()







    Pose_W              = quatr2world(Pose_Q[0,:].copy())	

    Pose_W = worldMirror(Pose_W.copy(), [0,1,0])
    #Pose_E_2[Index_Pose] = world2euler(Pose_W.copy())
    Pose_Q_2= world2quatr(Pose_W.copy()) 	
    Pose_E_2[Index_Pose]= quatr2euler(Pose_Q_2)
	

Pose_quatr2 = Pose_quatr.copy()



quatr2euler(euler2quatr(Pose_E_Orig[Index_Pose][3:6]/57.29578))
world2quatr(quatr2world(Pose_Q[0,:].copy()))
    
    
quatr2euler(world2quatr(quatr2world(euler2quatr(Pose_E_Orig[Index_Pose][3:6]/57.29578))))

for Index_Pose in range(len(Pose_Lines_E)):  
    Pose_E = Pose_Lines_E[Index_Pose].split(' ')
    Image = Pose_E[0] 
    
    World_stack, _ = worldRotatZ(quatr2world(Pose_quatr[Index_Pose][0:7].copy()), (-84)/57.29578)
    Pose_quatr[Index_Pose][3:7] = world2quatr(World_stack.copy()) 
    Pose_quatr[Index_Pose][0:3] = World_stack[0:3,3]
    Pose_file_E_W.write(Pose_E[0] + ' ' + Pose_E[1] + ' l ')    
    Pose_file_E_W.write(str("%03f" %Pose_quatr[Index_Pose][0]) + ' ')    
    Pose_file_E_W.write(str("%03f" %Pose_quatr[Index_Pose][1]) + ' ')    
    Pose_file_E_W.write(str("%03f" %Pose_quatr[Index_Pose][2]) + ' ')    
    Pose_file_E_W.write(str("%06f" %Pose_quatr[Index_Pose][3]) + ' ')    
    Pose_file_E_W.write(str("%06f" %Pose_quatr[Index_Pose][4]) + ' ')    
    Pose_file_E_W.write(str("%06f" %Pose_quatr[Index_Pose][5]) + ' ') 
    Pose_file_E_W.write(str("%06f" %Pose_quatr[Index_Pose][6]) + '\n') 
    
Pose_file_E_W.close()

#plt.plot(Pose_quatr[1:-1,3:])
#plt.legend(['X', 'Y', 'Z']) 
#plt.show()





plt.plot(Pose_quatr[1:-1,3:])
plt.plot(Pose_quatr2[1:-1,3:])
#plt.legend(['X', 'Y', 'Z']) 
plt.show()





Split=[
'poses_2020-10-08-13-36-36.npy',
'poses_2020-09-18-14-03-36_sub_filt.npy',
'poses_2019-10-24-13-01-04_LR.npy',
'poses_2019-10-24-13-24-37_LR.npy',
'poses_2019-10-24-13-20-47_LR.npy',
'poses_2019-10-24-13-18-50_LR.npy',
'poses_2019-10-24-13-11-26_LR.npy',
'poses_2019-10-24-13-09-02_LR.npy',
'poses_2019-10-24-13-06-57_LR.npy',
'poses_2019-10-24-13-05-43_LR.npy',
'poses_2019-10-24-13-01-41_LR.npy',
'poses_2019-10-24-13-02-46_LR.npy',
'poses_Artur_SubTotal.npy',
]


Sp_Ind = -1
Source='/run/user/1000/gvfs/smb-share:server=hpc-fs.qut.edu.au,share=jonmoham/Python/monodepth2-master/'
#Pose_File= Source+'3Dprint_water_SheepfiltNDI_02_Bch14_Lr5-5_2Loss_pairs_PoseOnly_NymLay50/mdp/models/weights_29/poses_Artur_SubTotal.npy'#+Split[1]
Pose_File= Source+'3Dprint_water_SheepfiltNDI_02_Bch14_Lr5-5_2Loss_pairs_PoseOnly_NymLay50/mdp/models/weights_29/'+Split[Sp_Ind]
Poses_3D = np.load(Pose_File) 
 
Poses_3D = Poses_3D.astype(np.float64) 
#Poses_3D_GLB = Poses_3D.copy() 
No_ref_fram = 1 
FrameIndex = 1 # 
Drifts = np.zeros((4,4), np.float64) 
Drifts[0:3, 3] = [-0.00003 , 0.0000, -0.00032]
Poses_3D_GLB = np.zeros((int(np.shape(Poses_3D)[0]/No_ref_fram), 4, 4), np.float64) 
Poses_3D_GLB[0,:,:] = np.eye(4) 
Quat_GLB = np.zeros((int(np.shape(Poses_3D)[0]/No_ref_fram), 1, 4), np.float64) 
Euler_GLB = np.zeros((int(np.shape(Poses_3D)[0]/No_ref_fram), 1, 3), np.float64) 
ii = 1 
for i in range( FrameIndex , len(Poses_3D)-No_ref_fram, No_ref_fram): 
    Poses_3D_GLB[ii,:,:] =  np.dot(Poses_3D[i-1,:,:],  Poses_3D_GLB[ii-1,:,:]) + Drifts
    Quat_GLB[ii]= world2quatr(np.squeeze(Poses_3D_GLB[ii,:,:])).astype(np.float64) 
    Euler_GLB[ii]=world2euler(np.squeeze(Poses_3D_GLB[ii,:,:])).astype(np.float64) 
    ii += 1 
    #print (i) 
#plt.plot(np.squeeze(Poses_3D_GLB[1:, 0:3, 3])) 
Sign = -1  
if FrameIndex < No_ref_fram/2: 
    Sign = 1 

trans_weight = 1/np.array([.5, .5, 2.])
Coeff = np.array([60, 60, 75])*Sign
#trans_weight = 1/np.array([1, 1, 1.])
fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(np.squeeze(Poses_3D_GLB[1:, 0:1, 3])*Coeff[0]*Sign*trans_weight[0] + 0) 
ax1.plot(np.squeeze(Poses_3D_GLB[1:, 1:2, 3])*Coeff[1]*Sign*trans_weight[1] - 0) 
ax1.plot(np.squeeze(Poses_3D_GLB[1:, 2:3, 3])*Coeff[2]*Sign*trans_weight[2] + 0)  
ax1.legend(['X', 'Y', 'Z']) 
ax2.plot(np.squeeze(Euler_GLB[:-1,:])*Sign*57.25)
plt.show()



#%% counter rotate the monodepth2 to match the NDI coordidate
Poses_3D_GLB2 = Poses_3D_GLB.copy()  
#%%

#trans_weight = 1/np.array([.6, .5, 1.])
#trans_weight = 1/np.array([1, 1, 1.])

Poses_3D_GLB[:] = Poses_3D_GLB2.copy() 
for i in range(len(Poses_3D_GLB)): 
#    if Sign== -1:
#        Poses_3D_GLB[i,:,:] = worldMirror(Poses_3D_GLB[i,:,:].copy(), [0,0,1])
#    else:
    Poses_3D_GLB[i,0:3,3] = Poses_3D_GLB[i,0:3,3]#* trans_weight
    #World_stack, _ = worldRotatZ(Poses_3D_GLB[i,:,:].copy(), (-84)/57.29578)
    #Poses_3D_GLB[i,:,:] = World_stack.copy()
    Quat_GLB[i]= world2quatr(np.squeeze(Poses_3D_GLB[i,:,:])).astype(np.float64) 
    Euler_GLB[i]=world2euler(np.squeeze(Poses_3D_GLB[i,:,:])).astype(np.float64) 
#Sign = -1  
#if FrameIndex < No_ref_fram/2: 
#    Sign = 1 
    
Poses_3D_GLB[1:, 0:1, 3] =Poses_3D_GLB[1:, 0:1, 3]*Coeff[0]*Sign*trans_weight[0]
Poses_3D_GLB[1:, 1:2, 3] =Poses_3D_GLB[1:, 1:2, 3]*Coeff[1]*Sign*trans_weight[1]
Poses_3D_GLB[1:, 2:3, 3] =Poses_3D_GLB[1:, 2:3, 3]*Coeff[2]*Sign*trans_weight[2]
Quat_GLB[:] = Quat_GLB[:]*Sign
Euler_GLB[:] = Euler_GLB[:]*Sign

fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(np.squeeze(Poses_3D_GLB[1:, 0:1, 3])) 
ax1.plot(np.squeeze(Poses_3D_GLB[1:, 1:2, 3])) 
ax1.plot(np.squeeze(Poses_3D_GLB[1:, 2:3, 3]))  
ax1.legend(['X', 'Y', 'Z']) 
ax2.plot(np.squeeze(Euler_GLB[:-1,:])*1)
plt.show()




Split2=[
'2020-10-08-13-36-36',
'2020-09-18-14-03-36_sub_filt',
'2019-10-24-13-01-04_LR',
'2019-10-24-13-24-37_LR',
'2019-10-24-13-20-47_LR',
'2019-10-24-13-18-50_LR',
'2019-10-24-13-11-26_LR',
'2019-10-24-13-09-02_LR',
'2019-10-24-13-06-57_LR',
'2019-10-24-13-05-43_LR',
'2019-10-24-13-01-41_LR',
'2019-10-24-13-02-46_LR',
'Artur_SubTotal',
]

File='/run/user/1000/gvfs/smb-share:server=hpc-fs.qut.edu.au,share=jonmoham/Python/monodepth2-master/splits/'+Split2[Sp_Ind]+'/'+Split2[Sp_Ind]+'.txt' 
Pose_file_E = open(File, 'r')
Pose_Lines_E = Pose_file_E.readlines()
Image = np.zeros((len(Pose_Lines_E), 7), np.float64)
Pose_file_E.close()


File_Quatr = open(Pose_File[:-3] + 'txt', 'w')
#start_image = 3300
Pose_File = '/home/yaqub/Downloads/Temp/Mono_CadaCoord/Mono_02_' + Split2[Sp_Ind] + '.txt'
File_Quatr = open(Pose_File , 'w')
for i in range(len(Poses_3D_GLB)):
    quaternion = world2quatr(np.squeeze(Poses_3D_GLB[i,:,:])).astype(np.float64)
    #File_Quatr.write(str(i+start_image).rjust(5,'0') + '.png ')
    File_Quatr.write(Pose_Lines_E[i].split(' ')[0] +'/'+Pose_Lines_E[i].split(' ')[1]+'.png ')
    File_Quatr.write(str("%04f" %Poses_3D_GLB[i,0,3]) + ' ')
    File_Quatr.write(str("%04f" %Poses_3D_GLB[i,1,3]) + ' ')
    File_Quatr.write(str("%04f" %Poses_3D_GLB[i,2,3]) + ' ')
    File_Quatr.write(str("%04f" %quaternion[0]) + ' ')
    File_Quatr.write(str("%04f" %quaternion[1]) + ' ')
    File_Quatr.write(str("%04f" %quaternion[2]) + ' ')
    File_Quatr.write(str("%04f" %quaternion[3]) + '\n')
File_Quatr.close()




Sp_Ind = -1
Pose_File = '/home/yaqub/Downloads/Temp/Mono_CadaCoord/Mono_02_' + Split2[Sp_Ind] + '.txt'

#Pose_File="/home/yaqub/Downloads/Temp/Mono_0-22_2020-09-18-14-03-36_sub3300-4200.txt"
Pose_file = open(Pose_File, 'r')
Pose_list = Pose_file.readlines()
#NamePose_file = open("/home/yaqub/Downloads/test_145_SegDepth_New2.txt", 'w')
offset = 0
cam = bpy.data.objects['Camera']
CoordQuater = cam.rotation_quaternion
CoordTrans  = cam.location
#Name_Pose_file = open("/home/yaqub/Downloads/Test2.txt", 'w')
Angles = [math.pi/2, math.pi, -math.pi/2, 0]
Angles = np.array(Angles)
Axes =   [[1.0, 0.0 , 0.0], [0.0, 1.0 , 0.0], [0.0, 0.0 , 1.0], [0.0, 1.0 , 1.0], [1.0, 0.0 , 1.0], [1.0, 1.0 , 0.0], [1, 1, 1]] 
Axes = np.array(Axes)
i=0
for Axes_Index in range(len(Axes)):
    Axis = Axes[Axes_Index]
    for Rot_Index1 in range(3):
        Blender2DSO = Euler(Angles[Rot_Index1]*Axis , 'XYZ').to_matrix().to_4x4()
        for Rot_Index2 in range(3):
            mat_alignRotation = Euler(Angles[Rot_Index2]*Axis, 'XYZ').to_matrix().to_4x4()  
            #Name_Pose_file = open("/home/yaqub/Downloads/Name_Axis" + str(Axis[0])+ str(Axis[1])+ str(Axis[2])+'Rot_Index1'+str(Rot_Index1)+'Rot_Index2'+str(Rot_Index2)+ ".txt", 'w')
            print(i); i +=1
            if i == 20:
                Name_Pose_file = open(Pose_File[:-4] + str(i) + ".txt", 'w')    
                for Index_Pose in range(len(Pose_list) - offset):
                    Index_Pose = Index_Pose + offset
                    Coordinates = Pose_list[Index_Pose].split(' ')
                    CoordQuater[0] = np.float64(Coordinates[4])
                    CoordQuater[1] = np.float64(Coordinates[5])
                    CoordQuater[2] = np.float64(Coordinates[6])
                    CoordQuater[3] = np.float64(Coordinates[7])
                    CoordTrans [0] = np.float64(Coordinates[1])
                    CoordTrans [1] = np.float64(Coordinates[2])
                    CoordTrans [2] = np.float64(Coordinates[3])    
                    #SLAMCorrdQuater = VSLAMMappingFromBlender2DSO(CoordTrans, CoordQuater)
                    Trans = Matrix.Translation(CoordTrans)
                    Rotat = CoordQuater.to_matrix().to_4x4()
                    DSOPose = Blender2DSO * Trans * Rotat * mat_alignRotation
                    #Name_Pose_file.write('/home/jonmoham/DataForTraining/BlenderData/Movie_Long_1/Masked_ImageDepth/' + Coordinates[0][0:-7]+'RGD.png ')
                    #Name_Pose_file.write(Coordinates[0][0:-7]+'RGD.png ')
                    Name_Pose_file.write(Coordinates[0]+' ')
    #                Name_Pose_file.write(str(DSOPose.translation[0])+' ' + str(DSOPose.translation[1])+' ' + str(DSOPose.translation[2]))
    #                Name_Pose_file.write(' ' + str(DSOPose.to_quaternion()[0])+' ' + str(DSOPose.to_quaternion()[1])+' ' + str(DSOPose.to_quaternion()[2])+' ' + str(DSOPose.to_quaternion()[3])+ ' \n') 
                    Name_Pose_file.write(str("%04f " %DSOPose.translation[0])+ str("%04f " %DSOPose.translation[1])+ str("%04f " %DSOPose.translation[2]))
                    Name_Pose_file.write(str("%04f " %DSOPose.to_quaternion()[0])+ str("%04f " %DSOPose.to_quaternion()[1]) + str("%04f " %DSOPose.to_quaternion()[2]) + str("%04f" %DSOPose.to_quaternion()[3])+ ' \n')                   

Pose_file.close()

















## mPort stuff

#loading the SMPL template
dd = pickle.load(open('/home/sergey/mPort/Yaqub/Python/smpl_models/smpl/SMPL_FEMALE.pkl', 'rb'), encoding='latin-1')
dd.keys()


PCL = np.array(dd['v_template'])
ax = plt.axes(projection='3d')
ax.scatter3D(PCL[:,0], PCL[:,1], PCL[:,2])
plt.show()




#dictionary to struct
class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

then call like this:
data_struct = Struct(**pickle.load(smpl_file, encoding='latin-1'))


#this should be called in the hello_smpl.py

m = load_model( '../models/basicmodel_m_lbs_10_207_0_v1.1.0.pkl' )
mm = np.array(m)
m_tor = m.copy()
m_leg = m.copy()
vert_ind_tor = []
vert_ind_leg = []
for i in range(len(mm)):
    if mm[i,1] > (-.5):
        vert_ind_tor.append(i)
    if mm[i,1] < (-.2):
        vert_ind_leg.append(i)

joint_ind_tor = [0, 1, 2, 3, 6, 9, 12, 13, 14,
	   15, 16, 17, 18, 19, 20, 21, 22, 23]
joint_ind_leg = [0, 1, 2, 4, 5, 7, 8, 10, 11]
J_reg = m.J_regressor.indices
two36_ind_leg = [i  for i,I in enumerate(J_reg) if sum(I==joint_ind_leg)]
two36_ind_tor = [i  for i,I in enumerate(J_reg) if sum(I==joint_ind_tor)]
m_tor.J = m.J[joint_ind_tor,:]
m_leg.J = m.J[joint_ind_leg,:]
#m_tor.J.a = m.J.a[:,joint_ind_tor]
#m_leg.J.a = m.J.a[:,joint_ind_leg]
m_tor.J_regressor = m.J_regressor[joint_ind_tor, :]
m_leg.J_regressor = m.J_regressor[joint_ind_leg, :]
m_tor.J_regressor = m_tor.J_regressor[:, vert_ind_tor]
m_leg.J_regressor = m_leg.J_regressor[:, vert_ind_leg]
m_tor.J_regressor_prior = m.J_regressor_prior[joint_ind_tor, :]
m_leg.J_regressor_prior = m.J_regressor_prior[joint_ind_leg, :]
m_tor.J_regressor_prior = m_tor.J_regressor_prior[:, vert_ind_tor]
m_leg.J_regressor_prior = m_leg.J_regressor_prior[:, vert_ind_leg]
m_tor.J_transformed = m.J_transformed[joint_ind_tor, :]
m_leg.J_transformed = m.J_transformed[joint_ind_leg, :]
m_tor.weights_prior = m.weights_prior[:,joint_ind_tor]
m_leg.weights_prior = m.weights_prior[:,joint_ind_leg]
m_tor.weights_prior = m_tor.weights_prior[vert_ind_tor,:]
m_leg.weights_prior = m_leg.weights_prior[vert_ind_leg,:]
m_tor.weights = m.weights[:,joint_ind_tor]
m_leg.weights = m.weights[:,joint_ind_leg]
m_tor.weights = m_tor.weights[vert_ind_tor,:]
m_leg.weights = m_leg.weights[vert_ind_leg,:]
m_tor.vert_sym_idxs = m.vert_sym_idxs[vert_ind_tor]
m_leg.vert_sym_idxs = m.vert_sym_idxs[vert_ind_leg]
m_tor.v_template = m.v_template[vert_ind_tor,:]
m_leg.v_template = m.v_template[vert_ind_leg,:]
m_tor.v_shaped = m.v_shaped[vert_ind_tor,:]
m_leg.v_shaped = m.v_shaped[vert_ind_leg,:]
m_tor.v_posed = m.v_posed[vert_ind_tor,:]
m_leg.v_posed = m.v_posed[vert_ind_leg,:]
m_tor.shapedirs = m.shapedirs[vert_ind_tor,:,:]
m_leg.shapedirs = m.shapedirs[vert_ind_leg,:,:]
m_tor.posedirs = m.posedirs[vert_ind_tor,:,:]
m_leg.posedirs = m.posedirs[vert_ind_leg,:,:]
m_tor.a = m.a[vert_ind_tor,:]
m_leg.a = m.a[vert_ind_leg,:]


torsoObj = open('SMPL2_FEMALE' +'.pkl', 'wb')
pickle.dump(dd3 , torsoObj)
torsoObj.close()


joblib.dump(m_leg, "basicmodel_m_lbs_10_207_0_v1.1.0_torso.pkl", compress=3)
joblib.dump(m_leg, "basicmodel_m_lbs_10_207_0_v1.1.0_leg.pkl", compress=3)



#Hands up pose
m.pose[1*3 + 1] = np.pi/8 # Leg joints
m.pose[2*3 + 1] = -np.pi/8
m.pose[1*3 + 2] = np.pi/32 # Leg joints
m.pose[2*3 + 2] = -np.pi/32

m.pose[14*3 + 0] = -np.pi/16 # Shoulder joints
m.pose[13*3 + 0] = -np.pi/16

m.pose[17*3 + 0] = -np.pi/2  # Shoulder joints
m.pose[17*3 + 1] = np.pi/3.2
m.pose[17*3 + 2] = -np.pi/12
m.pose[19*3 + 1] = np.pi/1.5
m.pose[21*3 + 1] = np.pi/8

m.pose[16*3 + 0] = -np.pi/2
m.pose[16*3 + 1] = -np.pi/3.2
m.pose[16*3 + 2] = np.pi/12
m.pose[18*3 + 1] = -np.pi/1.5 # Elbow joints
m.pose[20*3 + 1] = -np.pi/8



#this should be called in the bodey_model.py

with open('/home/sergey/mPort/Yaqub/Python/smpl_models/smpl/SMPL_NEUTRAL.pkl', 'rb') as smpl_file:
    m = pickle.load(smpl_file,encoding='latin1')

                                                 
mm = np.array(m['v_template'])
m_tor = m.copy()
m_leg = m.copy()
vert_ind_tor = []
vert_ind_leg = []
for i in range(len(m['v_template'])):
    if mm[i,1] > (-.5):
        vert_ind_tor.append(i)
    if mm[i,1] < (-.2):
        vert_ind_leg.append(i)


HandUp = np.array([3.14, 3.14, 0., 0., 0.392, 0.098,
                                        0., -0.39269908, -0.09817477, 0., 0., 0.,
                                        0., 0., 0., 0., 0., 0.,
                                        0., 0., 0., 0., 0., 0.,
                                        0., 0., 0., 0., 0., 0.,
                                        0., 0., 0., 0., 0., 0.,
                                        0., 0., 0., -0.196, 0., 0.,
                                        -0.196, 0., 0., 0., 0., 0.,
                                        -1.571, -0.981, 0.261, -1.570, 0.981, -0.261,
                                        0., -2.094, 0., 0., 2.094, 0.,
                                        0., -0.392, 0., 0., 0.392, 0.,
                                        0., 0., 0., 0., 0., 0.])


joint_ind_tor =        [0,         1, 2, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
m_tor['kintree_table']=np.array([4294967295,0, 0, 0, 3, 4, 5 , 5 , 5 , 6 , 7 , 8 , 10, 11, 12, 13, 14, 15])	    
joint_ind_leg =                 [0,         1, 2, 4, 5, 7, 8, 10, 11]
m_leg['kintree_table']=np.array([4294967295,0, 0, 1, 2, 3, 4 , 5,  6])	
J_reg = m['J_regressor'].indices
two36_ind_leg = [i  for i,I in enumerate(J_reg) if sum(I==joint_ind_leg)]
two36_ind_tor = [i  for i,I in enumerate(J_reg) if sum(I==joint_ind_tor)]
m_tor['J'] = m['J'][joint_ind_tor,:]
m_leg['J'] = m['J'][joint_ind_leg,:]
#m_tor['J.a = m['J.a[:,joint_ind_tor]
#m_leg['J.a = m['J.a[:,joint_ind_leg]
m_tor['J_regressor'] = m['J_regressor'][joint_ind_tor, :]
m_leg['J_regressor'] = m['J_regressor'][joint_ind_leg, :]
m_tor['J_regressor'] = m_tor['J_regressor'][:, vert_ind_tor]
m_leg['J_regressor'] = m_leg['J_regressor'][:, vert_ind_leg]
m_tor['J_regressor_prior'] = m['J_regressor_prior'][joint_ind_tor, :]
m_leg['J_regressor_prior'] = m['J_regressor_prior'][joint_ind_leg, :]
m_tor['J_regressor_prior'] = m_tor['J_regressor_prior'][:, vert_ind_tor]
m_leg['J_regressor_prior'] = m_leg['J_regressor_prior'][:, vert_ind_leg]
m_tor['weights_prior'] = m['weights_prior'][:,joint_ind_tor]
m_leg['weights_prior'] = m['weights_prior'][:,joint_ind_leg]
m_tor['weights_prior'] = m_tor['weights_prior'][vert_ind_tor,:]
m_leg['weights_prior'] = m_leg['weights_prior'][vert_ind_leg,:]
m_tor['weights'] = m['weights'][:,joint_ind_tor]
m_leg['weights'] = m['weights'][:,joint_ind_leg]
m_tor['weights'] = m_tor['weights'][vert_ind_tor,:]
m_leg['weights'] = m_leg['weights'][vert_ind_leg,:]
#m_tor['vert_sym_idxs'] = m['vert_sym_idxs'][vert_ind_tor]
#m_leg['vert_sym_idxs'] = m['vert_sym_idxs'][vert_ind_leg]
m_tor['v_template'] = m['v_template'][vert_ind_tor,:]
m_leg['v_template'] = m['v_template'][vert_ind_leg,:]
m_tor['shapedirs'] = m['shapedirs'][vert_ind_tor,:,:]
m_leg['shapedirs'] = m['shapedirs'][vert_ind_leg,:,:]




joint_ind_tor72 = np.zeros((len(joint_ind_tor)*3), np.int16)
joint_ind_leg72 = np.zeros((len(joint_ind_leg)*3), np.int16)
joint_ind_tor207= np.zeros(((len(joint_ind_tor)-1)*9), np.int16)
joint_ind_leg207= np.zeros(((len(joint_ind_leg)-1)*9), np.int16)
for i, I in enumerate(joint_ind_leg):
	joint_ind_leg72[i*3:(i*3)+3] = [I*3, I*3 + 1, I*3 + 2]
for i, I in enumerate(joint_ind_tor):
	joint_ind_tor72[i*3:(i*3)+3] = [I*3, I*3 + 1, I*3 + 2]
	
for i in range(len(joint_ind_tor[0:])-1):
	Inds = np.linspace(joint_ind_tor[i]*9,joint_ind_tor[i]*9+8, 9)
	Inds = Inds.astype('int16')
	joint_ind_tor207[i*9:(i*9)+9] = Inds
for i in range(len(joint_ind_leg[0:])-1):
	Inds = np.linspace(joint_ind_leg[i]*9,joint_ind_leg[i]*9+8, 9)
	Inds = Inds.astype('int16')
	joint_ind_leg207[i*9:(i*9)+9] = Inds
	

m_tor['pose'] = HandUp[joint_ind_tor72]
m_leg['pose'] = HandUp[joint_ind_leg72]
m_tor['body_pose'] = HandUp[joint_ind_tor72[1:]]
m_leg['body_pose'] = HandUp[joint_ind_leg72[1:]]

m_tor['posedirs'] = m['posedirs'][vert_ind_tor,:,:]
m_leg['posedirs'] = m['posedirs'][vert_ind_leg,:,:]
m_tor['posedirs'] = m_tor['posedirs'][:,:,joint_ind_tor207]
m_leg['posedirs'] = m_leg['posedirs'][:,:,joint_ind_leg207]


torsoObj = open('SMPLTorso_NEUTRAL' +'.pkl', 'wb')
pickle.dump(m_tor , torsoObj)
torsoObj.close()
legObj = open(  'SMPLLeg_NEUTRAL'   +'.pkl', 'wb')
pickle.dump(m_leg , legObj)
legObj.close()





pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.array(m.a))
o3d.visualization.draw_geometries([pcd])

radii = [0.005, 0.01, 0.02, 0.04]
rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    pcd, o3d.utility.DoubleVector(radii))
o3d.visualization.draw_geometries([pcd, rec_mesh])


#3D scatter plot
from matplotlib import pyplot as plt
PCLsm = modelVerts.detach().to('cpu').numpy()[0,:,:] 
PCLmes = meshVerts.detach().to('cpu').numpy()[0,:,:] 
ax = plt.axes(projection='3d')
ax.scatter3D(PCLsm[:,0], PCLsm[:,1], PCLsm[:,2])
ax.scatter3D(PCLmes[:,0], PCLmes[:,1], PCLmes[:,2], c='g')



U=Normals[0::5,0]
V=Normals[0::5,1]
W=Normals[0::5,2]
X=Vertices2[0::5,0]
Y=Vertices2[0::5,1]
Z=Vertices2[0::5,2]
ax = plt.axes(projection='3d')
ax.quiver(X,Y,Z,U,V,W, length=0.03, normalize=False)
plt.show()


#3D quiver 
U=np.array(PCL.normals)[:,0]
V=np.array(PCL.normals)[:,1]
W=np.array(PCL.normals)[:,2]
X=np.array(PCL.points)[:,0]
Y=np.array(PCL.points)[:,1]
Z=np.array(PCL.points)[:,2]
ax = plt.axes(projection='3d')
#ax.quiver(X,Y,Z,U,V,W, length=0.03, normalize=True)
ax.quiver(X,Y,Z,U,V,W, edgecolor='k', facecolor='None', linewidth=.5, length=0.03,)
plt.show()



#threshold based on normals angles
Normals = np.array(PCL.normals)
Angles = np.zeros((len(Normals)), np.float32)
Camera_normal=np.array([1,0,0.0])
for i in range(len(Angles)):
	Angles[i] = np.arccos(np.dot(Camera_normal, Normals[i,:])/(np.linalg.norm(
		Camera_normal)*np.linalg.norm(Normals[i,:]))) 



SmallAngs = np.where(Angles < np.pi/2)
LargeAngs = np.where(Angles > np.pi/2)

PCLsmall = np.squeeze(np.array(PCL.points)[SmallAngs,:])
PCLlarge = np.squeeze(np.array(PCL.points)[LargeAngs,:])
ax.scatter3D(PCLsmall[:,0], PCLsmall[:,1], PCLsmall[:,2],c='r')
ax.scatter3D(PCLlarge[:,0], PCLlarge[:,1], PCLlarge[:,2],c='y')



pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)








import numpy as np
import open3d as o3d

Mesh = o3d.io.read_triangle_mesh('/home/sergey/mPort/Datasets/smpl_models/smpl/smpl_female_handUp.obj')
Mesh.compute_vertex_normals()
Normals = np.array(Mesh.triangle_normals)
Vertices = np.array(Mesh.vertices)
Vertices2 = np.zeros((int(len(Vertices)/3),3))

for i in range(len(Vertices2)):   ## this could be improved
    Vertices2[i,0] = np.mean([Vertices[i * 3, 0], Vertices[(i * 3) + 1, 0], Vertices[(i * 3) + 2, 0]])
    Vertices2[i,1] = np.mean([Vertices[i * 3, 1], Vertices[(i * 3) + 1, 1], Vertices[(i * 3) + 2, 1]])
    Vertices2[i,2] = np.mean([Vertices[i * 3, 2], Vertices[(i * 3) + 1, 2], Vertices[(i * 3) + 2, 2]])


PCD = o3d.geometry.PointCloud()
PCD.points = o3d.utility.Vector3dVector(Vertices2)
PCD.estimate_normals()
for i in range(len(Vertices2)):   
    PCD.normals[i] = Normals[i,:]

o3d.io.write_point_cloud("smpl_neutral_handUp.ply", PCD)    
    


U=np.array(self.ply_normals) [0::5,0]
V=np.array(self.ply_normals) [0::5,1]
W=np.array(self.ply_normals) [0::5,2]
X=np.array(self.ply_vertices)[0::5,0]
Y=np.array(self.ply_vertices)[0::5,1]
Z=np.array(self.ply_vertices)[0::5,2]


U=np.array(PCD_sub.normals) [:,0]
V=np.array(PCD_sub.normals) [:,1]
W=np.array(PCD_sub.normals) [:,2]
X=np.array(PCD_sub.points)[:,0]
Y=np.array(PCD_sub.points)[:,1]
Z=np.array(PCD_sub.points)[:,2]




ax2 = plt.axes(projection='3d')
ax2.quiver(X,Y,Z,U,V,W, edgecolor='k', facecolor='None', linewidth=.5, length=0.03,)
plt.show()



PCLsm = np.array(m.r)
ax = plt.axes(projection='3d')
ax.scatter3D(PCLsm[:,0], PCLsm[:,1], PCLsm[:,2])

ax.view_init(-90,0)

#ax.scatter(x, z, y)
PCLsm2 = np.squeeze(joints3d.detach().to('cpu'))[0:24,:]
n = np.linspace(1, len(PCLsm2[:,2]), len(PCLsm2[:,2])).astype('int')
for i, _ in enumerate(n):
    #ax.annotate(txt, (x[i], z[i], y[i]))
    ax.text(PCLsm2[i,0], PCLsm2[i,1], PCLsm2[i,2], str(i)) 


23237724
23035734 interest amount 8757.13  
    5730 interest amount 8293.06 
import open3d as o3d
Vert = np.squeeze(modelVerts.detach().to('cpu').numpy())
PCD_sub = o3d.geometry.PointCloud()
PCD_sub.points = o3d.utility.Vector3dVector(Vert)
o3d.io.write_point_cloud("Angle_selected_front.ply", PCD_sub)


##sorting based on the value of the integer
    lossFiles = list(sorted(pathlib.Path('.',saveLossDir).glob(f'{1}_*_loss.npz'),
                             key=lambda i: int(os.path.splitext(os.path.basename(i).split('_')[1])[0])))

lossFiles=
[PosixPath('data_SMPL_output/EM_SV_Avg_4Direction_PartialDepth/Weichen_Leg_Slow_Turn_23_Oct_2021_at_12_50_pm/debug/1_0_loss.npz'),
 PosixPath('data_SMPL_output/EM_SV_Avg_4Direction_PartialDepth/Weichen_Leg_Slow_Turn_23_Oct_2021_at_12_50_pm/debug/1_1_loss.npz'),
 PosixPath('data_SMPL_output/EM_SV_Avg_4Direction_PartialDepth/Weichen_Leg_Slow_Turn_23_Oct_2021_at_12_50_pm/debug/1_2_loss.npz'), ...


# one line loop
o = [i for i in range(68)]







































