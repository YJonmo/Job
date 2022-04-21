import bpy
import bpy_extras
import bpy, _cycles
import os
from mathutils import *
import numpy as np
import math
from bpy.types import Operator
from bpy.types import Panel
from bpy_extras.io_utils import ExportHelper
from mathutils import Vector, Euler, Quaternion, Matrix

#https://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera/175037?noredirect=1#comment293646_175037
#https://github.com/GSORF/Visual-GPS-SLAM/blob/master/02_Utilities/BlenderAddon/addon_vslam_groundtruth_Blender280.py#L34

#---------------------------------------------------------------
# 3x4 P matrix from Blender camera
#---------------------------------------------------------------

# Build intrinsic camera parameters from Blender camera data
#
# See notes on this in 
# blender.stackexchange.com/questions/15102/what-is-blenders-camera-projection-matrix-model
def get_calibration_matrix_K_from_blender(camd):
    f_in_mm = camd.lens
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = camd.sensor_width
    sensor_height_in_mm = camd.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    if (camd.sensor_fit == 'VERTICAL'):
        # the sensor height is fixed (sensor fit is horizontal), 
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio 
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else: # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal), 
        # the sensor height is effectively changed with the pixel aspect ratio
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm
    # Parameters of intrinsic calibration matrix K
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_v
    u_0 = resolution_x_in_px * scale / 2
    v_0 = resolution_y_in_px * scale / 2
    skew = 0 # only use rectangular pixels
    K = Matrix(
        ((alpha_u, skew,    u_0),
        (    0  , alpha_v, v_0),
        (    0  , 0,        1 )))
    return K

# Returns camera rotation and translation matrices from Blender.
# 
# There are 3 coordinate systems involved:
#    1. The World coordinates: "world"
#       - right-handed
#    2. The Blender camera coordinates: "bcam"
#       - x is horizontal
#       - y is up
#       - right-handed: negative z look-at direction
#    3. The desired computer vision camera coordinates: "cv"
#       - x is horizontal
#       - y is down (to align to the actual pixel coordinates 
#         used in digital images)
#       - right-handed: positive z look-at direction
def get_3x4_RT_matrix_from_blender(cam):
    # bcam stands for blender camera
    R_bcam2cv = Matrix(
        ((1, 0,  0),
         (0, -1, 0),
         (0, 0, -1)))
    # Transpose since the rotation is object rotation, 
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euleWorld.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam * location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()
    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam*cam.location
    # Use location from matrix_world to account for constraints:     
    T_world2bcam = -1*R_world2bcam * location
    # Build the coordinate transform matrix from world to computer vision camera
    R_world2cv = R_bcam2cv*R_world2bcam
    T_world2cv = R_bcam2cv*T_world2bcam
    # put into 3x4 matrix
    RT = Matrix((
        R_world2cv[0][:] + (T_world2cv[0],),
        R_world2cv[1][:] + (T_world2cv[1],),
        R_world2cv[2][:] + (T_world2cv[2],)
         ))
    return RT

def get_3x4_P_matrix_from_blender(cam):
    K = get_calibration_matrix_K_from_blender(cam.data)
    RT = get_3x4_RT_matrix_from_blender(cam)
    return K*RT, K, RT

# ----------------------------------------------------------
# Alternate 3D coordinates to 2D pixel coordinate projection code
# adapted from https://blender.stackexchange.com/questions/882/how-to-find-image-coordinates-of-the-rendered-vertex?lq=1
# to have the y axes pointing up and origin at the top-left corner
def project_by_object_utils(cam, point):
    scene = bpy.context.scene
    co_2d = bpy_extras.object_utils.world_to_camera_view(scene, cam, point)
    render_scale = scene.render.resolution_percentage / 100
    render_size = (
            int(scene.render.resolution_x * render_scale),
            int(scene.render.resolution_y * render_scale),
            )
    return Vector((co_2d.x * render_size[0], render_size[1] - co_2d.y * render_size[1]))

#%%#####################################################################################
#%%#####################################################################################
#%%#####################################################################################

def VSLAMMappingFromBlender2DSO(translation, quaternion): # conver from Blender to SLAM coordinates
    '''
    Mapping from Blender to DSO world space:
    Blender:
        Up = Y-Axis
        Right = X-Axis
        Forward = -Z-Axis
    DSO:
        Up = -Y-Axis
        Right = X-Axis
        Forward = Z-Axis
    x =  x
    y = -y
    z = -z
    Matrix:
         1   0   0
         0  -1   0
         0   0  -1
    '''
    # Create space transformation matrix
    Blender2DSO = Euler([math.pi/2.0, 0.0, 0.0], 'XYZ').to_matrix().to_4x4()
    # Create matrix from translation and quaternion
    mat_rot = quaternion.to_matrix().to_4x4()
    mat_alignRotation = Euler([-math.pi, 0.0, 0.0], 'XYZ').to_matrix().to_4x4() # This is necessary, because I have decided to rotate Blenders camera such that it "lies" in
    # the x-y-plane instead of x-z-plane (i.e. a rotation about 90 degrees around the x-axis) and thus is oriented with regards to the procedurally generated city.
    mat_trans = Matrix.Translation(translation)
    BlenderPoseCam2World = mat_trans * mat_rot * mat_alignRotation
    # Transform from Blender world space to DSO world space
    DSOPose = Blender2DSO * BlenderPoseCam2World
    # Mirror Quaternion along X-Axis is performed by flipping signs of the y and z component
    # Source: Philipp Kurth -> https://stackoverflow.com/questions/32438252/efficient-way-to-apply-mirror-effect-on-quaternion-rotation
    #newQuaternion = quaternion
    #newQuaternion.y = -quaternion.y
    #newQuaternion.z = -quaternion.z
    #return DSOPose.translation, DSOPose.to_quaternion() 
    return DSOPose


def VSLAMMappingFromDSO2Blender(translation, quaternion):
    '''
    Mapping from DSO to Blender space:
    
    x = -x
    y = -z
    z =  y
    
    Matrix:
        -1   0   0
         0   1   0
         0   0  -1
    '''
    # Create space transformation matrix
    DSO2Blender = Euler([-math.pi/2.0, 0.0, 0.0], 'XYZ').to_matrix().to_4x4()
    # Create matrix from translation and quaternion
    mat_rot = quaternion.to_matrix().to_4x4()
    mat_alignRotation = Euler([math.pi, 0.0, 0.0], 'XYZ').to_matrix().to_4x4() # This is necessary, because I have decided to rotate Blenders camera such that it "lies" in
    # the x-y-plane instead of x-z-plane (i.e. a rotation about 90 degrees around the x-axis) and thus is oriented with regards to the procedurally generated city.
    mat_trans = Matrix.Translation(translation)
    DSOPoseCam2World = mat_trans * mat_rot * mat_alignRotation # * mat_rotTurnAroundY
    # Transform from DSO world space to Blender world space
    BlenderPose = DSO2Blender * DSOPoseCam2World
    # Mirror Quaternion along X-Axis is performed by flipping signs of the y and z component
    # Source: Philipp Kurth -> https://stackoverflow.com/questions/32438252/efficient-way-to-apply-mirror-effect-on-quaternion-rotation
    #newQuaternion = quaternion
    #newQuaternion.y = -quaternion.y
    #newQuaternion.z = -quaternion.z
    #return BlenderPose.translation, BlenderPose.to_quaternion()
    return BlenderPose



#https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
def quatr2world(Coordinates):         #
    X=   np.float64(Coordinates[1])
    Y=   np.float64(Coordinates[2])
    Z=   np.float64(Coordinates[3])
    W_O= np.float64(Coordinates[4])
    X_O= np.float64(Coordinates[5])
    Y_O= np.float64(Coordinates[6])
    Z_O= np.float64(Coordinates[7])
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

def quatr2world(Coordinates):         #Coordinates is the quaternion
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

def world2quatr(World):
    World = World.astype(np.float64)                # 
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

def euler2quatr(Euler):         # roll, pitch, yaw
    #roll = np.float64(Euler[0])
    #pitch= np.float64(Euler[1])
    #yaw  = np.float64(Euler[2])
    
    cy = np.cos(Euler[2] * 0.5);
    sy = np.sin(Euler[2] * 0.5);
    cp = np.cos(Euler[1] * 0.5);
    sp = np.sin(Euler[1] * 0.5);
    cr = np.cos(Euler[0] * 0.5);
    sr = np.sin(Euler[0] * 0.5);

    Quaternion = np.zeros((4), np.float64) 
    Quaternion[0] = cr * cp * cy + sr * sp * sy;
    Quaternion[1] = sr * cp * cy - cr * sp * sy;
    Quaternion[2] = cr * sp * cy + sr * cp * sy;
    Quaternion[3] = cr * cp * sy - sr * sp * cy;

    return Quaternion;


# Calculates Rotation Matrix given euler angles.   Works well
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



#
## Checks if a matrix is a valid rotation matrix.
#def isRotationMatrix(R) :
#    Rt = np.transpose(R)
#    shouldBeIdentity = np.dot(Rt, R)
#    I = np.identity(3, dtype = R.dtype)
#    n = np.linalg.norm(I - shouldBeIdentity)
#    return n < 1e-6
## Calculates rotation matrix to euler angles
## The result is the same as MATLAB except the order
## of the euler angles ( x and z are swapped ).
#def world2euler(R) :
#    assert(isRotationMatrix(R))
#    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
#    singular = sy < 1e-6
#    if  not singular :
#        x = math.atan2(R[2,1] , R[2,2])
#        y = math.atan2(-R[2,0], sy)
#        z = math.atan2(R[1,0], R[0,0])
#    else :
#        x = math.atan2(-R[1,2], R[1,1])
#        y = math.atan2(-R[2,0], sy)
#        z = 0
#    return np.array([x, y, z])



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




#def rotmat2axang(matrix):
#    #https://github.com/eayvali/Pose-Estimation-for-Sensor-Calibration/blob/c9cfdac899df24440fbaf825f8feb422b79a35b9/helpers.py
#    """Convert the rotation matrix into the axis-angle notation.
#       The result is consistent with matlab implementation vrrotmat2vec
#    Conversion equations
#    ====================
#        x = Qzy-Qyz
#        y = Qxz-Qzx
#        z = Qyx-Qxy
#        r = hypot(x,hypot(y,z))
#        t = Qxx+Qyy+Qzz
#        theta = atan2(r,t-1)
#    @param matrix:  The 3x3 rotation matrix to update.
#    @type matrix:   3x3 numpy array
#    @return:    The 3D rotation axis and angle.
#    @rtype:     numpy 3D rank-1 array, float
#    """
#
#    # Axes.
#    axis = np.zeros(3, np.float64)
#    axis[0] = matrix[2,1] - matrix[1,2]
#    axis[1] = matrix[0,2] - matrix[2,0]
#    axis[2] = matrix[1,0] - matrix[0,1]
#
#    # Angle.
#    r = np.hypot(axis[0], np.hypot(axis[1], axis[2]))
#    t = matrix[0,0] + matrix[1,1] + matrix[2,2]
#    theta = math.atan2(r, t-1)
#
#    # Normalise the axis.
#    axis = axis / r
#
#    # Return the data.
#    return axis, theta 


def worldRotatX(World, angle):
    #angle = angle/57.29578
    #https://www.cs.helsinki.fi/group/goa/mallinnus/3dtransf/3drot.html#X-Axis%20Rotation
    x_new = World[0,3]
    y_new = World[1,3]*np.cos(angle) - World[2,3]*np.sin(angle)
    z_new = World[1,3]*np.sin(angle) + World[2,3]*np.cos(angle) 
    Trans= np.array([[1,  0,  0,  x_new],
        [0,  1,  0,  y_new],
        [ 0, 0,  1,  z_new],
        [ 0, 0,  0,      1]])
    Rz = np.array([[1,  0,  0,  0],
        [ 0,          np.cos(angle),  np.sin(angle),  0],
        [ 0,         -np.sin(angle),  np.cos(angle),  0],
        [ 0,          0,              0,          1]])
    World[0:4, 3] = [0,0,0,1]
    #World[:4, :4] = np.dot(np.dot(Rz[:4,:4], World[:4, :4]), Trans)
    World[:4, :4] = np.dot(Rz[:4,:4], World[:4, :4])
    World[:, 3] = Trans[:,3]
    return World, Rz

def worldRotatY(World, angle):
    #angle = angle/57.29578
    #https://www.cs.helsinki.fi/group/goa/mallinnus/3dtransf/3drot.html#X-Axis%20Rotation
    x_new = World[2,3]*np.sin(angle) + World[0,3]*np.cos(angle)
    y_new = World[1,3]
    z_new = World[2,3]*np.cos(angle) - World[0,3]*np.sin(angle)       
    Trans= np.array([[1,  0,  0,  x_new],
        [0,  1,  0,  y_new],
        [ 0, 0,  1,  z_new],
        [ 0, 0,  0,      1]])
    Rz = np.array([[np.cos(angle), 0,      -np.sin(angle),  0],
        [ 0,          1,              0,      0],
        [ np.sin(angle),              0, np.cos(angle),      0],
        [ 0,          0,              0,          1]])
    World[0:4, 3] = [0,0,0,1]
    #World[:4, :4] = np.dot(np.dot(Rz[:4,:4], World[:4, :4]), Trans)
    World[:4, :4] = np.dot(Rz[:4,:4], World[:4, :4])
    World[:, 3] = Trans[:,3]
    return World, Rz


def worldRotatZ(World, angle):
    #angle = angle/57.29578
    #https://www.cs.helsinki.fi/group/goa/mallinnus/3dtransf/3drot.html#X-Axis%20Rotation
    x_new = World[0,3]*np.cos(angle) - World[1,3]*np.sin(angle)
    y_new = World[0,3]*np.sin(angle) + World[1,3]*np.cos(angle)
    z_new = World[2,3]       
    Trans= np.array([[1,  0,  0,  x_new],
        [0,  1,  0,  y_new],
        [ 0, 0,  1,  z_new],
        [ 0, 0,  0,      1]])
    Rz = np.array([[np.cos(angle),  np.sin(angle),  0,  0],
        [-np.sin(angle),  np.cos(angle),  0,  0],
        [ 0,          0,              1,      0],
        [ 0,          0,              0,          1]])
    World[0:4, 3] = [0,0,0,1]
    #World[:4, :4] = np.dot(np.dot(Rz[:4,:4], World[:4, :4]), Trans)
    World[:4, :4] = np.dot(Rz[:4,:4], World[:4, :4])
    World[:, 3] = Trans[:,3]
    return World, Rz

def worldMirror(World, axis):
    #smaple use: Mirroed = worldMirror(World_stack, [0,0,1])
    #https://gamedev.stackexchange.com/questions/149062/how-to-mirror-reflect-flip-a-4d-transformation-matrix
    axis = np.array(axis)
    Mirror = np.eye(4)
    assert(np.sum(axis) == 1) 
    axis_row = np.dot(axis, np.array([0, 1, 2]))
    Mirror[axis_row,:3] = axis*(-1)          
    return np.matmul(Mirror, World)






def quatMirror(Quat, axis):
    # Quat = [x, y, x, qw, qx, qy, qz]
    # axis = [0, 0, 1] # sample mirror on z axis
    #smaple use: Mirroed = quatMirror(World_stack, [0,0,1])
    Quat = np.squeeze(Quat.astype(np.float64))
    axis = np.array(axis)
    axis = np.logical_not(axis) + axis*[-1]
    Quat[0:3] *= axis
    Quat[4:7] *= (axis*[-1])          
    return Quat

'''

# ----------------------------------------------------------
if __name__ == "__main__":
    # Insert your camera name here
    cam = bpy.data.objects['Camera']
    P, K, RT = get_3x4_P_matrix_from_blender(cam)
    print("K")
    print(K)
    print("RT")
    print(RT)
    print("P")
    print(P)

    print("==== Tests ====")
    e1 = Vector((1, 0,    0, 1))
    e2 = Vector((0, 1,    0, 1))
    e3 = Vector((0, 0,    1, 1))
    O  = Vector((0, 0,    0, 1))

    p1 = P * e1
    p1 /= p1[2]
    print("Projected e1")
    print(p1)
    print("proj by object_utils")
    print(project_by_object_utils(cam, Vector(e1[0:3])))

    p2 = P * e2
    p2 /= p2[2]
    print("Projected e2")
    print(p2)
    print("proj by object_utils")
    print(project_by_object_utils(cam, Vector(e2[0:3])))

    p3 = P * e3
    p3 /= p3[2]
    print("Projected e3")
    print(p3)
    print("proj by object_utils")
    print(project_by_object_utils(cam, Vector(e3[0:3])))

    pO = P * O
    pO /= pO[2]
    print("Projected world origin")
    print(pO)
    print("proj by object_utils")
    print(project_by_object_utils(cam, Vector(O[0:3])))

    # Bonus code: save the 3x4 P matrix into a plain text file
    # Don't forget to import numpy for this
    nP = numpy.matrix(P)
    numpy.savetxt("/tmp/P3x4.txt", nP)  # to select precision, use e.g. fmt='%.2f'
'''
