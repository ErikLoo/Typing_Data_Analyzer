import numpy as np
import glob as glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Ellipse
import pickle
from scipy import signal


def levenshteinDistance(s1, s2):
    # the mininum of # of operations needed to change a word into another
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def true_coord():
    '''

    the true coords of each key on the phone
    '''
    true_dict = {
        'a': [108,1660],
        'b': [648,1825],
        'c': [432,1825],
        'd': [324,1660],
        'e': [270,1495],
        'f': [432,1660],
        'g': [540,1660],
        'h': [648,1660],
        'i': [810,1495],
        'j': [756,1660],
        'k': [864,1660],
        'l': [972,1660],
        'm': [864,1825],
        'n': [756,1825],
        'o': [918,1495],
        'p': [1026,1495],
        'q': [54,1495],
        'r': [378,1495],
        's': [216,1660],
        't': [486,1495],
        'u': [702,1495],
        'v': [540,1825],
        'w': [162,1495],
        'x': [324,1825],
        'y': [594,1495],
        'z': [216,1825],
        ' ': [539,1988]}
    return true_dict


def get_touch_data(df_touch_time,df_touch,start_t, end_t):
    '''

    get the x y coords of a key
    '''
    coord_data = []
    y_offset = 1405.95
    for i, t in df_touch_time.iteritems():
        # print("start_t: " + str(start_t) + " end_t: " + str(end_t) + "| t: " + str(t))

        if t>=start_t and t<=end_t:
            x = df_touch['x'].iloc[i]
            y = df_touch['y'].iloc[i]
            coord_data.append([x,y+y_offset])
            # print("---start t: " + str(start_t) + " end_t: " + str(end_t) + "| t: " + str(t) + " x: " + str(x) + "| y: " + str(y))

    # only return the first and last entry for interpolation
    # print("start t: " + str(start_t) + " end_t: " + str(end_t) + " coord list: " + str(coord_data))
    # return [coord_data[0],coord_data[-1]]
    if len(coord_data)==0:
        print("start time: " + str(start_t) + " | end time: " + str(end_t))

    # return [coord_data[int(len(coord_data)/2)],coord_data[-1]]
    return [coord_data[0],coord_data[-1]]



def create_touch_data_dict(folder_name):
    '''
    {char : [[x1,y1],[x2,y2]]}
    '''

    char_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                 'u', 'v', 'w', 'x', 'y', 'z', ' ']

    char_touch_dict = {}

    true_coord_dict = true_coord()

    for fname in glob.glob(folder_name + '/*.xls'):
        # print(fname)
        # wrd_count = 1
        # duration = 0
        df_time = pd.read_excel(fname, sheet_name='input_time')
        df_touch = pd.read_excel(fname, sheet_name='touch_data')
        for i, char in df_time['intended character'].iteritems():
            if char == 'Space':
                char = ' '

            if isinstance(char, str):
                if char.lower() in char_list:
                    start_t = df_time['start time'].iloc[i]
                    end_t = df_time['end time'].iloc[i]

                    # if end_t-start_t<50:
                    #     start_t = end_t-100
                    # this is something issue with the start_time variable. DO NOT USE it !
                    start_t = end_t - 100

                    coord_data = get_touch_data(df_touch['time'],df_touch,start_t,end_t)
                    # print(coord_data)
                    center_coords = np.array(true_coord_dict[char.lower()])

                    # get the mid value not the first value
                    actual_coords = np.array(coord_data[0])

                    # print(np.array(list(coord_data[0])))
                    dist = np.linalg.norm(center_coords-actual_coords)

                    # print("dist: " + str(dist))

                    if dist <= 1.5*165:
                        '''
                        Get rid of the data that are far out of range 1.5 times the height
                        '''
                        if char not in char_touch_dict.keys():
                            char_touch_dict[char] = [coord_data]
                        else:
                            char_touch_dict[char].append(coord_data)

            else:
                # invalid correction operation
                pass

    return char_touch_dict



def show_char_pattern(character,char_touch_dict,posture):
    '''
    plot the touched points on each key
    :param character:
    :param char_touch_dict:
    :return:
    '''
    if character in char_touch_dict.keys():
        img = mpimg.imread('keyboard_screen_shot.jpg')
        imgplot = plt.imshow(img)
        coord_list = char_touch_dict[character]
        char_count = len(coord_list)
        for coord in coord_list:
            pt = coord[0]
            plt.plot(pt[0], pt[1], ".", markersize=1, color='red')

            # pt2 = coord[1]
            # plt.plot(pt2[0], pt2[1], ".", markersize=1, color='blue')

        # print("saving pattern for " + character)
        # plt.savefig(posture + "_training_data_pattern/" + character + "_" + str(char_count) + '_pattern.png', dpi=200)
        plt.clf()


def draw_center_point_on_image():
    '''
    Draw the center points over key.
    For verification purpose
    :return:
    '''
    img = mpimg.imread('keyboard_screen_shot.jpg')
    true_dict = true_coord()
    imgplot = plt.imshow(img)
    # print(true_dict.values())
    for coord in true_dict.values():
        plt.plot(coord[0], coord[1], ".", markersize=5, color='red')
    plt.show()


def get_motion_data(df_accel,df_gyro,start_t, end_t):
    '''

    save the motion data into a dict
    {char : [accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z]}
    10 data points for each element
    '''
    df_accel_time = df_accel['time']
    df_gyro_time = df_gyro['time']

    accel_x_list = []
    accel_y_list = []
    accel_z_list = []

    gyro_x_list = []
    gyro_y_list = []
    gyro_z_list = []

    # y_offset = 1405.95
    for i, t in df_accel_time.iteritems():
        # print("start_t: " + str(start_t) + " end_t: " + str(end_t) + "| t: " + str(t))

        if t>=start_t and t<=end_t:
            accel_x = df_accel['accel_x'].iloc[i]
            accel_y = df_accel['accel_y'].iloc[i]
            accel_z = df_accel['accel_z'].iloc[i]

            accel_x_list.append(accel_x)
            accel_y_list.append(accel_y)
            accel_z_list.append(accel_z)

            # print("---start t: " + str(start_t) + " end_t: " + str(end_t) + "| t: " + str(t) + " x: " + str(x) + "| y: " + str(y))

    # re-sample the data list to 10 data points per list in here
    rs_accel_x = signal.resample(accel_x_list, num = 10)
    rs_accel_y = signal.resample(accel_y_list, num = 10)
    rs_accel_z = signal.resample(accel_z_list, num = 10)




    for i, t in df_gyro_time.iteritems():
        # print("start_t: " + str(start_t) + " end_t: " + str(end_t) + "| t: " + str(t))

        if t>=start_t and t<=end_t:
            gyro_x = df_gyro['gyro_x'].iloc[i]
            gyro_y = df_gyro['gyro_y'].iloc[i]
            gyro_z = df_gyro['gyro_z'].iloc[i]

            gyro_x_list.append(gyro_x)
            gyro_y_list.append(gyro_y)
            gyro_z_list.append(gyro_z)

            # print("---start t: " + str(start_t) + " end_t: " + str(end_t) + "| t: " + str(t) + " x: " + str(x) + "| y: " + str(y))

    # re-sample the data list to 10 data points per list in here
    # rs_gyro_x = signal.resample(gyro_x_list, num = 10)
    # rs_gyro_y = signal.resample(gyro_y_list, num = 10)
    # rs_gyro_z = signal.resample(gyro_z_list, num = 10)

    # integrate to obtain the velocity change
    speed_x = integrate_to_get_speed(start_t,end_t,accel_x_list)
    speed_y = integrate_to_get_speed(start_t,end_t,accel_y_list)
    speed_z = integrate_to_get_speed(start_t,end_t,accel_z_list)

    # get the change in the angular velocity
    omega_x = gyro_x_list[-1]
    omega_y = gyro_y_list[-1]
    omega_z = gyro_z_list[-1]

    accl_x = accel_x_list[-1]
    accl_y = accel_y_list[-1]
    accl_z = accel_z_list[-1]


    # motion_data = np.hstack((rs_accel_x,rs_accel_y,rs_accel_z,rs_gyro_x,rs_gyro_y,rs_gyro_z))

    # motion_data = np.hstack((rs_accel_x,rs_accel_y,rs_accel_z))
    # motion_data = np.hstack((rs_gyro_x, rs_gyro_y, rs_gyro_z))

    # motion_data = np.hstack((speed_x,speed_y,speed_z,omega_x,omega_y,omega_z))

    # motion_data = np.hstack((accl_x,accl_y,accl_z,speed_x,speed_y,speed_z,omega_x,omega_y,omega_z))

    # motion_data = np.hstack((omega_x,omega_y,omega_z))
    motion_data = np.hstack((accl_x,accl_y,accl_z))
    # print(motion_data)
    # # viz the original signal and the resampled signal
    # plt.plot(np.linspace(0,len(accel_x_list),len(accel_x_list)),accel_x_list, ".", markersize=1, color='blue')
    # # plt.plot(np.linspace(0,len(accel_x_list),10),rs_accel_x, ".", markersize=5, color='red')
    # plt.show()
    # combine the data from all the axes into one feature vector

    return motion_data


def integrate_to_get_speed(start_t,end_t,accel_vec):
    '''
    Use numerical integration to get speed
    '''
    delta_t = (end_t-start_t)/(len(accel_vec)-1)
    trap = np.trapz(accel_vec)

    # print("delta: " + str(delta_t))

    return delta_t*trap
