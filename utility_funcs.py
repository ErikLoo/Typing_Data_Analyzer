import numpy as np
import glob as glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from matplotlib.patches import Ellipse
import pickle
from scipy import signal


def levenshteinDistance(s1, s2):
    # the mininum of # of operations needed to change a word into another

    ins_count = np.absolute(np.heaviside(len(s1)-len(s2),0)*(len(s1)-len(s2)))
    del_count = np.absolute(np.heaviside(len(s2)-len(s1),0)*(len(s2)-len(s1)))

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

    sub_count = distances[-1]-(ins_count+del_count)

    return distances[-1],ins_count,del_count,sub_count


def true_coord():
    '''

    adjust these coords for different phone models
    the true coords of each key on Samsung Galaxy S8
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

    return coord_data[0]


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

    # avg filer window size = 50
    n = 50

    MA_x = pd.Series(df_accel['accel_x'].rolling(n, min_periods=n).median(), name='accel_x_MA')
    MA_y = pd.Series(df_accel['accel_y'].rolling(n, min_periods=n).median(), name='accel_y_MA')
    MA_z = pd.Series(df_accel['accel_z'].rolling(n, min_periods=n).median(), name='accel_z_MA')
    df_accel_avg = df_accel.join(MA_x).join(MA_y).join(MA_z)

    df_accel_time = df_accel_avg['time']

    MA_x = pd.Series(df_gyro['gyro_x'].rolling(n, min_periods=n).mean(), name='gyro_x_MA')
    MA_y = pd.Series(df_gyro['gyro_y'].rolling(n, min_periods=n).mean(), name='gyro_y_MA')
    MA_z = pd.Series(df_gyro['gyro_z'].rolling(n, min_periods=n).mean(), name='gyro_z_MA')
    df_gyro_avg = df_gyro.join(MA_x).join(MA_y).join(MA_z)

    df_gyro_time = df_gyro_avg['time']

    accel_x_list = []
    accel_y_list = []
    accel_z_list = []

    gyro_x_list = []
    gyro_y_list = []
    gyro_z_list = []


    for i, t in df_accel_time.iteritems():
        # print("start_t: " + str(start_t) + " end_t: " + str(end_t) + "| t: " + str(t))

        if t>=start_t and t<=end_t:
            accel_x = df_accel_avg['accel_x_MA'].iloc[i]
            accel_y = df_accel_avg['accel_y_MA'].iloc[i]
            accel_z = df_accel_avg['accel_z_MA'].iloc[i]

            accel_x_list.append(accel_x)
            accel_y_list.append(accel_y)
            accel_z_list.append(accel_z)

    for i, t in df_gyro_time.iteritems():
        # print("start_t: " + str(start_t) + " end_t: " + str(end_t) + "| t: " + str(t))

        if t>=start_t and t<=end_t:
            gyro_x = df_gyro_avg['gyro_x_MA'].iloc[i]
            gyro_y = df_gyro_avg['gyro_y_MA'].iloc[i]
            gyro_z = df_gyro_avg['gyro_z_MA'].iloc[i]

            gyro_x_list.append(gyro_x)
            gyro_y_list.append(gyro_y)
            gyro_z_list.append(gyro_z)

    omega_x = gyro_x_list[-1]
    omega_y = gyro_y_list[-1]
    omega_z = gyro_z_list[-1]

    motion_data = np.hstack((omega_x,omega_y,omega_z))

    return motion_data


def integrate_to_get_speed(start_t,end_t,accel_vec):
    '''
    Use numerical integration to get speed
    '''
    delta_t = (end_t-start_t)/(len(accel_vec)-1)
    trap = np.trapz(accel_vec)

    # print("delta: " + str(delta_t))

    return delta_t*trap


def integrate_to_get_distance(start_t,end_t,accel_vec):
    dt = (end_t-start_t)/(len(accel_vec)-1)
    speed_vec=[]
    for i in range(len(accel_vec)):
        if i!=0:
            speed = integrate_to_get_speed(start_t,start_t+i*dt,accel_vec[0:i+1])
            speed_vec.append(speed)

    trap = np.trapz(speed_vec)

    return dt*trap


def generate_confusion_matrix(int_str,typed_str,f_name):
    char_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                 'u', 'v', 'w', 'x', 'y', 'z', ' ']
    int_chars = list(int_str)
    typed_chars = list(typed_str)

    data = {'int_chars': list(int_str),
            'typed_chars': list(typed_str)
            }

    df = pd.DataFrame(data, columns=['int_chars', 'typed_chars'])
    confusion_matrix = pd.crosstab(df['int_chars'], df['typed_chars'], rownames=['Intended'], colnames=['Typed'])

    row_count = confusion_matrix.shape[0]
    for i in range(row_count):
        confusion_matrix.iloc[[i]] = confusion_matrix.iloc[[i]]/confusion_matrix.iloc[[i]].values.sum()
    confusion_matrix.to_csv(f_name + '_confusion_mtx.csv')

    # print(f_name+" confusion mtx generated to file")
    return confusion_matrix
