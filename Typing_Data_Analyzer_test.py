import numpy as np
import pandas as pd
import glob
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Ellipse
import numpy as np
import pickle
from touch_model_train import Touch_model
from matplotlib.colors import LogNorm
from sklearn import mixture


def true_coord():
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


# calcualte the letter-wise accuracy
def text_entry_metric(folder_name):

    err_rate_list = []
    tlt_duration = 0
    tlt_wrd_count = 0
    char_list = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',' ']
    file_count = 0
    num_file = len(glob.glob(folder_name+'/*.xls'))

    for fname in glob.glob(folder_name+'/*.xls'):
        # print(fname)
        wrd_count = 1
        duration = 0
        df = pd.read_excel(fname,sheet_name='input_time')

        intend_char = df['intended character']
        int_str = ""
        for i, char in intend_char.iteritems():
            if char == 'Space':
                char = ' '
                wrd_count = wrd_count+1

            if isinstance(char, str):
                if char.lower() in char_list:
                    int_str = int_str + char.lower()
            else:
                #invalid correction operation
                pass


        typed_char = df['typed character']
        typed_str = ""
        for i, char in typed_char.iteritems():
            if char == 'Space':
                char = ' '

            if isinstance(char, str):
                if char.lower() in char_list:
                    typed_str = typed_str + char.lower()
            else:
                # invalid correction operation
                pass

        INF = levenshteinDistance(int_str,typed_str)
        C = max(len(int_str),len(typed_str)) - INF

        err_rate = INF/(INF+C)

        err_rate_list.append(err_rate)

        tlt_wrd_count = tlt_wrd_count + wrd_count

        # start_time = df['start time']
        # end_time = df['end time']

        duration = duration + df['end time'].iloc[-1] - df['start time'].iloc[0]
        tlt_duration  = tlt_duration + duration

        # print("intended: " + int_str + "| typed: " + typed_str + "| err rate: " + str(err_rate) + " | word count: " + str(wrd_count) + " | duration: " + str(duration))
        file_count = file_count + 1

        # print("Finished analyzing file " + str(file_count) + "/" + str(num_file))

    print(folder_name + "_avg uncorrected error rate:  " + str(sum(err_rate_list)/len(err_rate_list)))

    wpm = tlt_wrd_count/(tlt_duration/1000/60)

    print(folder_name + "_total wpm:  " + str(wpm) + " words per minute")
    print(" ")



def text_entry_metric_corrected(folder_name,model_name):

    err_rate_list = []
    tlt_duration = 0
    tlt_wrd_count = 0

    file_count = 0
    num_file = len(glob.glob(folder_name + '/*.xls'))

    char_list = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',' ']

    for fname in glob.glob(folder_name+'/*.xls'):
        # print(fname)
        wrd_count = 1
        duration = 0
        df = pd.read_excel(fname,sheet_name='input_time')

        df_time = pd.read_excel(fname, sheet_name='input_time')
        df_touch = pd.read_excel(fname, sheet_name='touch_data')


        intend_char = df['intended character']
        int_str = ""
        for i, char in intend_char.iteritems():
            if char == 'Space':
                char = ' '
                wrd_count = wrd_count+1

            if isinstance(char, str):
                if char.lower() in char_list:
                    int_str = int_str + char.lower()
            else:
                #invalid correction operation
                pass


        typed_char = df['typed character']
        typed_str = ""
        correct_str = ""
        t_model_str = ""
        old_char = " "
        for i, char in typed_char.iteritems():

            # get the coords
            # if the touch point is too far away
            if char != "Del":
                start_t = df_time['start time'].iloc[i]
                end_t = df_time['end time'].iloc[i]

                # there is a misalignment between the
                # if end_t - start_t < 50:
                #     start_t = end_t - 100

                start_t = end_t - 100

                coord_data = get_touch_data(df_touch['time'], df_touch, start_t, end_t)
                actual_coords = np.array(coord_data[0])

                if char == 'Space':
                    char = ' '

                if char == ' ':
                    pred_char = ' '
                    tm_char = ' '
                else:
                    pred_char, highest_p = predict_char(actual_coords,t_model_str,model_name)
                    # pred_char, highest_p = predict_char(actual_coords,typed_str,model_name)

                    tm_char, tm_p = predict_char_tm(actual_coords,typed_str)


                # pred_char = "gg"

                # print("Typed char: " + char + " | " + "Pred char: " + pred_char + " | Typed coord: " + str(coord_data))


                if isinstance(char, str):
                    if char.lower() in char_list:
                        typed_str = typed_str + char.lower()
                        correct_str = correct_str + pred_char.lower()
                        t_model_str = t_model_str + tm_char.lower()


                else:
                    # invalid correction operation
                    pass

            old_char = char


        INF = levenshteinDistance(int_str,correct_str)
        C = max(len(int_str),len(typed_str)) - INF

        err_rate = INF/(INF+C)

        err_rate_list.append(err_rate)

        tlt_wrd_count = tlt_wrd_count + wrd_count

        # start_time = df['start time']
        # end_time = df['end time']

        duration = duration + df['end time'].iloc[-1] - df['start time'].iloc[0]
        tlt_duration  = tlt_duration + duration

        # print("intended: " + int_str + "| typed: " + typed_str + "| err rate: " + str(err_rate) + " | word count: " + str(wrd_count) + " | duration: " + str(duration))
        file_count = file_count + 1

        # print("Finished analyzing file " + str(file_count) + "/" + str(num_file))
        if int_str == typed_str == correct_str:
            pass
        else:
            if file_count%2==10:
                print("Inten str: " + int_str)
                print("Typed str: " + typed_str)
                print("T_mod str: " + t_model_str)
                print("Combi str: " + correct_str + "\n")
                pass


    print(folder_name + "_avg uncorrected error rate revised by LM and " + model_name + " :  " + str(sum(err_rate_list)/len(err_rate_list)))

    wpm = tlt_wrd_count/(tlt_duration/1000/60)

    print(folder_name + "_total wpm:  " + str(wpm) + " words per minute")
    print(" ")


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



def create_touch_data_dict(folder_name):
    '''
    Visualize the data on an image
    Show the starting and the end point of each key
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

def get_touch_data(df_touch_time,df_touch,start_t, end_t):
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

    return [coord_data[int(len(coord_data)/2)],coord_data[-1]]


def build_model_and_generate_viz(f_name):
    char_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                 'u', 'v', 'w', 'x', 'y', 'z', ' ']
    char_dict = create_touch_data_dict(f_name)
    # char_dict = create_touch_data_dict("all_" + posture + "_data")
    # print(char_dict)
    for char in char_list:
        show_char_pattern(char, char_dict, None)

    return construct_gaussian_models(char_dict,None)

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

def construct_gaussian_models(char_dict,posture):
    # fit a 2D gaussian model for each key
    # plot the mean and variance ellipse on the keyboard layout
    img = mpimg.imread('keyboard_screen_shot.jpg')
    imgplot = plt.imshow(img)
    # convert data to numpy
    char_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                 'u', 'v', 'w', 'x', 'y', 'z', ' ']

    # create a gaussian model for every character

    model_dict = {}

    for char in char_list:
        if char in char_dict.keys():
            XY = convert_to_numpy(char_dict[char])
            x = XY[:, 0]
            y = XY[:, 1]
            cov = np.cov(x, y)
            lambda_, v = np.linalg.eig(cov)
            lambda_ = np.sqrt(lambda_)

            # plot the ellipse on the image
            ax = plt.gca()

            plt.plot(np.mean(x), np.mean(y), ".", markersize=1, color='red')

            ax.add_patch(Ellipse(xy=(np.mean(x), np.mean(y)),
                                 width=lambda_[0] * 2 * 2, height=lambda_[1] * 2 * 2,
                                 linewidth=1,
                                 facecolor='none',
                                 edgecolor='red',
                                 angle=np.rad2deg(np.arccos(v[0, 0]))))

            # save the model parameter in a dictionary
            model_dict[char] = {}
            model_dict[char]['mean'] = np.array([np.mean(x), np.mean(y)])
            model_dict[char]['cov'] = cov


            # print("Generated plot for " + char)
    # plt.scatter(x, y)
    # plt.savefig(posture + '_training_data_pattern/' + posture + '_2D_Gauss.png', dpi=200)
    print("Model generated")
    return model_dict



def convert_to_numpy(data):
    '''
    Only append the starting data
    :param data:
    :return:
    '''
    temp_data = []
    for coord in data:
        temp_data.append(coord[0])
    final_data = np.array(temp_data)

    return final_data

def predict_char_tm(coord,context=None):
    # import the touch model
    f = open('TM.pickle', 'rb')
    touch_model = pickle.load(f)
    f.close()

    highest_p = 0
    pred_char = ''
    char_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                 'u', 'v', 'w', 'x', 'y', 'z', ' ']

    for char in char_list:
        p_tm = touch_model.score_per_char(char,coord)
        p_lm = 1
        p_combined = p_tm

        # print("char: " + char + "| context: " + context +  "| p_tm: " + str(p_tm) + "| p_lm: " + str(p_lm))

        if p_combined>highest_p:
            highest_p = p_combined
            pred_char = char

    return pred_char,highest_p


def predict_char(coord,context,model_name):

    # import the touch model
    f = open(model_name, 'rb')
    touch_model = pickle.load(f)
    f.close()

    # import the language model
    f = open('KNLM.pickle', 'rb')
    KNL_model = pickle.load(f)
    f.close()
    highest_p = 0
    pred_char = ''
    char_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                 'u', 'v', 'w', 'x', 'y', 'z', ' ']


    for char in char_list:
        p_tm = touch_model.score_per_char(char,coord)
        # p_lm = KNL_model.score(char,list(context.split(" ")[-1]))
        p_lm = KNL_model.score(char,list(context))

        p_combined = p_tm*p_lm
        # p_combined = p_tm

        # print("char: " + char + "| context: " + context +  "| p_tm: " + str(p_tm) + "| p_lm: " + str(p_lm))

        if p_combined>highest_p:
            highest_p = p_combined
            pred_char = char

    return pred_char,highest_p





if __name__ == '__main__':

    #
    text_entry_metric("all_sitting_data_testing")
    text_entry_metric("all_walking_data_testing")

    # text_entry_metric_corrected("all_sitting_data_testing","TM.pickle")
    # text_entry_metric_corrected("all_sitting_data_testing","TM2.pickle")
    # text_entry_metric_corrected("all_walking_data_testing","TM.pickle")
    # text_entry_metric_corrected("all_walking_data_testing","TM2.pickle")
    # text_entry_metric_corrected("all_combined_data_testing","TM3.pickle")
    # text_entry_metric_corrected("all_sitting_data_testing","TM3.pickle")
    # text_entry_metric_corrected("all_walking_data_testing","TM3.pickle")
    text_entry_metric_corrected("all_combined_data_testing", "TM.pickle")
    text_entry_metric_corrected("all_combined_data_testing", "TM2.pickle")
    # text_entry_metric_corrected("all_combined_data_testing", "TM3.pickle")






    # print("----")
    #
    # text_entry_metric("all_walking_data_testing")
    # model_walking = build_model_and_generate_viz("all_walking_data_training")
    # text_entry_metric_corrected("all_walking_data_testing", model_walking)
    #
    # print("----")
    #
    # text_entry_metric_corrected("All_sitting_data_testing", model_walking)
    #
    # print("----")
    #
    # text_entry_metric_corrected("all_walking_data_testing", model_sitting)