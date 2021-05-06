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

    print(folder_name + "_avg uncorrected error rate:  " + str(sum(err_rate_list)/len(err_rate_list)))

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
                    start_t =  df_time['start time'].iloc[i]
                    end_t = df_time['end time'].iloc[i]
                    coord_data = get_touch_data(df_touch['time'],df_touch,start_t,end_t)
                    # print(coord_data)
                    center_coords = np.array(true_coord_dict[char.lower()])

                    actual_coords = np.array(coord_data[0])

                    # print(np.array(list(coord_data[0])))
                    dist = np.linalg.norm(center_coords-actual_coords)

                    # print("dist: " + str(dist))

                    if dist <= 1*165:
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
        if t>=start_t and t<=end_t:
            x = df_touch['x'].iloc[i]
            y = df_touch['y'].iloc[i]
            coord_data.append([x,y+y_offset])

    # only return the first and last entry for interpolation
    return [coord_data[0],coord_data[-1]]

def build_model_and_generate_viz(posture):
    char_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                 'u', 'v', 'w', 'x', 'y', 'z', ' ']
    char_dict = create_touch_data_dict("Eric_sitting_data_3")
    for char in char_list:
        show_char_pattern(char, char_dict, posture)

    return construct_gaussian_models(char_dict,posture)

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

        print("saving pattern for " + character)
        plt.savefig(posture + "_character_pattern/" + character + "_" + str(char_count) + '_pattern.png', dpi=200)
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

            first_v = v[:,0]
            slope = first_v[1]/first_v[0]

            ax.add_patch(Ellipse(xy=(np.mean(x), np.mean(y)),
                                 width=lambda_[0] * 2 * 2, height=lambda_[1] * 2 * 2,
                                 linewidth=1,
                                 facecolor='none',
                                 edgecolor='red',
                                 angle=np.rad2deg(np.arctan(slope))))
            # print("char: " + char)
            # print(lambda_)
            # print(v)

            # save the model parameter in a dictionary
            model_dict[char] = {}
            model_dict[char]['mean'] = np.array([np.mean(x), np.mean(y)])
            model_dict[char]['cov'] = cov


            print("Generated plot for " + char)
    # plt.scatter(x, y)
    plt.savefig(posture + '_character_pattern/' + posture + '_2D_Gauss.png', dpi=200)
    return model_dict



def convert_to_numpy(data):
    temp_data = []
    for coord in data:
        temp_data.append(coord[0])
    final_data = np.array(temp_data)

    return final_data

def predict_char(model_list,coord):
    # build a model from the dict
    highest_p = 0
    pred_char = ''

    for char in model_list.keys():
        mean = model_list[char]['mean']
        cov = model_list[char]['cov']
        prob = multivariate_gaussian(coord,mean,cov)
        # print("p(" + char + "|" + str(coord) + ") : " + str(prob))
        if prob>=highest_p:
            highest_p = prob
            pred_char = char

    # print("the predicted character is : [" + pred_char + "] with a prob of " + str(highest_p) )
    return pred_char,highest_p

def multivariate_gaussian(pos,mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos.

    pos is an array constructed by packing the meshed arrays of variables
    x_1, x_2, x_3, ..., x_k into its _last_ dimension.

    """
    pos = np.array(pos)
    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    return np.exp(-fac / 2) / N


def interactive_demo():
    f = open('TM.pickle', 'rb')
    model = pickle.load(f)
    f.close()
    img = mpimg.imread('keyboard_screen_shot.jpg')
    # imgplot = plt.imshow(img)

    ax = plt.gca()
    fig = plt.gcf()
    implot = ax.imshow(img)

    def onclick(event):
        if event.xdata != None and event.ydata != None:
            coord = (event.xdata,event.ydata)
            pred_char,highest_p = model.score(coord)
            print("The model predicts [" + pred_char + " ] at " + str(coord) + " with p = "+ str(highest_p))

    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()


if __name__ == '__main__':

    # print(levenshteinDistance("the quick brown","th quick brpown"))
    # text_entry_metric("Eric_sitting")
    # text_entry_metric("Eric_walking")

    # text_entry_metric("all_sitting_data")
    # text_entry_metric("all_walking_data")

    # char_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
    #              'u', 'v', 'w', 'x', 'y', 'z', ' ']
    # char_dict = create_touch_data_dict("Eric_sitting_data")
    # construct_gaussian_models(char_dict)
    # convert_to_numpy(char_dict['a'])
    # print(char_dict)
    # # print(char_dict['t'])
    # for char in char_list:
    #     show_char_pattern(char,char_dict,)
    # draw_center_point_on_image()

    #

    # # generate_viz("walking")
    # # plot_pattern("walking")
    # mu = np.array([103.74646775265957, 1665.556011944339])
    # Sigma = np.array([[812.59797956, 132.85392401],
    #    [132.85392401, 628.5496348 ]])
    #
    # p = multivariate_gaussian(coord, mu, Sigma)
    #
    # print(p)

    # this is for demo
    sitting_models = build_model_and_generate_viz("sitting")
    # interactive_demo()
