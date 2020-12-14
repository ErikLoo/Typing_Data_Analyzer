import numpy as np
import pandas as pd
import glob
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Ellipse
import numpy as np
import pickle
from matplotlib.colors import LogNorm
from sklearn import mixture

class Touch_model:
    def __init__(self,model_para):
        self.model_para = model_para

    def multivariate_gaussian(self,pos, mu, Sigma):
        """Return the multivariate Gaussian distribution on array pos.

        pos is an array constructed by packing the meshed arrays of variables
        x_1, x_2, x_3, ..., x_k into its _last_ dimension.

        """
        pos = np.array(pos)
        n = mu.shape[0]
        Sigma_det = np.linalg.det(Sigma)
        Sigma_inv = np.linalg.inv(Sigma)
        N = np.sqrt((2 * np.pi) ** n * Sigma_det)
        # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
        # way across all the input variables.
        fac = np.einsum('...k,kl,...l->...', pos - mu, Sigma_inv, pos - mu)

        return np.exp(-fac / 2) / N

    def score(self, coord):
        # build a model from the dict
        highest_p = 0
        pred_char = ''

        for char in self.model_para.keys():
            mean = self.model_para[char]['mean']
            cov = self.model_para[char]['cov']
            prob = self.multivariate_gaussian(coord, mean, cov)
            # print("p(" + char + "|" + str(coord) + ") : " + str(prob))
            if prob >= highest_p:
                highest_p = prob
                pred_char = char

        return pred_char, highest_p

    def score_per_char(self,char,coord):
        # build a model from the dict

        mean = self.model_para[char]['mean']
        cov = self.model_para[char]['cov']
        prob = self.multivariate_gaussian(coord, mean, cov)
        return prob


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

    if len(coord_data)==0:
        print("start time: " + str(start_t) + " | end time: " + str(end_t))

    return [coord_data[int(len(coord_data)/2)],coord_data[-1]]


def build_model_and_generate_viz(f_name):
    char_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                 'u', 'v', 'w', 'x', 'y', 'z', ' ']
    char_dict = create_touch_data_dict(f_name)

    return construct_gaussian_models(char_dict,None)


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


if __name__ == '__main__':
    # train the model
    model_sitting = build_model_and_generate_viz("all_sitting_data_training")
    touch_model_sitting = Touch_model(model_sitting)

    f = open('TM.pickle', 'wb')
    pickle.dump(touch_model_sitting, f)
    f.close()

    print("Touch model trained on sitting data saved to file")

    model_walking = build_model_and_generate_viz("all_walking_data_training")
    touch_model_walking = Touch_model(model_walking)

    f = open('TM2.pickle', 'wb')
    pickle.dump(touch_model_walking, f)
    f.close()

    print("Touch model trained on walking data saved to file")

    model_comb = build_model_and_generate_viz("all_combined_data_training")
    touch_model_comb = Touch_model(model_comb)

    f = open('TM3.pickle', 'wb')
    pickle.dump(touch_model_comb, f)
    f.close()

    print("Touch model trained on comb data saved to file")

