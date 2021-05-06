from utility_funcs import *
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Ellipse
import pickle


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

        # print(pos)
        # print(mu)
        # print(Sigma)

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

        # print("touch model")
        # print(coord)
        # print(mean)
        # print(cov)

        prob = self.multivariate_gaussian(coord, mean, cov)
        return prob


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

                    actual_coords = get_touch_data(df_touch['time'], df_touch, start_t, end_t)
                    center_coords = np.array(true_coord_dict[char.lower()])

                    dist = np.linalg.norm(center_coords - actual_coords)

                    if dist <= 1.5*165:
                        '''
                        Get rid of the data that are far out of range 1.5 times the height
                        '''
                        if char not in char_touch_dict.keys():
                            char_touch_dict[char] = [actual_coords]
                        else:
                            char_touch_dict[char].append(actual_coords)

            else:
                # invalid correction operation
                pass

    return char_touch_dict


def build_model_and_generate_viz(f_name):
    char_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                 'u', 'v', 'w', 'x', 'y', 'z', ' ']
    char_dict = create_touch_data_dict(f_name)


    return construct_gaussian_models(char_dict,f_name)


def construct_gaussian_models(char_dict,f_name):
    # fit a 2D gaussian model for each key
    # plot the mean and variance ellipse on the keyboard layout
    img = mpimg.imread('keyboard_screen_shot.jpg')
    imgplot = plt.imshow(img)
    # convert data to numpy
    char_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                 'u', 'v', 'w', 'x', 'y', 'z', ' ']

    # create a gaussian model for every character

    if 'sitting' in f_name.split('_'):
        my_color = 'red'
    else:
        my_color = 'green'


    model_dict = {}

    for char in char_list:
        if char in char_dict.keys():
            # XY = convert_to_numpy(char_dict[char])

            XY = np.array(char_dict[char])

            x = XY[:, 0]
            y = XY[:, 1]

            # change the covariance terms to zero
            cov = np.cov(x, y)
            cov[0][1],cov[1][0] = 0,0

            lambda_, v = np.linalg.eig(cov)
            lambda_ = np.sqrt(lambda_)

            # plot the ellipse on the image
            ax = plt.gca()

            plt.plot(np.mean(x), np.mean(y), ".", markersize=1, color=my_color)
            first_v = v[:, 0]
            slope = first_v[1] / first_v[0]

            ax.add_patch(Ellipse(xy=(np.mean(x), np.mean(y)),
                                 width=lambda_[0] * 2 * 2, height=lambda_[1] * 2 * 2,
                                 linewidth=1,
                                 facecolor='none',
                                 edgecolor=my_color,
                                 angle=np.rad2deg(np.arctan(slope))))

            # save the model parameter in a dictionary
            model_dict[char] = {}
            model_dict[char]['mean'] = np.array([np.mean(x), np.mean(y)])
            model_dict[char]['cov'] = cov


            # print("Generated plot for " + char)
    # plt.scatter(x, y)
    print("Generated viz for " + f_name)
    plt.savefig(f_name+'_2D_Gauss.png', dpi=200)
    return model_dict


def train_touch_model():

    f = open('TM_W.pickle', 'wb')
    pickle.dump( Touch_model(build_model_and_generate_viz("walking_data_training")), f)
    f.close()


if __name__ == '__main__':

    train_touch_model()


