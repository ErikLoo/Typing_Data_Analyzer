from utility_funcs import *
from sklearn.covariance import empirical_covariance
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from touch_model_train import Touch_model
import matplotlib.pyplot as plt

import os.path
from os import path

class BLR_model:
    def __init__(self,model_para):
        self.model_para = model_para

    def multivariate_gaussian(self,pos, mu, Sigma):
        """Return the multivariate Gaussian distribution on array pos.

        pos is an array constructed by packing the meshed arrays of variables
        x_1, x_2, x_3, ..., x_k into its _last_ dimension.

        """
        x = np.array(pos)
        n = mu.shape[0]

        Sigma_det = np.linalg.det(Sigma)

        Sigma_inv = np.linalg.inv(Sigma)

        N = np.sqrt((2 * np.pi) ** n * Sigma_det)
        # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
        # way across all the input variables.
        fac = np.einsum('...k,kl,...l->...', x - mu, Sigma_inv, x - mu)
        # print("a: " + str(-fac / 2) + " b: " + str(N))
        return np.exp(-fac / 2) / N

        # use log instead of exp to avoid overflow
        # return -fac/2 - np.log(N)

    def score(self,motion,coord):
        # build a model from the dict
        x = coord[0]
        y = coord[1]
        x_model = self.model_para[0]
        y_model = self.model_para[1]
        var_xy = self.model_para[2]

        x_mean,x_std = x_model.predict([motion],return_std=True)
        y_mean,y_std = y_model.predict([motion],return_std=True)

        # we naively assume x and y are indepently distributed

        cov = np.array([[x_std[0]**2,var_xy],[var_xy,y_std[0]**2]])
        mean = np.array([x_mean[0],y_mean[0]])
        pos = np.array([x,y])

        prob = self.multivariate_gaussian(pos,mean,cov)

        return prob


def create_char_motion_touch_dict(folder_name):
    # create features vectors and predictions
    '''
    create a dictionary storing {char:{"motion":[speed_x,speed_y,speed_z],"coord":[[x,y]]}}
    '''
    char_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                 'u', 'v', 'w', 'x', 'y', 'z', ' ']

    char_dict = {}

    true_coord_dict = true_coord()

    for fname in glob.glob(folder_name + '/*.xls'):
        # print(fname)
        # wrd_count = 1
        # duration = 0
        df_time = pd.read_excel(fname, sheet_name='input_time')
        df_touch = pd.read_excel(fname, sheet_name='touch_data')
        df_accel = pd.read_excel(fname, sheet_name='accel_data')
        df_gyro = pd.read_excel(fname, sheet_name='gyro_data')

        old_start_t=0

        for i, char in df_time['intended character'].iteritems():
            if char == 'Space':
                char = ' '

            if isinstance(char, str):
                if char.lower() in char_list:
                    start_t = df_time['start time'].iloc[i]
                    end_t = df_time['end time'].iloc[i]

                    # only record the time segment about 0.1 before the start time
                    # start_t = end_t - 100
                    if i==0:
                        old_end_t = end_t-100
                        old_start_t = start_t-100

                    actual_coords = get_touch_data(df_touch['time'],df_touch,start_t,end_t)

                    motion_data = get_motion_data(df_accel,df_gyro,old_start_t,start_t)

                    old_end_t = end_t
                    old_start_t = start_t
                    # print(coord_data)
                    center_coords = np.array(true_coord_dict[char.lower()])


                    dist = np.linalg.norm(center_coords-actual_coords)

                    if dist <= 1.5*165:
                        '''
                        Get rid of the data that are far out of range 1.5 times the button height
                        '''
                        if char not in char_dict.keys():
                            char_dict[char] = [[motion_data],[actual_coords]]
                        else:
                            char_dict[char][0].append(motion_data)
                            char_dict[char][1].append(actual_coords)


            else:
                # invalid correction operation
                pass

    return char_dict


def build_motion_model(f_name):
    '''
    create a bayes ridge regression model: P(touch_point|accel,Character)
    :return:
    '''
    # kernel = DotProduct() + WhiteKernel()

    char_dict = create_char_motion_touch_dict(f_name)

    char_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                 'u', 'v', 'w', 'x', 'y', 'z', ' ']

    # print(char_dict)

    model_dict = {}

    print("start building model")
    for char in char_list:
        if char in char_dict.keys():
            X = np.array(char_dict[char][0])
            Y = np.array(char_dict[char][1])

            y_1 = Y[:,0]
            y_2 = Y[:,1]


            blr_model_1 = linear_model.BayesianRidge(compute_score=True)
            blr_model_1.fit(X,y_1)

            blr_model_2 = linear_model.BayesianRidge(compute_score=True)
            blr_model_2.fit(X,y_2)

            # print("char: " + char + " | x_lambda: " + str(x_lambda) + "| y_lambda: " + str(y_lambda))

            blr_model_tlt = BLR_model([blr_model_1,blr_model_2,0])
            model_dict[char] = blr_model_tlt

    return model_dict



def train_BLR_model():
    f = open('BLR_M_W.pickle', 'wb')
    pickle.dump(build_motion_model("walking_data_training"), f)
    f.close()


def viz_BLR_model(BLR_model_name):

    '''
    visualize the distrbution of touch points affected by the motion
    '''

    f = open(BLR_model_name, 'rb')
    blr_model = pickle.load(f)
    f.close()
    char_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                 'u', 'v', 'w', 'x', 'y', 'z', ' ']

    x_motion_list= np.array([[-1, 0, 0],[0,0,0],[1,0,0]])
    y_motion_list= np.array([[0, -1, 0],[0,0,0],[0,1,0]])
    z_motion_list= np.array([[0,0,-1],[0,0,0],[0,0,1]])

    query_list = np.array([x_motion_list,y_motion_list,z_motion_list])
    name_list= ['X','Y','Z']
    color_list = ['blue','red','orange']

    # print(query_list)
    legend_list = []

    for j, motion_list in enumerate(query_list):
        img = mpimg.imread('keyboard_screen_shot.jpg')
        imgplot = plt.imshow(img)
        str_1 = str(motion_list[0])
        str_2 = str(motion_list[1])
        str_3 = str(motion_list[2])
        for i, motion in enumerate(motion_list):

            # print(motion)
            for char in char_list:
                x_model = blr_model[char].model_para[0]
                y_model = blr_model[char].model_para[1]

                x_mean, x_std = x_model.predict([motion], return_std=True)
                y_mean, y_std = y_model.predict([motion], return_std=True)
                var_xy = blr_model[char].model_para[2]

                # we naively assume x and y are indepently distributed
                cov = np.array([[x_std[0] ** 2, var_xy], [var_xy, y_std[0] ** 2]])
                mean = np.array([x_mean[0], y_mean[0]])

                lambda_, v = np.linalg.eig(cov)
                lambda_ = np.sqrt(lambda_)

                # plot the ellipse on the image
                ax = plt.gca()

                plt.plot(mean[0], mean[1], ".", markersize=0.5, color=color_list[i])

                first_v = v[:, 0]
                slope = first_v[1] / first_v[0]

                ax.add_patch(Ellipse(xy=(mean[0], mean[1]),
                                     width=lambda_[0] * 2 * 2, height=lambda_[1] * 2 * 2,
                                     linewidth=0.5,
                                     facecolor='none',
                                     edgecolor=color_list[i],
                                     angle=np.rad2deg(np.arctan(slope))))


        print("Generated viz of BLR model")
        plt.text(850, 500, str_1, fontsize=5, color='blue')
        plt.text(850, 550, str_2, fontsize=5, color='red')
        plt.text(850, 600, str_3, fontsize=5, color='orange')
        plt.savefig('BLR_model_viz/' + BLR_model_name + '_' + name_list[j] + '.png', dpi=200)
        plt.clf()



if __name__ == '__main__':
    train_BLR_model()

