

from utility_funcs import *
from sklearn.covariance import empirical_covariance
from sklearn.preprocessing import MinMaxScaler

class Motion_model:
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

    def score(self, motion):
        # build a model from the dict
        highest_p = 0
        pred_char = ''

        for char in self.model_para.keys():
            mean = self.model_para[char]['mean']
            cov = self.model_para[char]['cov']
            prob = self.multivariate_gaussian(motion, mean, cov)
            # print("p(" + char + "|" + str(coord) + ") : " + str(prob))
            if prob >= highest_p:
                highest_p = prob
                pred_char = char

        return pred_char, highest_p

    def score_per_char(self,char,motion):
        # build a model from the dict

        mean = self.model_para[char]['mean']
        cov = self.model_para[char]['cov']
        # print('char: ' +char)
        prob = self.multivariate_gaussian(motion, mean, cov)
        return prob


def create_char_motion_dict(folder_name):
    # create features vectors and predictions
    '''
    Visualize the data on an image
    Show the starting and the end point of each key
    '''
    char_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                 'u', 'v', 'w', 'x', 'y', 'z', ' ']

    char_motion_dict = {}

    true_coord_dict = true_coord()

    for fname in glob.glob(folder_name + '/*.xls'):
        # print(fname)
        # wrd_count = 1
        # duration = 0
        df_time = pd.read_excel(fname, sheet_name='input_time')
        df_touch = pd.read_excel(fname, sheet_name='touch_data')
        df_accel = pd.read_excel(fname, sheet_name='accel_data')
        df_gyro = pd.read_excel(fname, sheet_name='gyro_data')


        for i, char in df_time['intended character'].iteritems():
            if char == 'Space':
                char = ' '

            if isinstance(char, str):
                if char.lower() in char_list:
                    start_t = df_time['start time'].iloc[i]
                    end_t = df_time['end time'].iloc[i]

                    # only record the time segment about 0.1 before the start time
                    # start_t = end_t - 100

                    coord_data = get_touch_data(df_touch['time'],df_touch,start_t,end_t)
                    # might not be 0.1s before the start time
                    # motion_data = get_motion_data(df_accel,df_gyro,start_t-100,start_t)
                    # motion_data = get_motion_data(df_accel,df_gyro,start_t-100,start_t)
                    motion_data = get_motion_data(df_accel,df_gyro, start_t - 100, start_t)

                    # print(coord_data)
                    center_coords = np.array(true_coord_dict[char.lower()])

                    # get the mid value not the first value
                    actual_coords = np.array(coord_data[0])

                    dist = np.linalg.norm(center_coords-actual_coords)

                    if dist <= 1.5*165:
                        '''
                        Get rid of the data that are far out of range 1.5 times the button height
                        '''
                        if char not in char_motion_dict.keys():
                            char_motion_dict[char] = [motion_data]
                        else:
                            char_motion_dict[char].append(motion_data)

            else:
                # invalid correction operation
                pass

    return char_motion_dict



def construct_motion_gaussian_models(char_dict):
    # fit a 2D gaussian model for each key
    # plot the mean and variance ellipse on the keyboard layout
    # img = mpimg.imread('keyboard_screen_shot.jpg')
    # imgplot = plt.imshow(img)
    # convert data to numpy
    char_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                 'u', 'v', 'w', 'x', 'y', 'z', ' ']

    # create a gaussian model for every character

    model_dict = {}

    scaler = MinMaxScaler()

    for char in char_list:
        if char in char_dict.keys():

            # print(char)
            # print(np.array(char_dict[char]).T.shape)

            dim = np.array(char_dict[char]).shape
            X = np.array(char_dict[char])
            scaler.fit(X)
            # X_t = scaler.transform(X)
            X_t = X
            dim = X_t.shape
            mu = np.mean(X_t,axis=0)

            # print(np.array(char_dict[char].shape))
            if dim[0]>1:
                # cannot use np.cov when there is only one data point
                # cov = np.cov(np.array(char_dict[char]).T)
                cov = empirical_covariance(X_t)
                # print(cov)
            else:
                cov = np.zeros((dim[1],dim[1]))
            # save the model parameter in a dictionary
            model_dict[char] = {}
            model_dict[char]['mean'] = mu

            model_dict[char]['cov'] = cov

            sigma_det = np.linalg.det(cov)

            # by definition sigma_det should not be zero!!!
            # as the covariance matrix is semi-positive definite
            if sigma_det<=0:
                print(char + " : " + str(sigma_det))
                print(X_t.shape)
                print(cov)
            # print(cov.shape)
            # print("Generated plot for " + char)
    # plt.scatter(x, y)
    # plt.savefig(posture + '_training_data_pattern/' + posture + '_2D_Gauss.png', dpi=200)
    return model_dict


def build_motion_model(f_name):
    '''
    create p(motion | C)
    Assume the motion is normally distributed
    :return:
    '''
    char_dict = create_char_motion_dict(f_name)
    model_dict = construct_motion_gaussian_models(char_dict)

    # print(model_dict)
    return Motion_model(model_dict)


if __name__ == '__main__':
    # warnings.simplefilter('error')

    # motion_model = build_motion_model("all_sitting_data_training")
    #
    # f = open('MM.pickle', 'wb')
    # pickle.dump(motion_model, f)
    # f.close()

    motion_model = build_motion_model("Eric_walking_data_2")

    f = open('MM_walking.pickle', 'wb')
    pickle.dump(motion_model, f)
    f.close()