from utility_funcs import *
from sklearn.covariance import empirical_covariance
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

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

        x_mean,x_std = x_model.predict([motion],return_std=True)
        y_mean,y_std = y_model.predict([motion],return_std=True)

        # we naively assume x and y are indepently distributed
        cov = np.array([[x_std[0]**2,0],[0,y_std[0]**2]])
        mean = np.array([x_mean[0],y_mean[0]])
        pos = np.array([x,y])

        # print('BLR')
        # print(pos)
        # print(mean)
        # print(cov)

        prob = self.multivariate_gaussian(pos,mean,cov)

        return prob


def create_char_motion_touch_dict(folder_name):
    # create features vectors and predictions
    '''
    {char:{"motion":[speed_x,speed_y,speed_z],"coord":[[x,y]]}}
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

                    motion_data = get_motion_data(df_accel,df_gyro, start_t - 100, start_t)

                    # print(coord_data)
                    center_coords = np.array(true_coord_dict[char.lower()])

                    # get the first value of the coord data at start_t
                    actual_coords = np.array(coord_data[0])

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
    create p(motion | C)
    Assume the motion is normally distributed
    :return:
    '''
    kernel = DotProduct() + WhiteKernel()

    char_dict = create_char_motion_touch_dict(f_name)

    char_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                 'u', 'v', 'w', 'x', 'y', 'z', ' ']



    model_dict = {}

    for char in char_list:
        if char in char_dict.keys():
            X = np.array(char_dict[char][0])
            Y = np.array(char_dict[char][1])

            # print(Y)

            y_1 = Y[:,0]
            y_2 = Y[:,1]

            blr_model_1 = linear_model.BayesianRidge()
            blr_model_1.fit(X,y_1)

            blr_model_2 = linear_model.BayesianRidge()
            blr_model_2.fit(X,y_2)

            # x_t = [X[0]]
            # y_mean_1,y_std_1 = blr_model_1.predict(x_t,return_std=True)
            # y_mean_2,y_std_2 = blr_model_2.predict(x_t,return_std=True)

            blr_model_tlt = BLR_model([blr_model_1,blr_model_2])
            model_dict[char] = blr_model_tlt

    return model_dict



if __name__ == '__main__':
    # warnings.simplefilter('error')

    motion_model = build_motion_model("all_walking_data_training")
    #
    f = open('BLR_M.pickle', 'wb')
    pickle.dump(motion_model, f)
    f.close()

    # motion_model = build_motion_model("Eric_walking_data_2")
    # #
    # f = open('MM_walking.pickle', 'wb')
    # pickle.dump(motion_model, f)
    # f.close()