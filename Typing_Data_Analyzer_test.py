import warnings
from utility_funcs import *
from touch_model_train import Touch_model
from motion_model_train import Motion_model
from Bayes_regression_model_train import BLR_model

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


def text_entry_metric_corrected(folder_name,touch_model_name,motion_model_name,BLR_model_name):

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
        df_accel = pd.read_excel(fname, sheet_name='accel_data')
        df_gyro = pd.read_excel(fname, sheet_name='gyro_data')




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

                # start_t = end_t - 100

                coord_data = get_touch_data(df_touch['time'], df_touch, start_t, end_t)
                motion_data = get_motion_data(df_accel, df_gyro, start_t - 100, start_t)

                actual_coords = np.array(coord_data[0])

                if char == 'Space':
                    char = ' '

                if char == ' ':
                    pred_char = ' '
                    tm_char = ' '
                else:
                    pred_char, highest_p = predict_char(actual_coords,t_model_str,motion_data,touch_model_name,motion_model_name,BLR_model_name)
                    # pred_char, highest_p = predict_char(actual_coords,typed_str,touch_model_name)

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
            if file_count%5==0:
                print("Inten str: " + int_str)
                print("Typed str: " + typed_str)
                print("T_mod str: " + t_model_str)
                print("Combi str: " + correct_str + "\n")

    name_model = " "
    if touch_model_name!=None:
        name_model = "touch model"

    if BLR_model_name!=None:
        name_model = "BLR_model"

    print(folder_name + "_avg uncorrected error rate revised by LM + " + name_model + " : " + str(sum(err_rate_list)/len(err_rate_list)))

    wpm = tlt_wrd_count/(tlt_duration/1000/60)

    print(folder_name + "_total wpm:  " + str(wpm) + " words per minute")
    print(" ")


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


def predict_char(coord,context,motion,touch_model_name,motion_model_name,BLR_model_name):

    # import the touch model
    if touch_model_name != None:
        f = open(touch_model_name, 'rb')
        touch_model = pickle.load(f)
        f.close()
    else:
        touch_model = None

    # import the language model
    f = open('KNLM.pickle', 'rb')
    KNL_model = pickle.load(f)
    f.close()

    # import the motion model
    if motion_model_name != None:
        f = open(motion_model_name, 'rb')
        motion_model = pickle.load(f)
        f.close()
    else:
        # print("motion_model")
        motion_model = None

        # import the motion model
    if BLR_model_name != None:
        f = open(BLR_model_name, 'rb')
        blr_model = pickle.load(f)
        f.close()
    else:
        blr_model = None

    highest_p = 0
    pred_char = ''
    char_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                 'u', 'v', 'w', 'x', 'y', 'z', ' ']

    p_mm = 1
    p_blrm = 1
    p_tm = 1
    for char in char_list:
        if touch_model_name != None:
            p_tm = touch_model.score_per_char(char,coord)

        # p_lm = KNL_model.score(char,list(context.split(" ")[-1]))
        p_lm = KNL_model.score(char,list(context))

        if motion_model_name != None:
            p_mm = motion_model.score_per_char(char,motion)


        if BLR_model_name != None:
           p_blrm = blr_model[char].score(motion,coord)

        p_combined = p_tm*p_lm*p_blrm

        # print("char: " + char + "| context: " + context +  "| p_tm: " + str(p_tm) + "| p_lm: " + str(p_lm))

        if p_combined>highest_p:
            highest_p = p_combined
            pred_char = char

    # print(str(touch_model==None) + " | " + str(blr_model==None))

    return pred_char,highest_p




if __name__ == '__main__':
    warnings.simplefilter('error')
    #
    # text_entry_metric("all_sitting_data_testing")
    # text_entry_metric("all_walking_data_testing")

    # text_entry_metric_corrected("all_sitting_data_testing","TM.pickle")
    # text_entry_metric_corrected("all_sitting_data_testing","TM2.pickle")
    # text_entry_metric_corrected("all_walking_data_testing","TM.pickle")
    # text_entry_metric_corrected("all_walking_data_testing","TM2.pickle")
    # text_entry_metric_corrected("all_combined_data_testing","TM3.pickle")
    # text_entry_metric_corrected("all_sitting_data_testing","TM3.pickle")
    # text_entry_metric_corrected("all_walking_data_testing","TM3.pickle")
    # text_entry_metric_corrected("all_combined_data_testing", "TM.pickle")
    # text_entry_metric_corrected("all_combined_data_testing", "TM2.pickle")
    # text_entry_metric_corrected("all_combined_data_testing", "TM3.pickle")

    # text_entry_metric("all_sitting_data_testing2")
    # # text_entry_metric_corrected("all_sitting_data_testing2","TM.pickle","MM.pickle")
    # text_entry_metric_corrected("all_sitting_data_testing2","TM.pickle",motion_model_name=None)

    text_entry_metric("all_walking_data_testing")
    text_entry_metric_corrected("all_walking_data_testing",touch_model_name=None,motion_model_name=None,BLR_model_name = "BLR_M.pickle")
    text_entry_metric_corrected("all_walking_data_testing",touch_model_name="TM.pickle",motion_model_name=None,BLR_model_name=None)



    # text_entry_metric_corrected("Eric_sitting_data_3", "TM.pickle")

