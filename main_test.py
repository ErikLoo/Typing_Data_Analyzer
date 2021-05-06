import warnings
from utility_funcs import *
from touch_model_train import Touch_model,train_touch_model
from motion_model_train import Motion_model,train_motion_model
from Bayes_regression_model_train import BLR_model,train_BLR_model
# from Bayes_regression_mix_model_train import BLR_mix_model,train_BLR_mix_model


def text_entry_metric(folder_name):
    '''
    calculate the letter-wise text entry accuracy
    before applying any models
    '''
    err_rate_list = []
    tlt_duration = 0
    tlt_wrd_count = 0
    char_list = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',' ']
    file_count = 0
    # num_file = len(glob.glob(folder_name+'/*.xls'))

    int_str_list = []
    typed_str_list = []


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

        if(len(int_str)==len(typed_str)):
            int_str_list+= list(int_str)
            typed_str_list+=list(typed_str)


        INF,ins_count,del_count,sub_count = levenshteinDistance(int_str,typed_str)


        C = max(len(int_str),len(typed_str)) - INF

        err_rate = INF/(INF+C)

        err_rate_list.append(err_rate)

        tlt_wrd_count = tlt_wrd_count + wrd_count

        duration = duration + df['end time'].iloc[-1] - df['start time'].iloc[0]
        tlt_duration  = tlt_duration + duration

        file_count = file_count + 1

        # print("Finished analyzing file " + str(file_count) + "/" + str(num_file))

    print(folder_name + "_avg uncorrected error rate:  " + str(sum(err_rate_list)/len(err_rate_list)))

    wpm = tlt_wrd_count/(tlt_duration/1000/60)

    print(folder_name + "_total wpm:  " + str(wpm) + " words per minute")

    # print("len of int str list: " + str(len(int_str_list)) + " | len of typed str list: " + str(len(typed_str_list)))
    mtx = generate_confusion_matrix(int_str_list,typed_str_list,"original_")
    print(" ")
    return sum(err_rate_list)/len(err_rate_list)

def text_entry_metric_corrected(folder_name,touch_model_name,motion_model_name,BLR_model_name,switch_lm):
    '''
    text entry accuracy after applying the models
    touch_model_name: name of the touch model to be used
    motion_model_name:name of the motion model to be used
    BLR_model_name: name of the bayes linear regression model to be used
    switch_lm: switch on/off the language model
    '''
    err_rate_list = []
    tlt_duration = 0
    tlt_wrd_count = 0

    file_count = 0

    char_list = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',' ']

    int_str_list = []
    correct_str_list = []

    for fname in glob.glob(folder_name+'/*.xls'):
        # print(fname)
        wrd_count = 1
        duration = 0
        df = pd.read_excel(fname,sheet_name='input_time')

        df_time = pd.read_excel(fname, sheet_name='input_time')
        df_touch = pd.read_excel(fname, sheet_name='touch_data')
        df_accel = pd.read_excel(fname, sheet_name='accel_data')
        df_gyro = pd.read_excel(fname, sheet_name='gyro_data')

        # "intend char": the character that should be entered
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

        # typed_char: the character actually entered
        typed_char = df['typed character']
        typed_str = ""
        correct_str = ""
        t_model_str = ""

        old_start_t=0

        for i, char in typed_char.iteritems():

            # get the coords
            # if the touch point is too far away
            if char != "Del":
                start_t = df_time['start time'].iloc[i]
                end_t = df_time['end time'].iloc[i]

                if i == 0:
                    old_end_t = end_t - 100
                    old_start_t = start_t - 100

                # the point of contact for the typed character
                touch_pt = get_touch_data(df_touch['time'], df_touch, start_t, end_t)
                # the motion data associated with typing the character
                motion_data = get_motion_data(df_accel, df_gyro, old_start_t, start_t)

                old_start_t = start_t

                if char == 'Space':
                    char = ' '

                if char == ' ':
                    pred_char = ' '
                    tm_char = ' '
                else:
                    # pred_char: the character predicted by the combination of models (touch model + motion model + language mmodel)
                    # I used tm_model_str as the context str for the language model because it produces better results than correct_str
                    pred_char, highest_p = predict_char(touch_pt,t_model_str,correct_str,motion_data,touch_model_name,motion_model_name,BLR_model_name,switch_lm)

                    # tm_char: the character predicted by the touch model only
                    tm_char, tm_p = predict_char_tm(touch_pt,touch_model_name)

                if isinstance(char, str):
                    if char.lower() in char_list:
                        typed_str = typed_str + char.lower()
                        correct_str = correct_str + pred_char.lower()
                        t_model_str = t_model_str + tm_char.lower()

                else:
                    # invalid correction operation
                    pass

            old_char = char

        if (len(int_str) == len(correct_str)):
            int_str_list += list(int_str)
            correct_str_list += list(correct_str)

        INF, ins_count, del_count, sub_count = levenshteinDistance(int_str,correct_str)


        C = max(len(int_str),len(typed_str)) - INF

        err_rate = INF/(INF+C)

        err_rate_list.append(err_rate)

        tlt_wrd_count = tlt_wrd_count + wrd_count

        duration = duration + df['end time'].iloc[-1] - df['start time'].iloc[0]
        tlt_duration  = tlt_duration + duration

        file_count = file_count + 1

        # print("Finished analyzing file " + str(file_count) + "/" + str(num_file))
        if int_str == typed_str == correct_str:
            pass
        else:
            if file_count%5==0:
                pass

    if touch_model_name=="":
        touch_model_name = "off"

    if BLR_model_name=="":
        BLR_model_name = "off"

    if motion_model_name=="":
        motion_model_name= "off"

    print(folder_name + "_avg uncorrected error rate Touch Model: " + touch_model_name + " | Motion Model: " + motion_model_name +" | BLR Model: " + BLR_model_name + " : " + str(sum(err_rate_list)/len(err_rate_list)))
    wpm = tlt_wrd_count/(tlt_duration/1000/60)
    print(folder_name + "_total wpm:  " + str(wpm) + " words per minute")
    mtx = generate_confusion_matrix(int_str_list,correct_str_list,"corrected")

    print(" ")
    return sum(err_rate_list)/len(err_rate_list)


def predict_char_tm(coord,touch_model_name):
    '''
    predict the next character based on the touch model only
    '''

    # import the touch model
    if touch_model_name != "":
        f = open(touch_model_name, 'rb')
        touch_model = pickle.load(f)
        f.close()
    else:
        # f = open('TM.pickle', 'rb')
        # touch_model = pickle.load(f)
        # f.close()
        return ' ',0.5

    highest_p = 0
    pred_char = ''
    char_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                 'u', 'v', 'w', 'x', 'y', 'z', ' ']

    for char in char_list:
        p_tm = touch_model.score_per_char(char,coord)
        p_lm = 1
        p_combined = p_tm

        if p_combined>highest_p:
            highest_p = p_combined
            pred_char = char

    return pred_char,highest_p


def predict_char(coord,context_tm,context_blrm,motion,touch_model_name,motion_model_name,BLR_model_name,switch_lm):
    '''
    predict the next character based on the combination of models (touch model + motion model + language)
    '''

    # import the touch model
    if touch_model_name != "":
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
    if motion_model_name != "":
        f = open(motion_model_name, 'rb')
        motion_model = pickle.load(f)
        f.close()
    else:
        # print("motion_model")
        motion_model = None

        # import the motion model
    if BLR_model_name != "":
        f = open(BLR_model_name, 'rb')
        blr_model = pickle.load(f)
        f.close()
    else:
        blr_model = None

    highest_p = 0
    pred_char = ''
    char_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                 'u', 'v', 'w', 'x', 'y', 'z', ' ']

    for char in char_list:
        if touch_model_name != "":
            p_tm = touch_model.score_per_char(char,coord)
        else:
            p_tm=0

        if switch_lm == 'on':

            p_lm_1 = KNL_model.score(char, list(context_tm))
            p_lm_2 = KNL_model.score(char, list(context_blrm))
        else:
            p_lm_1,p_lm_2 = 1,1


        if motion_model_name != "":
            p_mm = motion_model.score_per_char(char,motion)
        else:
            p_mm=1

        if BLR_model_name != "":
           p_blrm = blr_model[char].score(motion,coord)
        else:
            p_blrm = 0

        p_combined = 0.5*p_tm*p_lm_1 + 0.5*p_blrm*p_mm*p_lm_2


        if p_combined>highest_p:
            highest_p = p_combined
            pred_char = char

    return pred_char,highest_p



if __name__ == '__main__':
    warnings.simplefilter('error')

    print("Before applying any models:")
    text_entry_metric("walking_data_testing")

    # 
    train_touch_model()
    train_BLR_model()

    print("After applying models:")
    text_entry_metric_corrected("walking_data_testing", touch_model_name="TM_W.pickle", motion_model_name="",BLR_model_name="",switch_lm="on")
    text_entry_metric_corrected("walking_data_testing", touch_model_name="", motion_model_name="",BLR_model_name="BLR_M_W.pickle", switch_lm="on")





