import numpy as np
import pandas as pd
import glob

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

            if isinstance(char,str):
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



def viz_touch_data(folder_name):

    char_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                 'u', 'v', 'w', 'x', 'y', 'z', ' ']

    char_touch_dict = {}

    for fname in glob.glob(folder_name + '/*.xls'):
        # print(fname)
        wrd_count = 1
        duration = 0
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
                    if char not in char_touch_dict.keys():
                        char_touch_dict[char] = [coord_data]
                    else:
                        char_touch_dict[char].append(coord_data)

            else:
                # invalid correction operation
                pass

        print(char_touch_dict)

def get_touch_data(df_touch_time,df_touch,start_t, end_t):
    coord_data = []
    for i, t in df_touch_time.iteritems():
        if t>=start_t and t<=end_t:
            x = df_touch['x'].iloc[i]
            y = df_touch['y'].iloc[i]
            coord_data.append((x,y))

    # only return the first and last entry for interpolation
    return [coord_data[0],coord_data[-1]]

if __name__ == '__main__':

    # print(levenshteinDistance("the quick brown","th quick brpown"))
    text_entry_metric("Eric_sitting")
    text_entry_metric("Eric_walking")
    # text_entry_metric("Steve_sitting")
    text_entry_metric("Chris_sitting")
    text_entry_metric("Chris_walking")

    text_entry_metric("Elise_sitting")
    text_entry_metric("Elise_walking")

    # viz_touch_data("Typing_data_test")
