import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import json



def moving_average(df, n,fname):
    """Calculate the moving average for a given window.

    :param df: pandas.DataFrame
    :param n: window size
    :return: pandas.DataFrame
    """

    MA_x = pd.Series(df['x'].rolling(n, min_periods=n).mean(), name='x_MA_' + str(n))
    MA_y = pd.Series(df['y'].rolling(n, min_periods=n).mean(), name='y_MA_' + str(n))
    MA_z = pd.Series(df['z'].rolling(n, min_periods=n).mean(), name='z_MA_' + str(n))


    df = df.join(MA_x).join(MA_y).join(MA_z)

    # save the plot
    df_plt = df[['t_0','x_MA_' + str(n),'y_MA_' + str(n),'z_MA_' + str(n)]]
    ax = df_plt.plot(x='t_0', y=['x_MA_' + str(n),'y_MA_' + str(n),'z_MA_' + str(n)], kind='line')
    # ymin, ymax = ax.get_ylim()
    # ax.vlines(x='t_0', ymin=ymin, ymax=ymax - 1, color='r')
    # plt.show()
    # plot the lines to separate different regions of the plot based on the json file

    plt.savefig(sfname+'/plot_mean_win = ' + str(n))

    return df

def moving_median(df, n,fname,ssfname,j_file):
    """Calculate the moving median for a given window.

    :param df: pandas.DataFrame
    :param n: window size
    :return: pandas.DataFrame
    """
    MA_x = pd.Series(df['x'].rolling(n, min_periods=n).median(), name='x_ME_' + str(n))
    MA_y = pd.Series(df['y'].rolling(n, min_periods=n).median(), name='y_ME_' + str(n))
    MA_z = pd.Series(df['z'].rolling(n, min_periods=n).median(), name='z_ME_' + str(n))

    df = df.join(MA_x).join(MA_y).join(MA_z)

    # save the plot

    df_plt = df[['t_1','x_ME_' + str(n),'y_ME_' + str(n),'z_ME_' + str(n)]]
    ax = df_plt.plot(x='t_1', y=['x_ME_' + str(n),'y_ME_' + str(n),'z_ME_' + str(n)], kind='line')

    # plot the vertical axes
    for seg in j_file:
        if seg['label1_2'].strip() in 'abcdefg':
            t_start = seg['time1_2']
            t_end = seg['time2_1']
            ax.axvline(t_start, color="red", linestyle="--")
            ax.axvline(t_end, color="blue", linestyle="--")
            pos = t_start+ (t_end-t_start)/4
            ax.text(x=pos,y=0,s = str(round((t_end-t_start)/1000))+'s', weight = 'bold')


    fig_name = ssfname.split('\\')[-1].split('.')[0] + "_win = " + str(n)
    plt.savefig(sfname+'/' + fig_name)

    print("created " + fname + '\\' + fig_name + " ...")

    # go through plot by pressing a button
    # plt.draw()
    # plt.waitforbuttonpress(0)
    # plt.close()


    return df


for fname in glob.glob('*'):
    # if cvs file is not found
    if len(glob.glob(fname+'/*.csv'))==0:
        for sfname in glob.glob(fname+'/*'):
            with open(sfname+'/labels.json') as f:
                j_file = json.load(f)


            for ssfname in glob.glob(sfname+'/*.csv'):
                #apply the function in here
                print("Reading " + ssfname + " ...")
                df = pd.read_csv(ssfname)

                # print(df.head)
                if list(df.columns)!=['t_0','t_1','x','y','z'] and len(list(df.columns))<=5:
                    # add a header is there is none
                    df.columns = ['t_0', 't_1', 'x', 'y', 'z']
                else:
                    # Clear the old value
                    df = df[['t_0', 't_1', 'x', 'y', 'z']]

                # Use t_1 as the reference time

                # print(df.head)

                # add filters in here
                df = moving_median(df,8,sfname,ssfname,j_file)
                df = moving_median(df,16,sfname,ssfname,j_file)
                # df = moving_median(df,32,sfname)

                # print(df.head)
                df.to_csv(ssfname,index=False)

                # plot the data points

                # df_plt = df[['t_0','x',]]
                # df_plt.plot(x='t_0', y=['x','y','z'], kind='line')
                # # plt.show()
                # plt.savefig(sfname+'/plot_x_y_z')
    else:
        pass


