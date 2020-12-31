# Step 1: plot all three axes on the same graph
# Step 2: label the start and end time of each tap
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt

def moving_average_accel(df,df_it,n,fname):
    """Calculate the moving average for a given window.

    :param df: pandas.DataFrame
    :param n: window size
    :return: pandas.DataFrame
    """

    MA_x = pd.Series(df['accel_x'].rolling(n, min_periods=n).mean(), name='accel_x_MA_' + str(n))
    MA_y = pd.Series(df['accel_y'].rolling(n, min_periods=n).mean(), name='accel_y_MA_' + str(n))
    MA_z = pd.Series(df['accel_z'].rolling(n, min_periods=n).mean(), name='accel_z_MA_' + str(n))

    MA_x-=4
    MA_z+=4

    df = df.join(MA_x).join(MA_y).join(MA_z)

    df_plt = df[['time','accel_x_MA_' + str(n),'accel_y_MA_' + str(n),'accel_z_MA_' + str(n)]]
    ax = df_plt.plot(x='time', y=['accel_x_MA_' + str(n),'accel_y_MA_' + str(n),'accel_z_MA_' + str(n)], kind='line',xticks = df_plt['time'][::1000],figsize= (20,10))
    old_t_start = 0
    rep_count = 0

    for i, char in df_it['intended character'].iteritems():
        t_start = df_it['start time'].iloc[i]
        t_end = df_it['end time'].iloc[i]

        if t_start == old_t_start:
            rep_count =rep_count +1

        ax.axvline(t_start, color="red", linestyle="--")
        ax.axvline(t_end, color="blue", linestyle="--")
        pos = t_start + (t_end - t_start) / 4
        if char == 'Space':
            char = "|-|"
        ax.text(x=pos, y=0, s=char, weight='bold', fontsize=10)
        old_t_start = t_start

    fig_name = fname.split('\\')[-1].split('.')[0] + "_win = " + str(n) + "_accel_MA"
    plt.savefig(fname.split('\\')[0] + '/' + fig_name)


def moving_median_accel(df,df_it,n,fname):
    """Calculate the moving average for a given window.

    :param df: pandas.DataFrame
    :param n: window size
    :return: pandas.DataFrame
    """

    MA_x = pd.Series(df['accel_x'].rolling(n, min_periods=n).median(), name='accel_x_MM_' + str(n))
    MA_y = pd.Series(df['accel_y'].rolling(n, min_periods=n).median(), name='accel_y_MM_' + str(n))
    MA_z = pd.Series(df['accel_z'].rolling(n, min_periods=n).median(), name='accel_z_MM_' + str(n))


    df = df.join(MA_x).join(MA_y).join(MA_z)

    df_plt = df[['time','accel_x_MM_' + str(n),'accel_y_MM_' + str(n),'accel_z_MM_' + str(n)]]
    ax = df_plt.plot(x='time', y=['accel_x_MM_' + str(n),'accel_y_MM_' + str(n),'accel_z_MM_' + str(n)], kind='line',xticks = df_plt['time'][::1000],figsize= (20,10))
    old_t_start = 0
    rep_count = 0

    for i, char in df_it['intended character'].iteritems():
        t_start = df_it['start time'].iloc[i]
        t_end = df_it['end time'].iloc[i]

        if t_start == old_t_start:
            rep_count =rep_count +1

        ax.axvline(t_start, color="red", linestyle="--")
        ax.axvline(t_end, color="blue", linestyle="--")
        pos = t_start + (t_end - t_start) / 4
        if char == 'Space':
            char = "|-|"
        ax.text(x=pos, y=0, s=char, weight='bold', fontsize=10)
        old_t_start = t_start

    fig_name = fname.split('\\')[-1].split('.')[0] + "_win = " + str(n) + "_accel_MM"
    plt.savefig(fname.split('\\')[0] + '/' + fig_name)


def moving_average_gyro(df, df_it, n, fname):
        """Calculate the moving average for a given window.

        :param df: pandas.DataFrame
        :param n: window size
        :return: pandas.DataFrame
        """

        MA_x = pd.Series(df['gyro_x'].rolling(n, min_periods=n).mean(), name='gyro_x_MA_' + str(n))
        MA_y = pd.Series(df['gyro_y'].rolling(n, min_periods=n).mean(), name='gyro_y_MA_' + str(n))
        MA_z = pd.Series(df['gyro_z'].rolling(n, min_periods=n).mean(), name='gyro_z_MA_' + str(n))

        # add 1 or subtract 1 to separate the three gyro axes for better viz
        MA_x-=1
        MA_z+=1

        df = df.join(MA_x).join(MA_y).join(MA_z)

        df_plt = df[['time', 'gyro_x_MA_' + str(n), 'gyro_y_MA_' + str(n), 'gyro_z_MA_' + str(n)]]
        ax = df_plt.plot(x='time', y=['gyro_x_MA_' + str(n), 'gyro_y_MA_' + str(n), 'gyro_z_MA_' + str(n)],
                         kind='line', xticks=df_plt['time'][::1000], figsize=(20, 10))
        old_t_start = 0
        rep_count = 0

        for i, char in df_it['intended character'].iteritems():
            t_start = df_it['start time'].iloc[i]
            t_end = df_it['end time'].iloc[i]

            if t_start == old_t_start:
                rep_count = rep_count + 1

            ax.axvline(t_start, color="red", linestyle="--")
            ax.axvline(t_end, color="blue", linestyle="--")
            pos = t_start + (t_end - t_start) / 4
            if char == 'Space':
                char = "|-|"
            ax.text(x=pos, y=0, s=char, weight='bold', fontsize=10)
            old_t_start = t_start

        fig_name = fname.split('\\')[-1].split('.')[0] + "_win = " + str(n) + "_gyro_MA"
        plt.savefig(fname.split('\\')[0] + '/' + fig_name)

        print("created " + fname.split('\\')[0] + '/' + fig_name + " ...")
        print("rep count: " + str(rep_count))
        # go through plot by pressing a button
        # plt.draw()
        # plt.waitforbuttonpress(0)
        # plt.close()

        return df


for fname in glob.glob('motion_data_viz/*.xls'):
    # print("Reading " + fname + " ...")
    df_it = pd.read_excel(fname,sheet_name='input_time')
    df_accel = pd.read_excel(fname,sheet_name='accel_data')
    df_gyro = pd.read_excel(fname,sheet_name='gyro_data')

    df = moving_average_accel(df_accel,df_it,20,fname)
    # df = moving_median_accel(df_accel,df_it,10,fname)


    # df = moving_average_gyro(df_gyro,df_it,10,fname)




