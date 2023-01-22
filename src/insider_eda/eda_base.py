import os
import time
from datetime import datetime

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.pyplot import cm
from plotly.subplots import make_subplots
from statsmodels.graphics.tsaplots import month_plot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import quarter_plot
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.tools import diff
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import grangercausalitytests


class Exploratory_data_analysis:
    """
    This class is for Exploratory Data Analysis.

    This class can be used to perform some common EDA tasks on a given DataFrame.

    :param df: DataFrame containing the data
    :type df: pd.DataFrame
    :param target_name: The name of the target variable, if the data is not time series this parameter can be set as False
    :type target_name: str or bool
    :param time_series: Boolean indicating whether the data is time series
    :type time_series: bool
    :raises ValueError: If the DataFrame index is not a pandas datetime index when time_series is set to True
    """

    def __init__(self, df: pd.DataFrame, target_name=False, time_series=False):
        """
        The init method of the Exploratory_data_analysis class initializes the class with a DataFrame, target variable name, and a boolean value indicating whether the data is time-series.

        :param df: pandas DataFrame containing the data
        :type df: pd.DataFrame
        :param target_name: The name of the target variable, if the data is not time series this parameter can be set as False
        :type target_name: str or bool
        :param time_series: Boolean indicating whether the data is time series
        :type time_series: bool
        :raises ValueError: If the DataFrame index is not a pandas datetime index when time_series is set to True
        """
        self.target_name = target_name
        self.df = df
        if time_series == True:
            if type(self.df.index) is pd.DatetimeIndex:
                self.x_date = self.df.index
            else:
                raise ValueError("DataFrame index must be pandas datetime.")
            self.y_target = self.df[self.target_name]

    # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    #                                                                 UTILITY AND CALCULATION
    # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

    @staticmethod
    def crosscorr(datax, datay, lag=0):
        """Lag-N cross correlation. Taken from: https://stackoverflow.com/questions/33171413/cross-correlation-time-lag-correlation-with-pandas

        Args:
            datax (pandas.Series): Data for the x values.
            datay (pandas.Series): Data for the y values.
            lag (int, optional): Number of lags to be applied. Defaults to 0.

        Returns:
            float: Croscorrelation between X and Y for n-lags.
        """

        return datax.corr(datay.shift(lag))

    def crosscorrelation_generator(self,
                                   y_variable: str,
                                   x_variable: str,
                                   max_lags=12):
        """Function to compute the crosscorrelation for a target variable over a period of (+/-) lags.

        Args:
            y_variable (str): Name of the target variable.
            x_variable (str): Name of the feature.
            max_lags (int, optional): Number of lags that are to be computed. Defaults to 12.

        Returns:
            pandas.DataFrame: Returns the correlation in a dataframe on "Lag" and "Correlation".
        """
        # Generate the cross correlation list.
        xcov_monthly = [
            self.crosscorr(self.df[y_variable], self.df[x_variable], lag=lag)
            for lag in range(-max_lags, max_lags + 1)
        ]  # NOTE needs to be +1 to reach the value.

        return pd.DataFrame(
            {
                "Lag": np.array(range(-max_lags, max_lags + 1)),
                "Correlation": xcov_monthly,
            }
        )

    def insider_activity(self, df: pd.DataFrame):
        """
        Function to analyze insider activity by counting the number of buy, sale, and option exercise transactions in a given dataframe over a period of time.

        Args:
        df (pandas.DataFrame): The dataframe containing the data on insider activity. The dataframe should contain a column named "Transaction" that specifies the type of transaction (i.e. "Buy", "Sale", "Option Exercise"). The dataframe should also contain a column named "Date" that specifies the date of the transaction.

        Returns:
        dictionary: A dictionary containing the following dataframes:
        - "df_buy": Dataframe containing all rows of the input dataframe where the "Transaction" column is "Buy"
        - "df_sale": Dataframe containing all rows of the input dataframe where the "Transaction" column is "Sale"
        - "df_opt": Dataframe containing all rows of the input dataframe where the "Transaction" column is "Option Exercise"
        - "df_count": Dataframe containing the count of all transactions by date
        - "df_buy_count": Dataframe containing the count of buy transactions by date
        - "df_sale_count": Dataframe containing the count of sale transactions by date
        - "df_opt_count": Dataframe containing the count of option exercise transactions by date
        """
        # Generate the cross correlation list.
        df_buy = df[df["Transaction"] == "Buy"]
        df_sale = df[df["Transaction"] == "Sale"]
        df_opt = df[df["Transaction"] == "Option Exercise"]

        df_count = df.groupby(df["Date"]).count()
        df_buy_count = df_buy.groupby(df_buy["Date"]).count()
        df_sale_count = df_sale.groupby(df_sale["Date"]).count()
        df_opt_count = df_opt.groupby(df_opt["Date"]).count()

        return {
            "df_buy": df_buy,
            "df_sale": df_sale,
            "df_opt": df_opt,
            "df_count": df_count,
            "df_buy_count": df_buy_count,
            "df_sale_count": df_sale_count,
            "df_opt_count": df_opt_count,
        }

    def transactions_per_insider(self, df: pd.DataFrame):
        """
        Function to count the number of transactions per insider in a given dataframe.

        Args:
        df (pandas.DataFrame): The dataframe containing the data on insider activity. The dataframe should contain a column named "Insider Trading" that specifies the name of the insider involved in the transaction.

        Returns:
        DataFrame: A DataFrame containing the count of transactions per insider, grouped by the number of transactions. The columns of the DataFrame are:
        - "Name": The name of the insider
        - "trans_num": The number of transactions made by the insider
        - "count": The number of insiders who made the same number of transactions
        """
        trans_per_insider = pd.DataFrame(df["Insider Trading"].value_counts())
        trans_per_insider = trans_per_insider.reset_index()
        trans_per_insider.columns = ["Name", "trans_num"]
        trans = trans_per_insider.groupby(
            trans_per_insider["trans_num"]).count()
        trans.columns = ["count"]
        return trans

    def top_contributor(self, threshold=0):
        """
        Function to identify the top contributors of insider activity in a given dataframe based on a threshold value.

        Args:
        threshold (int, optional): The minimum number of incidents an insider must have to be considered a top contributor. Defaults to 0.

        Returns:
        DataFrame: A DataFrame containing the top contributors of insider activity, where the number of incidents is greater than the threshold value. The columns of the DataFrame are:
        - "Insider Trading": The name of the insider
        - "incidents_num": The number of insider activity incidents made by the insider
        """

        contributor = pd.DataFrame(self.df["Insider Trading"].value_counts())
        contributor.columns = ["incidents_num"]
        return contributor[contributor["incidents_num"] > threshold]

    def market_cap(self):
        """
        Function to calculate the market cap of the top contributors of insider activity in a given dataframe.

        Returns:
        DataFrame: A DataFrame containing the market cap value of the top contributors of insider activity, where the number of incidents is greater than the threshold value. The columns of the DataFrame are:
        - "Contributor": The name of the top contributor
        - "Value ($)": The market cap value of the top contributor
        """
        num_of_contributors = list(self.top_contributor().index)
        total_value = []
        for num_of_contributor in num_of_contributors:
            contributor = self.df[self.df["Insider Trading"] == num_of_contributor]
            contributor_value = list(contributor["Value ($)"])[0]
            total_value.append(contributor_value)
        top_market_cap = pd.DataFrame(num_of_contributors)
        top_market_cap.columns = ["Contributor"]
        top_market_cap["Value ($)"] = total_value
        top_market_cap = top_market_cap[
            top_market_cap["Value ($)"] != "unknown"]
        top_market_cap["Value ($)"] = [
            float(x) for x in top_market_cap["Value ($)"]
        ]
        return top_market_cap

    def calculate_future_prices(self, stock_df_copy: pd.DataFrame):
        """
        Function to calculate the future prices of a stock based on the transaction date in a given dataframe.

        Args:
        stock_df_copy (pandas.DataFrame): The dataframe containing the stock's historical prices.

        Returns:
        DataFrame: A DataFrame containing the transaction information and the future prices of the stock. The new columns of the DataFrame are:
        - "Close": The closing price of the stock on the transaction date
        - "Close_day1": The closing price of the stock on the day after the transaction date
        - "Close_day2": The closing price of the stock on the second day after the transaction date
        - "Close_day3": The closing price of the stock on the third day after the transaction date
        - "Close_day4": The closing price of the stock on the fourth day after the transaction date
        - "Close_day5": The closing price of the stock on the fifth day after the transaction date
        - "Close_month": The closing price of the stock on the 30th day after the transaction date
        """
        df_copy = self.df.copy()

        df_copy["new_trans_date"] = [
            time.strptime(str(y.date()), "%Y-%m-%d") for y in df_copy["Date"]
        ]
        stock_df_copy.index = [
            time.strptime(str(x.date()), "%Y-%m-%d")
            for x in stock_df_copy.Date
        ]
        df_copy = df_copy.reset_index()

        act_day, day_1, day_2, day_3, day_4, day_5, month = ([] for _ in range(7))
        for i in range(len(df_copy)):
            for j in range(len(stock_df_copy)):
                if df_copy["new_trans_date"][i] == stock_df_copy.index[j]:
                    act_day.append(stock_df_copy["Close"][j])
                    day_1.append(stock_df_copy["Close"][j + 1])
                    day_2.append(stock_df_copy["Close"][j + 2])
                    day_3.append(stock_df_copy["Close"][j + 3])
                    day_4.append(stock_df_copy["Close"][j + 4])
                    day_5.append(stock_df_copy["Close"][j + 5])
                    month.append(stock_df_copy["Close"][j + 30])
        df_copy = df_copy.assign(
            Close=act_day,
            Close_day1=day_1,
            Close_day2=day_2,
            Close_day3=day_3,
            Close_day4=day_4,
            Close_day5=day_5,
            Close_month=month,
        )
        del act_day, day_1, day_2, day_3, day_4, day_5, month
        return df_copy

    def calculate_returns(self, stock_df_copy: pd.DataFrame, diff: str):
        """
        Calculate returns for a stock dataframe

        Args:
        stock_df_copy (pd.DataFrame): The dataframe containing the stock data
        diff (str): The column name to use as the base for calculating returns

        Returns:
        pd.DataFrame: Returns the input dataframe with additional columns for day1_return, day2_return, day3_return, day4_return, day5_return, and month_return
        """
        df_copy = self.df.copy()

        act_day, day_1, day_2, day_3, day_4, day_5, month = ([] for _ in range(7))
        for day in range(len(stock_df_copy)):
            day_1.append((
                (stock_df_copy["Close_day1"][day] - stock_df_copy[diff][day]) /
                stock_df_copy[diff][day]) * 100)
            day_2.append((
                (stock_df_copy["Close_day2"][day] - stock_df_copy[diff][day]) /
                stock_df_copy[diff][day]) * 100)
            day_3.append((
                (stock_df_copy["Close_day3"][day] - stock_df_copy[diff][day]) /
                stock_df_copy[diff][day]) * 100)
            day_4.append((
                (stock_df_copy["Close_day4"][day] - stock_df_copy[diff][day]) /
                stock_df_copy[diff][day]) * 100)
            day_5.append((
                (stock_df_copy["Close_day5"][day] - stock_df_copy[diff][day]) /
                stock_df_copy[diff][day]) * 100)
            month.append(
                ((stock_df_copy["Close_month"][day] - stock_df_copy[diff][day])
                 / stock_df_copy[diff][day]) * 100)
        df_copy = df_copy.assign(
            day1_return=day_1,
            day2_return=day_2,
            day3_return=day_3,
            day4_return=day_4,
            day5_return=day_5,
            month_return=month,
        )
        del act_day, day_1, day_2, day_3, day_4, day_5, month
        return df_copy

    def boxplot_prep(self, df: pd.DataFrame, col_list: list):
        """
        Prepare a dataframe for plotting in a boxplot

        Args:
        df (pd.DataFrame): The input dataframe
        col_list (list): A list of columns in the input dataframe to use in the boxplot

        Returns:
        pd.DataFrame: Returns a new dataframe with columns 'day' and 'return', containing the values from the specified columns in the input dataframe
        """

        col_name, return_value = [], []
        for col in col_list:
            for x in range(len(df[col])):
                col_name.append(col)
                return_value.append(df[col][x])
        return_df = pd.DataFrame(col_name)
        return_df.columns = ["day"]
        return_df["return"] = return_value
        del col_name, return_value

        return return_df

    def show_returns(self, df: pd.DataFrame, threshold: int, include: list,
                     returns_type: str):
        """
        Show returns for specified activities

        Args:
        df (pd.DataFrame): The input dataframe containing the data
        threshold (int): The cut-off limit for showing returns
        include (list): A list of activities to show returns for (options: "buy", "sale", "opt")
        returns_type (str): The return type to show, options: "short" or "long"

        Returns:
        dictionary: A dictionary containing dataframes of returns for the specified activities, with keys "buy", "sale", "opt"
        """

        combined_df = {}
        future_prices = self.calculate_future_prices(df)
        return_df = self.calculate_returns(future_prices, "Cost")
        df_buy = return_df[return_df["Transaction"] == "Buy"].reset_index()
        df_sale = return_df[return_df["Transaction"] == "Sale"].reset_index()
        df_opt = return_df[return_df["Transaction"] ==
                           "Option Exercise"].reset_index()
        if returns_type == "short":
            col_name = [
                "day1_return",
                "day2_return",
                "day3_return",
                "day4_return",
                "day5_return",
            ]
        else:
            col_name = ["month_return"]
        buy = self.boxplot_prep(df_buy, col_name)
        sale = self.boxplot_prep(df_sale, col_name)
        opt = self.boxplot_prep(df_opt, col_name)

        for act in include:
            if act == "buy":
                if threshold:
                    buy = buy[buy["return"] < threshold]
                combined_df["buy"] = buy
            elif act == "sale":
                if threshold:
                    sale = sale[sale["return"] < threshold]
                combined_df["sale"] = sale
            else:
                if threshold:
                    opt = opt[opt["return"] < threshold]
                combined_df["opt"] = opt
        return combined_df

    # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    #                                                                 MATPLOTLIB FUNCTIONS
    # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

    def single_timeseries_plot(
            self,
            y_variable: str,
            rolling_mean=False,
            rolling_std=False,
            save_path=None,
            title="",
            figsize=(14, 7),
            dpi=100,
            streamlit=False,
            **kwargs,
    ):
        """Function to create a single series timeseries plot for a target variable.

        Args:
            y_variable (str): Column name of the target variable in the dataframe.
            rolling_mean (boolean, optional): Select if rolling mean is calculated. Default 6 month.
            rolling_std (boolean, optional): Select if rolling standard deviation is calculated. Default 6 month.
            save_path (str, optional): Optional save path for a .png image of the plot. Should be direct path. Defaults to None.
            title (str, optional): Title of the plot. Defaults to "".
            figsize (tuple, optional): Figure size of the plot in inch. Defaults to (14,7).
            dpi (int, optional): DPI value of the plot. Defaults to 100.
        """
        # Parse some kwargs configurations
        fontsize_title = (kwargs["fontsize_title"]
                          if kwargs.get("fontsize_title") else 20)
        fontsize_label = (kwargs["fontsize_label"]
                          if kwargs.get("fontsize_label") else 16)
        fontsize_legend = (kwargs["fontsize_legend"]
                           if kwargs.get("fontsize_legend") else 14)
        rolling_window = kwargs["rolling_window"] if kwargs.get(
            "rolling_window") else 6
        xlabel = kwargs["xlabel"] if kwargs.get("xlabel") else "Date"
        file_name_addition = (kwargs["file_name_addition"]
                              if kwargs.get("file_name_addition") else ""
                              )  # add any additional string to the file name.
        # Set to false since facecolor is set to default. Would overwrite
        # facecolor to make transparent.
        transparent = kwargs["transparent"] if kwargs.get(
            "transparent") else False
        facecolor = kwargs["facecolor"] if kwargs.get(
            "facecolor") else "#151934"
        x_range = kwargs["x_range"] if kwargs.get("x_range") else None
        y_range = kwargs["y_range"] if kwargs.get("y_range") else None

        # Generate the plot
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.plot(self.x_date, self.df[y_variable], label=f"Trend {y_variable}")
        if rolling_mean:
            ax.plot(
                self.df[y_variable].rolling(rolling_window).mean(),
                label="Moving Average",
            )
        if rolling_std:
            ax.plot(
                self.df[y_variable].rolling(rolling_window).std(),
                label="Moving Standard Deviation",
            )
        if (rolling_mean) or (rolling_std):
            plt.legend(fontsize=fontsize_legend)
        # Set the x and y range:
        if isinstance(x_range, list):
            ax.set_xlim(
                datetime.strptime(x_range[0], "%d/%m/%Y").date(),
                datetime.strptime(x_range[1], "%d/%m/%Y").date(),
            )
        if isinstance(y_range, list):
            ax.set_ylim(y_range[0], y_range[1])
        # Plot aesthetics
        ax.set_title(label=title, fontsize=fontsize_title)
        ax.set_xlabel(xlabel=xlabel, fontsize=fontsize_label)
        ax.set_ylabel(ylabel=f"{y_variable.title()}", fontsize=fontsize_label)
        fig.tight_layout()

        if save_path is not None:
            fig.savefig(
                os.path.join(
                    save_path,
                    f"single_timeseries_{y_variable}{file_name_addition}" +
                    ".png",
                ),
                facecolor=facecolor,
                transparent=transparent,
            )
        plt.show()
        if streamlit:
            return fig

    def monthly_plot(
            self,
            y_variable: str,
            save_path=None,
            figsize=(20, 7),
            dpi=80,
            streamlit=False,
            **kwargs,
    ):
        """Function to plot the monthly trend of a target variable.

        Args:
            y_variable (str): Column name of the target variable in the dataframe.
            save_path (str, optional): Optional save path for a .png image of the plot. Should be direct path. Defaults to None.
            figsize (tuple, optional): Figure size of the plot in inch. Defaults to (20,7).
            dpi (int, optional): DPI value of the plot. Defaults to 80.
        """
        # Parse some kwargs configurations
        fontsize_title = (kwargs["fontsize_title"]
                          if kwargs.get("fontsize_title") else 20)
        fontsize_label = (kwargs["fontsize_label"]
                          if kwargs.get("fontsize_label") else 14)
        line_color = kwargs["line_color"] if kwargs.get(
            "line_color") else "cyan"
        zorder = kwargs["zorder"] if kwargs.get("zorder") else 0
        file_name_addition = (kwargs["file_name_addition"]
                              if kwargs.get("file_name_addition") else ""
                              )  # add any additional string to the file name.
        # Set to false since facecolor is set to default. Would overwrite
        # facecolor to make transparent.
        transparent = kwargs["transparent"] if kwargs.get(
            "transparent") else False
        facecolor = kwargs["facecolor"] if kwargs.get(
            "facecolor") else "#151934"

        # Generate the plot
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        fig = month_plot(x=self.df[y_variable].dropna(), ax=ax)

        # Plot aesthetics
        ax.set_title(label=f"Month Plot {y_variable.title()}",
                     fontsize=fontsize_title)
        ax.set_xlabel(xlabel="Month", fontsize=fontsize_label)
        ax.set_ylabel(ylabel=y_variable.title(), fontsize=fontsize_label)

        # Change line color
        lines = ax.get_lines()
        for line in lines:
            line.set_color(line_color)
        for line in lines:
            line.set_zorder(zorder)
        fig.tight_layout()

        if save_path is not None:
            fig.savefig(
                os.path.join(
                    save_path, f"monthly_plot_{y_variable}{file_name_addition}.png"
                ),
                facecolor=facecolor,
                transparent=transparent,
            )
        plt.show()
        if streamlit:
            return fig

    def quarterly_plot(
            self,
            y_variable: str,
            save_path=None,
            figsize=(20, 7),
            dpi=80,
            streamlit=False,
            **kwargs,
    ):
        """Function to plot the quarterly trend of a target variable.

        Args:

            y_variable (str): Column name of the target variable in the dataframe.
            save_path (str, optional): Optional save path for a .png image of the plot. Should be direct path. Defaults to None.
            figsize (tuple, optional): Figure size of the plot in inch. Defaults to (20,7).
            dpi (int, optional): DPI value of the plot. Defaults to 80.
        """
        # Parse some kwargs configurations
        fontsize_title = (kwargs["fontsize_title"]
                          if kwargs.get("fontsize_title") else 20)
        fontsize_label = (kwargs["fontsize_label"]
                          if kwargs.get("fontsize_label") else 14)
        line_color = kwargs["line_color"] if kwargs.get(
            "line_color") else "cyan"
        zorder = kwargs["zorder"] if kwargs.get("zorder") else 0
        file_name_addition = (kwargs["file_name_addition"]
                              if kwargs.get("file_name_addition") else ""
                              )  # add any additional string to the file name.
        # Set to false since facecolor is set to default. Would overwrite
        # facecolor to make transparent.
        transparent = kwargs["transparent"] if kwargs.get(
            "transparent") else False
        facecolor = kwargs["facecolor"] if kwargs.get(
            "facecolor") else "#151934"

        # Generate the plot
        df_sub = self.df[y_variable].copy()
        df_sub.index = self.df.index.to_period("Q")
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        fig = quarter_plot(x=df_sub.dropna(), ax=ax)

        # Plot aesthetics
        ax.set_title(label=f"Month Plot {y_variable.title()}",
                     fontsize=fontsize_title)
        ax.set_xlabel(xlabel="Month", fontsize=fontsize_label)
        ax.set_ylabel(ylabel=y_variable.title(), fontsize=fontsize_label)

        # Change line color
        lines = ax.get_lines()
        for line in lines:
            line.set_color(line_color)
        for line in lines:
            line.set_zorder(zorder)
        fig.tight_layout()

        if save_path is not None:
            fig.savefig(
                os.path.join(
                    save_path,
                    f"quarterly_plot_{y_variable}{file_name_addition}" +
                    ".png",
                ),
                facecolor=facecolor,
                transparent=transparent,
            )
        plt.show()
        if streamlit:
            return fig

    def seasonal_boxplot_ym(
            self,
            y_variable: str,
            save_path=None,
            figsize=(20, 7),
            dpi=80,
            streamlit=False,
            **kwargs,
    ):
        """Function that creates the seasonal boxplot for year and month.

        Args:
            y_variable (str): Column name of the target variable in the dataframe.
            save_path (str, optional): Optional save path for a .png image of the plot. Should be direct path. Defaults to None.
            figsize (tuple, optional): Figure size of the plot in inch. Defaults to (20,7).
            dpi (int, optional): DPI value of the plot. Defaults to 80.
        """
        # Parse some kwargs configurations
        fontsize_title = (kwargs["fontsize_title"]
                          if kwargs.get("fontsize_title") else 20)
        fontsize_label = (kwargs["fontsize_label"]
                          if kwargs.get("fontsize_label") else 14)
        fontsize_ticks = (kwargs["fontsize_ticks"]
                          if kwargs.get("fontsize_ticks") else 14)
        x_labelrotation = (kwargs["x_labelrotation"]
                           if kwargs.get("x_labelrotation") else 45)
        box_line_color = (kwargs["x_labelrotation"]
                          if kwargs.get("x_labelrotation") else "silver")
        file_name_addition = (kwargs["file_name_addition"]
                              if kwargs.get("file_name_addition") else ""
                              )  # add any additional string to the file name.
        # Set to false since facecolor is set to default. Would overwrite
        # facecolor to make transparent.
        transparent = kwargs["transparent"] if kwargs.get(
            "transparent") else False
        facecolor = kwargs["facecolor"] if kwargs.get(
            "facecolor") else "#151934"

        # Prepare data for plot by adding year and month column.
        self.df["year"] = [d.year for d in self.df.index]
        self.df["month"] = [d.strftime("%b") for d in self.df.index]

        # Create plots
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=figsize, dpi=dpi)

        sns.boxplot(
            x="year",
            y=y_variable,
            data=self.df,
            ax=axs[0],
            boxprops=dict(edgecolor=box_line_color),
            capprops=dict(color=box_line_color),
            whiskerprops=dict(color=box_line_color),
            flierprops=dict(
                color=box_line_color,
                markerfacecolor=box_line_color,
                markeredgecolor=box_line_color,
            ),
            medianprops=dict(color=box_line_color),
        )
        sns.boxplot(
            x="month",
            y=y_variable,
            data=self.df.loc[~self.df.year.isin([1991, 2000]), :],
            ax=axs[1],
            boxprops=dict(edgecolor=box_line_color),
            capprops=dict(color=box_line_color),
            whiskerprops=dict(color=box_line_color),
            flierprops=dict(
                color=box_line_color,
                markerfacecolor=box_line_color,
                markeredgecolor=box_line_color,
            ),
            medianprops=dict(color=box_line_color),
        )

        # Plot Aesthetics
        axs[0].set_title(label="Year-wise Box Plot\n(The Trend)",
                         fontsize=fontsize_title)
        axs[1].set_title(label="Month-wise Box Plot\n(The Seasonality)",
                         fontsize=fontsize_title)

        axs[0].set_xlabel(xlabel="Year".title(), fontsize=fontsize_label)
        axs[1].set_xlabel(xlabel="Month".title(), fontsize=fontsize_label)

        axs[0].set_ylabel(ylabel=y_variable.title(), fontsize=fontsize_label)
        axs[1].set_ylabel(ylabel=y_variable.title(), fontsize=fontsize_label)

        axs[0].tick_params(axis="x",
                           labelsize=fontsize_ticks,
                           labelrotation=x_labelrotation)
        axs[1].tick_params(axis="x",
                           labelsize=fontsize_ticks,
                           labelrotation=x_labelrotation)

        axs[0].tick_params(axis="y", labelsize=fontsize_ticks)
        axs[1].tick_params(axis="y", labelsize=fontsize_ticks)
        fig.tight_layout()

        # Remove the two helper columns
        self.df.drop(["month", "year"], axis=1, inplace=True)

        if save_path is not None:
            fig.savefig(
                os.path.join(
                    save_path,
                    f"ym_seasonal_decompose_{y_variable}{file_name_addition}" +
                    ".png",
                ),
                facecolor=facecolor,
                transparent=transparent,
            )
        plt.show()
        if streamlit:
            return fig

    def target_lag_plots(
            self,
            y_variable: str,
            lags=8,
            save_path=None,
            figsize=(16, 7),
            streamlit=False,
            **kwargs,
    ):
        """Function to create a series of lag plots (number specified by lags) for the specified variable.

        Args:
            y_variable (str): Column name of the target variable in the dataframe.
            lags (int, optional): Number of lags to be added. Please not that depending on the number of lags you need to specify the plot_matrix_shape e.g. 240. Defaults to 8.
            save_path (str, optional): Optional save path for a .png image of the plot. Should be direct path. Defaults to None.
            figsize (tuple, optional): Figure size of the plot in inch. Defaults to (16,7).
        """
        # Parse some kwargs configurations
        fontsize_title = (kwargs["fontsize_title"]
                          if kwargs.get("fontsize_title") else 20)
        plot_matrix_shape = (kwargs["plot_matrix_shape"]
                             if kwargs.get("plot_matrix_shape") else 240)
        file_name_addition = (kwargs["file_name_addition"]
                              if kwargs.get("file_name_addition") else ""
                              )  # add any additional string to the file name.
        # Set to false since facecolor is set to default. Would overwrite
        # facecolor to make transparent.
        transparent = kwargs["transparent"] if kwargs.get(
            "transparent") else False
        facecolor = kwargs["facecolor"] if kwargs.get(
            "facecolor") else "#151934"

        plt.figure(figsize=figsize)
        plt.suptitle(f"Lag Correlation Plot for {y_variable}",
                     fontsize=fontsize_title)

        # Abstract values and convert to columns for the target varaiable
        values = self.df[y_variable]
        columns = [values]

        # Append the lags t+1.
        columns.extend(values.shift(i) for i in range(1, (lags + 1)))
        df_lag = pd.concat(columns, axis=1)
        columns = ["t+1"]

        # Append the lags t-h
        columns.extend(f"t-{i}" for i in range(1, (lags + 1)))
        df_lag.columns = columns

        plt.figure(1)
        for i in range(1, (lags + 1)):
            # this is for dimensions rows, cols
            ax = plt.subplot(plot_matrix_shape + i)
            ax.set_title(f"t+1 vs t-{i}")
            plt.scatter(x=df_lag["t+1"].values, y=df_lag[f"t-{i}"].values, s=1)
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(
                os.path.join(
                    save_path, f"lag_plot_{y_variable}{file_name_addition}.png"
                ),
                facecolor=facecolor,
                transparent=transparent,
            )
        plt.show()
        if streamlit:
            return plt

    def plot_acf_pacf(
            self,
            y_variable: str,
            diff_target=False,
            lags=60,
            save_path=None,
            streamlit=False,
            figsize=(15, 6),
            **kwargs,
    ):
        """Function to create the autocorrelation and partial autocorrelation plot.

        Args:
            y_variable (str): Column name of the target variable in the dataframe.
            diff_target (bool, optional): Select if the target column is to find the discrete differences. Defaults to False.
            lags (int, optional): Number of lags for the correlation. Defaults to 60.
            save_path (str, optional): Optional save path for a .png image of the plot. Should be direct path. Defaults to None.
            figsize (tuple, optional): Figure size of the plot in inch. Defaults to (15,6).
        """
        # Parse some kwargs configurations
        k_diff = kwargs["k_diff"] if kwargs.get("k_diff") else 1
        file_name_addition = (kwargs["file_name_addition"]
                              if kwargs.get("file_name_addition") else ""
                              )  # add any additional string to the file name.
        # Set to false since facecolor is set to default. Would overwrite
        # facecolor to make transparent.
        transparent = kwargs["transparent"] if kwargs.get(
            "transparent") else False
        facecolor = kwargs["facecolor"] if kwargs.get(
            "facecolor") else "#151934"

        # Generate plot
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)

        # Check if target variable is differentiated
        if diff_target:
            y_target = diff(self.df[y_variable], k_diff=k_diff)
        else:
            y_target = self.df[y_variable].copy()
        # Handle missing values by dropping them.
        y_target.dropna(inplace=True)

        # Plot acf and pacf
        plot_acf(y_target.tolist(), lags=lags, ax=ax[0])
        # just the plot
        plot_pacf(y_target.tolist(), lags=lags, ax=ax[1])
        # just the plot
        fig.tight_layout()

        if save_path is not None:
            fig.savefig(
                os.path.join(
                    save_path, f"acf_pacf_{y_variable}{file_name_addition}.png"
                ),
                facecolor=facecolor,
                transparent=transparent,
            )
        plt.show()
        if streamlit:
            return fig

    def plot_seasonal_decomposition(
            self,
            y_variable: str,
            save_path=None,
            figsize=(16, 12),
            streamlit=False,
            **kwargs,
    ):
        """Function to create the seasonal composition plot.

        Args:
            y_variable (str): Column name of the target variable in the dataframe.
            save_path (str, optional): Optional save path for a .png image of the plot. Should be direct path. Defaults to None.
            figsize (tuple, optional): Figure size of the plot in inch. Defaults to (16,12).
        """
        # Parse some kwargs configurations
        fontsize_title = kwargs["k_diff"] if kwargs.get("k_diff") else 20
        extrapolate_trend = (kwargs["extrapolate_trend"]
                             if kwargs.get("extrapolate_trend") else "freq")
        decompose_model = (kwargs["decompose_model"]
                           if kwargs.get("decompose_model") else "additive"
                           )  # Can be "additive", "multiplicative",
        title_label = (
            kwargs["title_label"] if kwargs.get("title_label") else
            f"{decompose_model.title()} Decomposition of {y_variable}")
        axhline_color = (kwargs["axhline_color"]
                         if kwargs.get("axhline_color") else "white")
        axhline_linewidth = (kwargs["axhline_linewidth"]
                             if kwargs.get("axhline_linewidth") else 1.5)
        file_name_addition = (kwargs["file_name_addition"]
                              if kwargs.get("file_name_addition") else ""
                              )  # add any additional string to the file name.
        # Set to false since facecolor is set to default. Would overwrite
        # facecolor to make transparent.
        transparent = kwargs["transparent"] if kwargs.get(
            "transparent") else False
        facecolor = kwargs["facecolor"] if kwargs.get(
            "facecolor") else "#151934"

        # Set plot Params
        plt.rcParams.update({
            "figure.figsize": figsize,
            "lines.markersize": 2,
        })

        # Generate plot and plot features
        result_add = seasonal_decompose(
            self.df[y_variable],
            model=decompose_model,
            extrapolate_trend=extrapolate_trend,
        )
        result_add.plot().suptitle(title_label, fontsize=fontsize_title)
        plt.axhline(0, c=axhline_color, linewidth=axhline_linewidth)
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(
                os.path.join(
                    save_path,
                    f"seasonal_decomposition_{y_variable}{file_name_addition}"
                    + ".png",
                ),
                facecolor=facecolor,
                transparent=transparent,
            )
        plt.show()
        if streamlit:
            return plt

    def ask_adfuller(self, col_list: list, autolag="aic", **kwargs):
        """Function to run ad fuller test on target variable.

        Args:
            col_list (list): List of column name of the target variable in the dataframe.
            autolag (str, optional): Method to use when automatically determining the lag length among the values. Defaults to "aic". Can be AIC, BIC, t-stat.
        """
        # Parse some kwargs configuration
        # "c" default, "ct" constant and trend, "ctt" constant linear and quatratic, "n" non constant
        regression = kwargs["maxlag"] if kwargs.get("maxlag") else "c"
        # Run the test:
        for y_variable in col_list:
            test_results = adfuller(self.df[y_variable],
                                    regression=regression,
                                    autolag=autolag)
            print(
                "---------------------------------------------------------------------------------------------------------------------"
            )
            print(f"AD Fuller Test for {y_variable}:")
            print(
                "---------------------------------------------------------------------------------------------------------------------"
            )
            print("Test statistic: ", test_results[0])
            print("p-value: ", test_results[1])
            print("Critical Values:", test_results[4])
            print(
                "----------------------------------------------------------------------------------------------------------------------"
            )

    def plot_stl_decomposition(
            self,
            y_variable: str,
            seasonal=11,
            trend=15,
            save_path=None,
            streamlit=False,
            figsize=(16, 12),
            **kwargs,
    ):
        """Function to create the seasonal composition plot.

        Args:
            y_variable (str): Column name of the target variable in the dataframe.
            save_path (str, optional): Optional save path for a .png image of the plot. Should be direct path. Defaults to None.
            figsize (tuple, optional): Figure size of the plot in inch. Defaults to (16,12).
        """
        # Parse some kwargs configurations
        fontsize_title = kwargs["k_diff"] if kwargs.get("k_diff") else 20
        title_label = (kwargs["title_label"] if kwargs.get("title_label") else
                       f"STL Decomposition of {y_variable}")
        axhline_color = (kwargs["axhline_color"]
                         if kwargs.get("axhline_color") else "white")
        axhline_linewidth = (kwargs["axhline_linewidth"]
                             if kwargs.get("axhline_linewidth") else 1.5)
        file_name_addition = (kwargs["file_name_addition"]
                              if kwargs.get("file_name_addition") else ""
                              )  # add any additional string to the file name.
        # Set to false since facecolor is set to default. Would overwrite
        # facecolor to make transparent.
        transparent = kwargs["transparent"] if kwargs.get(
            "transparent") else False
        facecolor = kwargs["facecolor"] if kwargs.get(
            "facecolor") else "#151934"

        # Set plot Params
        plt.rcParams.update({
            "figure.figsize": figsize,
            "lines.markersize": 2,
        })

        # Generate plot and plot features
        stl = STL(self.df[y_variable], seasonal=seasonal, trend=trend)
        result_add = stl.fit()
        fig = result_add.plot().suptitle(title_label, fontsize=fontsize_title)

        # Adjust color of axhline
        fig.axhline(0, c=axhline_color, linewidth=axhline_linewidth)
        fig.tight_layout()

        if save_path is not None:
            fig.savefig(
                os.path.join(
                    save_path,
                    f"stl_decomposition_{y_variable}{file_name_addition}" +
                    ".png",
                ),
                facecolor=facecolor,
                transparent=transparent,
            )
        plt.show()
        if streamlit:
            return fig

    def correlate_all_plot(
            self,
            y_variable: str,
            x_variables: list,
            max_lags=30,
            streamlit=False,
            save_path=None,
            figsize=(20, 35),
            rect=(0, 0, 1, 0.96),
            **kwargs,
    ):
        """Function to create a correlation plot between a target variable y and all the feature variables x.

        Args:
            y_variable (str): Column name of the target variable in the dataframe.
            x_variables (list): Column names of the feature variables in the dataframe. This should exclude y_variable.
            max_lags (int, optional): The maximum number of lags that are used for the correlation plot. Defaults to 30.
            save_path (str, optional): Optional save path for a .png image of the plot. Should be direct path. Defaults to None.
            figsize (tuple, optional): Figure size of the plot in inch. Defaults to (20,35).
            rect (tuple, optional): Tuple that indicates how the tight layout is configured. Defaults to (0,0,1,0.96).
        """
        # Parse some kwargs configurations
        fontsize_title = (kwargs["fontsize_title"]
                          if kwargs.get("fontsize_title") else 20)
        fontsize_sub_title = (kwargs["fontsize_title"]
                              if kwargs.get("fontsize_title") else 16)
        fontsize_label = (kwargs["fontsize_label"]
                          if kwargs.get("fontsize_label") else 14)
        n_x_ticks = kwargs["n_x_ticks"] if kwargs.get("n_x_ticks") else 10
        threshold_value = (kwargs["threshold_value"]
                           if kwargs.get("threshold_value") else 0.1)
        color_fillbetween = (kwargs["color_fillbetween"]
                             if kwargs.get("color_fillbetween") else "pink")
        alpha_fillbetween = (kwargs["alpha_fillbetween"]
                             if kwargs.get("alpha_fillbetween") else 0.2)
        xcorr_lw = kwargs["xcorr_lw"] if kwargs.get("xcorr_lw") else 2
        usevlines = kwargs["usevlines"] if kwargs.get("usevlines") else True
        normed = kwargs["normed"] if kwargs.get("normed") else True
        file_name_addition = (kwargs["file_name_addition"]
                              if kwargs.get("file_name_addition") else ""
                              )  # add any additional string to the file name.
        # Set to false since facecolor is set to default. Would overwrite
        # facecolor to make transparent.
        transparent = kwargs["transparent"] if kwargs.get(
            "transparent") else False
        facecolor = kwargs["facecolor"] if kwargs.get(
            "facecolor") else "#151934"

        # Generate Plots
        fig, axs = plt.subplots(
            nrows=int(np.ceil(len(x_variables) / 4)),
            ncols=4,
            sharex=True,
            sharey=True,
            figsize=figsize,
        )

        # Generate threshold dataset
        x_threshold = np.arange(-max_lags, max_lags + 10, n_x_ticks)
        y_upper_max = [threshold_value] * len(x_threshold)
        y_lower_max = [-threshold_value] * len(x_threshold)

        # Reshape if ndim == 1
        if axs.ndim == 1:
            axs = axs.reshape(1, -1)
        # Iterate through the the correlation plots. Adding 4 plots per row.
        for i, x_variable in enumerate(x_variables):
            # Add x an y value from dataframe.
            x, y = self.df[x_variable].fillna(0), self.df[y_variable].dropna()
            axs[i // 4, i % 4].xcorr(
                x,
                y,
                normed=normed,
                usevlines=usevlines,
                maxlags=max_lags,
                lw=xcorr_lw,
                detrend=mlab.detrend_mean,
            )
            axs[i // 4, i % 4].fill_between(
                x_threshold,
                y_upper_max,
                y_lower_max,
                color=color_fillbetween,
                alpha=alpha_fillbetween,
            )
            # Plot aestethics
            axs[i // 4, i % 4].set_title(x_variable,
                                         fontsize=fontsize_sub_title)
            axs[i // 4, i % 4].set_xlabel("<-- lag | lead -->",
                                          fontsize=fontsize_label)
            axs[i // 4, i % 4].grid(axis="x")
            axs[i // 4,
                i % 4].set_xticks(np.arange(-max_lags, max_lags + 5,
                                            n_x_ticks))
            axs[i // 4, i % 4].tick_params(axis="x", labelbottom=True)
        # Disable any unused or empty plots
        i += 1
        while i < axs.size:
            axs[i // 4, i % 4].set_visible(False)
            i += 1
        # Layout and plot
        fig.suptitle(f"Cross Correlation Against {y_variable.title()}",
                     fontsize=fontsize_title)
        fig.tight_layout(rect=rect)

        if save_path is not None:
            fig.savefig(
                os.path.join(
                    save_path,
                    f"cross_correlation_all_{y_variable}{file_name_addition}" +
                    ".png",
                ),
                facecolor=facecolor,
                transparent=transparent,
            )
        plt.show()
        if streamlit:
            return fig

    def single_correlate_plot(
            self,
            y_variable: str,
            x_variable: str,
            max_lags=30,
            streamlit=False,
            save_path=None,
            figsize=(15, 6),
            dpi=180,
            **kwargs,
    ):
        """Function to plot a correlation plot between  variable y and x for n lags.

        Args:
            y_variable (str): Column name of the target variable in the dataframe.
            x_variable (str): Column name of the feature variable in the dataframe.
            max_lags (int, optional): The maximum number of lags that are used for the correlation plot. Defaults to 30.
            save_path (str, optional): Optional save path for a .png image of the plot. Should be direct path. Defaults to None.
            figsize (tuple, optional): Figure size of the plot in inch. Defaults to (15,6).
            dpi (int, optional): dpi (int, optional): DPI value of the plot. Defaults to 180.
        """
        # Parse some kwargs configurations
        fontsize_title = (kwargs["fontsize_title"]
                          if kwargs.get("fontsize_title") else 20)
        fontsize_label = (kwargs["fontsize_label"]
                          if kwargs.get("fontsize_label") else 14)
        fontsize_xyticks = (kwargs["fontsize_xyticks"]
                            if kwargs.get("fontsize_xyticks") else 12)
        n_x_ticks = kwargs["n_x_ticks"] if kwargs.get("n_x_ticks") else 10
        threshold_value = (kwargs["threshold_value"]
                           if kwargs.get("threshold_value") else 0.1)
        color_fillbetween = (kwargs["color_fillbetween"]
                             if kwargs.get("color_fillbetween") else "pink")
        alpha_fillbetween = (kwargs["alpha_fillbetween"]
                             if kwargs.get("alpha_fillbetween") else 0.2)
        xcorr_lw = kwargs["xcorr_lw"] if kwargs.get("xcorr_lw") else 5
        usevlines = kwargs["usevlines"] if kwargs.get("usevlines") else True
        normed = kwargs["normed"] if kwargs.get("normed") else True
        file_name_addition = (kwargs["file_name_addition"]
                              if kwargs.get("file_name_addition") else ""
                              )  # add any additional string to the file name.
        # Set to false since facecolor is set to default. Would overwrite
        # facecolor to make transparent.
        transparent = kwargs["transparent"] if kwargs.get(
            "transparent") else False
        facecolor = kwargs["facecolor"] if kwargs.get(
            "facecolor") else "#151934"

        # Generate Plots
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Generate threshold dataset
        x_threshold = np.arange(-max_lags, max_lags + 10, n_x_ticks)
        y_upper_max = [threshold_value] * len(x_threshold)
        y_lower_max = [-threshold_value] * len(x_threshold)

        x, y = self.df[x_variable].fillna(0), self.df[y_variable].dropna()
        ax.xcorr(
            x,
            y,
            normed=normed,
            usevlines=usevlines,
            maxlags=max_lags,
            lw=xcorr_lw,
            detrend=mlab.detrend_mean,
        )
        ax.fill_between(
            x_threshold,
            y_upper_max,
            y_lower_max,
            color=color_fillbetween,
            alpha=alpha_fillbetween,
        )

        # Plot aestethics
        ax.set_title(f"{y_variable.title()} vs {x_variable.title()}",
                     fontsize=fontsize_title)
        ax.set_xlabel("<-- lead | lag -->", fontsize=fontsize_label)
        ax.set_xticks(np.arange(-max_lags, max_lags + 5, n_x_ticks))
        ax.tick_params(axis="x", labelbottom=True)
        ax.tick_params(axis="both", labelsize=fontsize_xyticks)

        fig.tight_layout()
        if save_path is not None:
            fig.savefig(
                os.path.join(
                    save_path,
                    f"cross_correlation_{y_variable}_v_{x_variable}{file_name_addition}"
                    + ".png",
                ),
                facecolor=facecolor,
                transparent=transparent,
            )
        plt.show()
        if streamlit:
            return fig

    def granger_causality_generator(self,
                                    y_variable: str,
                                    x_variable: str,
                                    max_lags=12):
        """Function to calculate the granger causality and return the values for the max_lag period as a dictionary.

        Args:
            y_variable (str): Name of the target variable.
            x_variable (str): Name of the feature.
            max_lags (int, optional): _description_. Defaults to 12.

        Returns:
            dict: Dictionary with keys "F-value", "P-value" and "Lag-range". Values are lists.
        """
        # Calculate Granger Causality
        lag_range = range(1, max_lags + 1)
        f_list = []
        p_list = []
        # Filter data:
        x, y = self.df[x_variable].fillna(0), self.df[y_variable].dropna()
        for lag in lag_range:
            res = grangercausalitytests(
                pd.DataFrame(y.dropna()).join(x.dropna(), how="inner"),
                maxlag=max_lags,
                verbose=False,
            )
            f, p, _, _ = res[lag][0]["ssr_ftest"]
            f_list.append(f)
            p_list.append(p)
        return {
            "F-value": f_list,
            "P-value": p_list,
            "Lag-range": np.array(lag_range).tolist(),
        }

    def single_granger_plot(
            self,
            y_variable: str,
            x_variable: str,
            max_lags=12,
            streamlit=False,
            save_path=None,
            figsize=(15, 6),
            dpi=180,
            **kwargs,
    ):
        """Function to plot the granger causality between x and y for n lags.

        Args:
            y_variable (str): Name of the target variable
            x_variable (str): Name of the feature variable
            max_lags (int, optional): Number of max lags applied in the granger function. Defaults to 12.
            save_path (str, optional): Optional save path for a .png image of the plot. Should be direct path. Defaults to None.
            figsize (tuple, optional): Figure size of the plot in inch. Defaults to (15,6).
            dpi (int, optional): dpi (int, optional): DPI value of the plot. Defaults to 180.
        """
        # Parse some kwargs configurations
        fontsize_title = (kwargs["fontsize_title"]
                          if kwargs.get("fontsize_title") else 20)
        fontsize_label = (kwargs["fontsize_label"]
                          if kwargs.get("fontsize_label") else 14)
        fontsize_xyticks = (kwargs["fontsize_xyticks"]
                            if kwargs.get("fontsize_xyticks") else 12)
        file_name_addition = (kwargs["file_name_addition"]
                              if kwargs.get("file_name_addition") else ""
                              )  # add any additional string to the file name.
        # Set to false since facecolor is set to default. Would overwrite
        # facecolor to make transparent.
        transparent = kwargs["transparent"] if kwargs.get(
            "transparent") else False
        facecolor = kwargs["facecolor"] if kwargs.get(
            "facecolor") else "#151934"
        show_pval = (kwargs["show_pval"] if kwargs.get("show_pval") else True
                     )  # If p-value is shown. Can be True and False.

        # Generate the Granger Causality
        grange_dict = self.granger_causality_generator(y_variable,
                                                       x_variable,
                                                       max_lags=max_lags)

        # Generate Plots
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        ax.bar(x=grange_dict["Lag-range"], height=grange_dict["F-value"])
        if show_pval:
            props = dict(boxstyle="round", facecolor="black", alpha=0.5)
            ax.text(
                x=0.5,
                y=0.8,
                s=f"minimum p-value = {min(grange_dict['P-value']):.3f}",
                transform=ax.transAxes,
                bbox=props,
            )
        # Plot aestethics
        ax.set_title(
            f"Granger causality, {y_variable.title()} vs {x_variable.title()}",
            fontsize=fontsize_title,
        )
        ax.set_xlabel("lag -->", fontsize=fontsize_label)
        ax.set_ylabel("Granger Score (F)", fontsize=fontsize_label)
        ax.set_xticks(list(grange_dict["F-value"]))
        ax.set_xticklabels(list(grange_dict["Lag-range"]))
        ax.tick_params(axis="x", labelbottom=True)
        ax.tick_params(axis="both", labelsize=fontsize_xyticks)

        sns.despine()
        fig.tight_layout()
        if save_path is not None:
            fig.savefig(
                os.path.join(
                    save_path,
                    f"granger_causality_{y_variable}_v_{x_variable}{file_name_addition}"
                    + ".png",
                ),
                facecolor=facecolor,
                transparent=transparent,
            )
        plt.show()
        if streamlit:
            return fig

    # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    #                                                                 PLOTLY FUNCTIONS
    # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

    def plotly_single_timeseries_plot(
            self,
            y_variable: str,
            rolling_mean=False,
            rolling_std=False,
            figsize=(1400, 500),
            streamlit=False,
            display_fig=True,
            **kwargs,
    ):
        """Function to plotly plot as a single time series plot. Select if rolling average and rolling standard deviation is included.

        Args:
            y_variable (str): Column name of the target variable in the dataframe.
            rolling_mean (boolean, optional): Select if rolling mean is calculated. Default 6 month.
            rolling_std (boolean, optional): Select if rolling standard deviation is calculated. Default 6 month.
            figsize (tuple, optional): Figure size of the plot in inch. Defaults to (1400, 500).
            streamlit (bool, optional): Select if fig object is returned from function. Defaults to False.
            display_fig (bool, optional): Select if figure is displayed. Defaults to True.

        Returns:
           plotly figure object: Returns plotly figure object if streamlit is true.
        """
        rolling_window = (
            kwargs["rolling_window"] if kwargs.get("rolling_window") else 6
        )  # Select rolling window for average and standard deviation.

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(x=self.x_date,
                       y=round(self.df[y_variable], 1),
                       name="Trend"))

        if rolling_mean:
            fig.add_trace(
                go.Scatter(
                    x=self.x_date,
                    y=round(self.df[y_variable].rolling(rolling_window).mean(),
                            1),
                    name="Moving Average",
                ))
        if rolling_std:
            fig.add_trace(
                go.Scatter(
                    x=self.x_date,
                    y=round(self.df[y_variable].rolling(rolling_window).std(),
                            1),
                    name="Moving Standard Deviation",
                ))
        fig.update_layout(
            autosize=False,
            width=figsize[0],
            height=figsize[1],
            legend=dict(yanchor="top", xanchor="right"),
            yaxis_title=y_variable.title(),
            hovermode="x unified",
            template="plotly_dark",
            margin=dict(l=80, r=30, t=30, b=50),
            plot_bgcolor="#151934",
            paper_bgcolor="#151934",
        )

        if display_fig:
            # NOTE this could also be adjusted to save the fig.
            fig.show(renderer="notebook_connected")
        if streamlit:
            return fig

    def plotly_seasonal_boxplot_ym(
            self,
            y_variable: str,
            box_group: str,
            figsize=(1400, 500),
            streamlit=False,
            display_fig=True,
            **kwargs,
    ):
        """Function to plot a single box plto for either month or year as defined by the box_group variable.

        Args:
            y_variable (str): Name of the target variable
            box_group (str): Select the box group aggregation. Can be yearly or monthly.
            figsize (tuple, optional): Figure size in inches. Defaults to (1400, 500).
            streamlit (bool, optional): Select if fig object is returned from function. Defaults to False.
            display_fig (bool, optional): Select if figure is displayed. Defaults to True.

        Returns:
            plotly figure object: Returns plotly figure object if streamlit is true.
        """

        # Prepare data for plot by adding year and month column.
        if box_group == "year":
            self.df[box_group] = [d.year for d in self.df.index]
        else:
            self.df[box_group] = [d.strftime("%b") for d in self.df.index]
        fig = go.Figure()

        fig.add_trace(go.Box(
            x=self.df[box_group],
            y=self.df[y_variable],
        ))

        fig.update_layout(
            title=f"Seasonal Boxplot - {box_group}",
            autosize=False,
            width=figsize[0],
            height=figsize[1],
            yaxis_title=y_variable.title(),
            template="plotly_dark",
            margin=dict(l=80, r=30, t=80, b=50),
            plot_bgcolor="#151934",
            paper_bgcolor="#151934",
        )

        # Remove the two helper columns
        self.df.drop([box_group], axis=1, inplace=True)

        if display_fig:
            # NOTE this could also be adjusted to save the fig.
            fig.show(renderer="notebook_connected")
        if streamlit:
            return fig

    def plotly_single_correlation(
            self,
            y_variable: str,
            x_variable: str,
            max_lags=12,
            figsize=(1400, 500),
            streamlit=False,
            display_fig=True,
            **kwargs,
    ):
        """Function to generate the crosscorrelation plot of a number of lags between target and feature variable.

        Args:
            y_variable (str): Name of the target variable
            x_variable (str): Name of the feature variable
            max_lags (int, optional): Number of max lags applied in the granger function. Defaults to 12.
            figsize (tuple, optional): Figure size of the plot in inch. Defaults to (1400, 500).
            streamlit (bool, optional): Select if fig object is returned from function. Defaults to False.
            display_fig (bool, optional): Select if figure is displayed. Defaults to True.

        Returns:
            plotly figure object: Returns plotly figure object if streamlit is true.
        """
        df_corr = self.crosscorrelation_generator(y_variable=y_variable,
                                                  x_variable=x_variable,
                                                  max_lags=max_lags)
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=np.array(
                    range(int(df_corr["Lag"].min() - 1),
                          int(df_corr["Lag"].max() + 2))),
                y=(len(df_corr) + 2) * [0.1],
                fill="tozeroy",
                fillcolor="rgba(245,218,223,0.2)",
                marker_color="rgba(245,218,223,0.0)",
                hoverinfo="skip",
                showlegend=False,
            ))

        fig.add_trace(
            go.Scatter(
                x=np.array(
                    range(int(df_corr["Lag"].min() - 1),
                          int(df_corr["Lag"].max() + 2))),
                y=(len(df_corr) + 2) * [-0.1],
                fill="tozeroy",
                fillcolor="rgba(245,218,223,0.2)",
                marker_color="rgba(245,218,223,0.0)",
                hoverinfo="skip",
                showlegend=False,
            ))

        fig.add_trace(
            go.Bar(
                x=df_corr["Lag"],
                y=df_corr["Correlation"],
                orientation="v",
                marker_color="rgba(98,249,252,0.9)",
            ))

        fig.update_layout(
            yaxis=dict(categoryorder="total ascending"),
            title=
            f"Crosscorrelation: {y_variable.title()} vs {x_variable.title()}",
            autosize=False,
            width=figsize[0],
            height=figsize[1],
            xaxis_title="<- Lag | Lead ->",
            yaxis_title="Correlation Score",
            xaxis=dict(
                tickmode="linear",
                tick0=1,
                dtick=1,
                range=(df_corr["Lag"].min() - 0.5, df_corr["Lag"].max() + 0.5),
            ),
            hovermode="x",
            template="plotly_dark",
            margin=dict(l=80, r=30, t=80, b=50),
            plot_bgcolor="#151934",
            paper_bgcolor="#151934",
            showlegend=False,
        )

        if display_fig:
            # NOTE this could also be adjusted to save the fig.
            fig.show(renderer="notebook_connected")
        if streamlit:
            return fig

    def plotly_single_granger(
            self,
            y_variable: str,
            x_variable: str,
            max_lags=12,
            figsize=(1400, 500),
            streamlit=False,
            display_fig=True,
            **kwargs,
    ):
        """Function to generate the single granger causality plot.

        Args:
            y_variable (str): Name of the target variable
            x_variable (str): Name of the feature variable
            max_lags (int, optional): Number of max lags applied in the granger function. Defaults to 12.
            figsize (tuple, optional): Figure size of the plot in inch. Defaults to (1400, 500).
            streamlit (bool, optional): Select if fig object is returned from function. Defaults to False.
            display_fig (bool, optional): Select if figure is displayed. Defaults to True.

        Returns:
            plotly figure object: Returns plotly figure object if streamlit is true.
        """

        # Generate the Granger Causality
        grange_dict = self.granger_causality_generator(y_variable,
                                                       x_variable,
                                                       max_lags=max_lags)

        fig = go.Figure()
        p_value = grange_dict["P-value"]
        fig.add_trace(
            go.Bar(
                y=grange_dict["F-value"],
                x=grange_dict["Lag-range"],
                orientation="v",
                marker_color="rgba(98,249,252,0.9)",
            ))

        fig.add_trace(
            go.Scatter(
                y=grange_dict["P-value"],
                x=grange_dict["Lag-range"],
                name="P-Value",
                mode="lines+markers",
                marker=dict(size=5, color="#735797"),
            ))

        p_value = grange_dict["P-value"]
        fig.update_layout(
            yaxis=dict(categoryorder="total ascending"),
            title=
            f"Granger Causality: {y_variable.title()} vs {x_variable.title()}",
            autosize=False,
            width=figsize[0],
            height=figsize[1],
            xaxis_title="Lag ->",
            yaxis_title="Granger Causality Score",
            xaxis=dict(tickmode="linear", tick0=1, dtick=1),
            hovermode="x",
            template="plotly_dark",
            margin=dict(l=80, r=30, t=80, b=50),
            plot_bgcolor="#151934",
            paper_bgcolor="#151934",
        )

        if display_fig:
            # NOTE this could also be adjusted to save the fig.
            fig.show(renderer="notebook_connected")
        if streamlit:
            return fig

    def plotly_insider_activity(
            self,
            start_date: str,
            end_date: str,
            figsize=(1400, 500),
            streamlit=False,
            display_fig=True,
            **kwargs,
    ):
        """
        Plot insider activity over time

        Args:
        start_date (str): The start date for the plot's x-axis
        end_date (str): The end date for the plot's x-axis
        figsize (tuple, optional): Tuple of width and height for the plot in inches. Defaults to (1400, 500).
        streamlit (bool, optional): If True, returns the plotly figure object. Defaults to False.
        display_fig (bool, optional): If True, displays the plot. Defaults to True.

        Returns:
        plotly.graph_objs._figure.Figure: Returns the plotly figure object if streamlit is True.
        """

        # Generate the insider activity
        combined_df = self.insider_activity(self.df)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=combined_df["df_count"].index,
                y=self.df.index,
                name="Overall Insider activity",
                line=dict(color="#000000"),
            ))

        fig.add_trace(
            go.Scatter(
                x=combined_df["df_buy_count"].index,
                y=self.df.index,
                name="Buy",
                line=dict(color="#008000"),
            ))

        fig.add_trace(
            go.Scatter(
                x=combined_df["df_sale_count"].index,
                y=self.df.index,
                name="Sale",
                line=dict(color="#FF0000"),
            ))

        fig.add_trace(
            go.Scatter(
                x=combined_df["df_opt_count"].index,
                y=self.df.index,
                name="Option Exercise",
                line=dict(color="#FFFF00"),
            ))
        # Plot aestethics
        fig.update_layout(
            title="Insider activity over time",
            xaxis_title="Date in months",
            yaxis_title="Number of incident",
            autosize=False,
            width=figsize[0],
            height=figsize[1],
            xaxis=dict(range=[start_date, end_date]),
            hovermode="x unified",
            margin=dict(l=80, r=30, t=30, b=50),
        )

        if display_fig:
            # NOTE this could also be adjusted to save the fig.
            fig.show(renderer="notebook_connected")
        if streamlit:
            return fig

    def plotly_individual_insider_activity(
            self,
            figsize=(1400, 500),
            streamlit=False,
            display_fig=True,
            **kwargs,
    ):
        """
        Function to generate the insider activity for every individual using Plotly.

        Args:
        self: The class object.
        figsize (tuple, optional): Figure size of the plot in inch. Defaults to (1400, 500).
        streamlit (bool, optional): Select if fig object is returned from function. Defaults to False.
        display_fig (bool, optional): Select if figure is displayed. Defaults to True.
        **kwargs: Additional keyword arguments passed to the function.

        Returns:
        plotly.graph_objs._figure.Figure: Returns a plotly figure object if streamlit is True. The figure
        shows the overall distribution, buy, sale and option exercise of insider activities for every individual.
        """

        # Generate the Insider activity
        combined_df = self.insider_activity(self.df)
        trans = self.transactions_per_insider(self.df)
        trans_buy = self.transactions_per_insider(combined_df["df_buy"])
        trans_sell = self.transactions_per_insider(combined_df["df_sale"])
        trans_opt = self.transactions_per_insider(combined_df["df_opt"])

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=trans.index,
                y=trans["count"],
                name="overall distribution",
                opacity=0.7,
                marker=dict(color="#e15c46"),
            ))

        fig.add_trace(
            go.Scatter(
                x=trans_buy.index,
                y=trans_buy["count"],
                name="Buy",
                mode="lines+markers",
                marker=dict(size=5, color="#735797"),
            ))

        fig.add_trace(
            go.Scatter(
                x=trans_sell.index,
                y=trans_sell["count"],
                name="Sale",
                mode="lines+markers",
                marker=dict(size=5, color="#7ac74c"),
            ))

        fig.add_trace(
            go.Scatter(
                x=trans_opt.index,
                y=trans_opt["count"],
                name="Option Exercise",
                mode="lines+markers",
                marker=dict(size=5, color="#FF0000"),
            ))

        fig.update_layout(
            title="Distribution of number of insiders per company",
            xaxis_title="Number of Insider events",
            yaxis_title="Number of Insider",
            autosize=False,
            width=figsize[0],
            height=figsize[1],
            # legend=dict(yanchor="top", xanchor="right"),
            hovermode="x unified",
            margin=dict(l=80, r=30, t=30, b=50),
        )

        if display_fig:
            # NOTE this could also be adjusted to save the fig.
            fig.show(renderer="notebook_connected")
        if streamlit:
            return fig

    def plotly_top_contributor(
            self,
            figsize=(1400, 500),
            streamlit=False,
            display_fig=True,
            **kwargs,
    ):
        """
        Function to generate the histogram plot for top insider activity.

        Args:
        figsize (tuple, optional): Figure size of the plot in inch. Defaults to (1400, 500).
        streamlit (bool, optional): Select if fig object is returned from function. Defaults to False.
        display_fig (bool, optional): Select if figure is displayed. Defaults to True.
        **kwargs: Additional parameters passed to the plotly function

        Returns:
        plotly figure object: Returns plotly figure object if streamlit is true.
        """

        # Generate the Top Contributor
        top_contributor = self.top_contributor().sort_values(
            by=["incidents_num"], ascending=False)
        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=top_contributor.index,
                y=top_contributor["incidents_num"],
                opacity=0.5,
                marker=dict(color="#4C9900"),
            ))

        fig.update_layout(
            title="Top contributor for insider activities",
            xaxis_title="Name of insider",
            yaxis_title="Total Number of Insider Incidents",
            autosize=False,
            width=figsize[0],
            height=figsize[1],
            hovermode="x unified",
            margin=dict(l=80, r=30, t=30, b=50),
        )

        if display_fig:
            # NOTE this could also be adjusted to save the fig.
            fig.show(renderer="notebook_connected")
        if streamlit:
            return fig

    def plotly_market_cap(
            self,
            figsize=(1400, 500),
            streamlit=False,
            display_fig=True,
            **kwargs,
    ):
        """
        Function to generate the histogram plot for market capital.

        Args:
        figsize (tuple, optional): Figure size of the plot in inch. Defaults to (1400, 500).
        streamlit (bool, optional): Select if fig object is returned from function. Defaults to False.
        display_fig (bool, optional): Select if figure is displayed. Defaults to True.
        **kwargs: Additional parameters passed to the plotly function

        Returns:
        plotly figure object: Returns plotly figure object if streamlit is true.
        """

        # Generate the market capital
        top_market_cap = self.market_cap().sort_values(by=["Value ($)"],
                                                       ascending=False)

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=top_market_cap["Contributor"],
                y=top_market_cap["Value ($)"],
                opacity=0.5,
                marker=dict(color="#4C9900"),
            ))

        fig.update_layout(
            title="Market Cap for Top Contributor of insider trading",
            xaxis_title="Name of insider",
            yaxis_title="Market Cap in USD",
            autosize=False,
            width=figsize[0],
            height=figsize[1],
            hovermode="x unified",
            margin=dict(l=80, r=30, t=30, b=50),
        )

        if display_fig == True:
            # NOTE this could also be adjusted to save the fig.
            fig.show(renderer="notebook_connected")
        if streamlit == True:
            return fig

    def plotly_market_vs_insider(
        self,
        df_timeseries: pd.DataFrame,
        include: list,
        start_date: str,
        end_date: str,
        figsize=(1400, 500),
        threshold=None,
        streamlit=False,
        display_fig=True,
        **kwargs,
    ):
        """
        Function to generate the plot to compare insider trading and market prices.

        Args:
        df_timeseries (pd.DataFrame): DataFrame containing the market prices
        include (list): List of insider trading activities to include in the plot
        start_date (str): Start date for the plot
        end_date (str): End date for the plot
        figsize (tuple, optional): Figure size of the plot in inch. Defaults to (1400, 500).
        threshold (int, optional): threshold for insider trading value to include in the plot
        streamlit (bool, optional): Select if fig object is returned from function. Defaults to False.
        display_fig (bool, optional): Select if figure is displayed. Defaults to True.
        **kwargs: Additional parameters passed to the plotly function

        Returns:
        plotly figure object: Returns plotly figure object if streamlit is true.
        """

        # Generate the Insider Activity
        combined_df = self.insider_activity(self.df)

        fig = go.Figure()

        # Generate list of color codes
        color = iter(["Red", "Green", "Blue"])
        # Iterate the list of transactions
        for trans in include:
            c = next(color)
            df = "df_" + trans
            # Generate grouping of activities
            grouped_trans = (combined_df[df].groupby(
                combined_df[df]["Date"]).sum(numeric_only=True))
            if threshold:
                grouped_trans = grouped_trans[
                    grouped_trans["Value ($)"] < threshold]
            fig.add_trace(
                go.Bar(
                    x=grouped_trans.index,
                    y=grouped_trans["Value ($)"],
                    name=trans,
                    opacity=0.7,
                    marker=dict(color=c),
                ))

        fig.add_trace(
            go.Scatter(
                x=df_timeseries["Date"],
                y=df_timeseries["Volume"],
                name="S&P 500",
                line=dict(color="rgba(50,50,50,0.2)"),
            ))

        fig.update_layout(
            title="Comparison of Insider trading with market value",
            xaxis_title="S&P 500 Volume",
            yaxis_title="Insider Transaction volume",
            autosize=False,
            width=figsize[0],
            height=figsize[1],
            hovermode="x unified",
            xaxis=dict(range=[start_date, end_date]),
            margin=dict(l=80, r=30, t=30, b=50),
        )

        if display_fig:
            # NOTE this could also be adjusted to save the fig.
            fig.show(renderer="notebook_connected")
        if streamlit:
            return fig

    def plotly_insider_activity_roles(
            self,
            figsize=(1400, 500),
            streamlit=False,
            display_fig=True,
            **kwargs,
    ):
        """
        Function to generate the insider activity plot with respect to roles.

        Args:
        figsize (tuple, optional): Figure size of the plot in inch. Defaults to (1400, 500).
        streamlit (bool, optional): Select if fig object is returned from function. Defaults to False.
        display_fig (bool, optional): Select if figure is displayed. Defaults to True.
        **kwargs: Additional parameters passed to the plotly function

        Returns:
        plotly figure object: Returns plotly figure object if streamlit is true.
        """

        # Generate the cross relation between Date and Relationship
        df_insider = pd.crosstab(self.df["Date"], self.df["Relationship"])

        fig = go.Figure()
        # Generate list of color codes
        color = iter(cm.rainbow(np.linspace(0, 1, len(df_insider.columns))))

        # Iterate roles and color
        for role in df_insider.columns:
            c = next(color)
            fig.add_trace(
                go.Scatter(
                    x=df_insider.index,
                    y=df_insider[role],
                    name=role,
                    mode="lines+markers",
                    marker=dict(size=5, color=c),
                ))
        fig.update_layout(
            title="Insider Activity for different Roles",
            xaxis_title="Date",
            yaxis_title="Number of Insider Incidents",
            autosize=False,
            width=figsize[0],
            height=figsize[1],
            hovermode="x unified",
            margin=dict(l=80, r=30, t=30, b=50),
        )

        if display_fig == True:
            # NOTE this could also be adjusted to save the fig.
            fig.show(renderer="notebook_connected")
        if streamlit == True:
            return fig

    def plotly_insider_activity_timeseries_plot(
        self,
        df_timeseries: pd.DataFrame,
        start_date: str,
        end_date: str,
        figsize=(1400, 500),
        include=False,
        streamlit=False,
        display_fig=True,
        **kwargs,
    ):
        """
        Function to generate the insider activity plot with respect to market prices in terms of volume.

        Args:
        df_timeseries (pd.DataFrame): DataFrame containing the market prices
        start_date (str): Start date for the plot
        end_date (str): End date for the plot
        figsize (tuple, optional): Figure size of the plot in inch. Defaults to (1400, 500).
        include (list): List of insider trading activities to include in the plot, defaults to ["buy","sale","opt"]
        streamlit (bool, optional): Select if fig object is returned from function. Defaults to False.
        display_fig (bool, optional): Select if figure is displayed. Defaults to True.
        **kwargs: Additional parameters passed to the plotly function

        Returns:
        plotly figure object: Returns plotly figure object if streamlit is true.
        """

        # Generate the Insider Activity
        combined_df = self.insider_activity(self.df)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=df_timeseries["Date"],
                y=df_timeseries["Close"],
                name="S&P",
                line=dict(color="rgba(50,50,50,0.2)"),
            ))
        if include == False:
            include = ["buy", "sale", "opt"]
        # Generate list of color codes
        color = iter(["Red", "Green", "Blue"])

        # Iterate the list of transactions
        for trans in include:
            c = next(color)
            df = "df_" + trans
            fig.add_trace(
                go.Scatter(
                    y=combined_df[df]["Cost"],
                    x=combined_df[df]["Date"],
                    name=trans,
                    mode="markers",
                    opacity=0.6,
                    marker=dict(size=15, color=c),
                ))
        fig.update_layout(
            title="Insider activity over time",
            xaxis_title="Date in months",
            yaxis_title="Number of incident",
            autosize=False,
            width=figsize[0],
            height=figsize[1],
            legend=dict(yanchor="top", xanchor="right"),
            xaxis=dict(range=[start_date, end_date]),
            hovermode="x unified",
            margin=dict(l=80, r=30, t=30, b=50),
        )

        if display_fig:
            # NOTE this could also be adjusted to save the fig.
            fig.show(renderer="notebook_connected")
        if streamlit:
            return fig

    def plotly_returns(
        self,
        stock_df: pd.DataFrame,
        include: list,
        returns: str,
        figsize=(1400, 500),
        threshold=False,
        streamlit=False,
        display_fig=True,
        **kwargs,
    ):
        """Function to generate a plotly figure object representing the returns of insider trades and S&P 500 stocks.

        Args:
        stock_df (pd.DataFrame): A DataFrame containing stock data.
        include (list): A list of stock symbols to include in the plot.
        returns (str): A string representing the type of returns to display on the plot.
        figsize (tuple, optional): A tuple representing the size of the plot in inches. Defaults to (1400, 500).
        threshold (bool, optional): A flag indicating whether to use a threshold. Defaults to False.
        streamlit (bool, optional): A flag indicating whether to return the plotly figure object. Defaults to False.
        display_fig (bool, optional): A flag indicating whether to display the figure. Defaults to True.
        **kwargs: Additional keyword arguments.

        Returns:
        plotly figure object: The plotly figure object if streamlit is set to True.
        """

        comb_df = self.show_returns(stock_df,
                                    threshold,
                                    include,
                                    returns_type=returns)
        # Generate list of color codes
        color = iter(["Red", "Green", "Blue"])

        fig = go.Figure()

        for trans, trans_df in comb_df.items():
            c = next(color)
            fig.add_trace(
                go.Box(
                    x=trans_df["day"],
                    y=trans_df["return"],
                    name=trans,
                    opacity=0.5,
                    marker=dict(color=c),
                ))
        fig.update_layout(
            title=
            f"{returns} terms returns on Insider trades and S&P 500 stocks",
            xaxis=dict(title="Return", zeroline=False),
            yaxis=dict(title="Returns in %", zeroline=False),
            autosize=True,
            width=figsize[0],
            height=figsize[1],
            hovermode="x unified",
            boxmode="group",
            margin=dict(l=80, r=30, t=30, b=50),
        )

        if display_fig:
            # NOTE this could also be adjusted to save the fig.
            fig.show(renderer="notebook_connected")
        if streamlit:
            return fig
