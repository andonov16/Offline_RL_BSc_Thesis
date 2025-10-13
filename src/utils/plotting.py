import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_hist(data_dict: dict, num_columns: int = 2, f_size: tuple = (16, 9), 
        custom_title: str = None, single_ax: bool = False, hist_kwargs: dict = dict()) -> plt.Figure:
    """
    Plots histograms for given datasets

    Accepts:
        - 'data_dict' (dict): key (str): the value of the given dataset; value(np.array) tha data
        - 'num_columns' (int): number of columns in the subplot grid
        - 'f_size' (tuple): the size of the fig (width, height)
        - 'single_ax' (bool): tells whether each dataset has its own axis or all hists are plotted on the same axis
        - `custom_title` (str): a custom title for the plot (None for some default title)
        - `hist_kwargs` (dict): additional arguments to pass to `plt.hist`

    Returns:
        - `plt.Figure`: a Matplotlib figure with the histogram(s)
    """
    if single_ax:
        return _plot_hist_single_axis(data_dict=data_dict, f_size=f_size, custom_title=custom_title, hist_kwargs=hist_kwargs)
    return _plot_hist_multiple_axes(data_dict=data_dict, num_columns=num_columns, f_size=f_size, custom_title=custom_title, hist_kwargs=hist_kwargs)


def _plot_hist_single_axis(data_dict: dict, f_size: tuple = (16, 9), custom_title: str = None, hist_kwargs: dict = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=f_size)

    if custom_title is None:
        ax.set_title('Histogram')
    else:
        ax.set_title(custom_title)

    ax.set_ylabel('Frequency')
    ax.set_xlabel('Data')

    for label, data in data_dict.items():
        ax.hist(data, label=label, **hist_kwargs)

    ax.legend()
    fig.tight_layout()

    return fig
    

def _plot_hist_multiple_axes(data_dict: dict, num_columns: int = 2, f_size: tuple = (16, 9), custom_title: str = dict(), hist_kwargs: dict = dict()) -> plt.Figure:
    num_rows = int(np.ceil(len(data_dict.keys())/num_columns))
    
    fig, axes = plt.subplots(ncols=num_columns, nrows=num_rows, figsize=f_size)
    axes = axes.flatten()
    
    for ax, data_name in zip(axes, data_dict.keys()):
        if custom_title is None:
            ax.set_title('Histogram: '+ data_name)
        else:
            ax.set_title(custom_title)
        ax.set_ylabel('Frequency')
        ax.set_xlabel('Data')
        ax.hist(data_dict[data_name], **hist_kwargs)
    
    fig.tight_layout()
    
    return fig


def plot_hexbin(data_dict: dict, num_columns: int = 2, f_size: tuple = (16,9)) -> plt.Figure:
    """
    Plots hexbin for given datasets

    Parameters:
        - 'data_dict' (dict): key (str): the value of the given dataset; value(np.array) tha data
        - 'num_columns' (int): number of columns in the subplot grid
        - 'f_size' (tuple): the size of the fig (width, height)

    Returns:
        - `plt.Figure`: A Matplotlib figure with the histogram(s).
    """
    num_rows = int(np.ceil(len(data_dict.keys())/num_columns))
    
    fig, axes = plt.subplots(ncols=num_columns, nrows=num_rows, figsize=f_size)
    axes = axes.flatten()
    
    all_x = np.concatenate([data[:, 0] for data in data_dict.values()])
    all_y = np.concatenate([data[:, 1] for data in data_dict.values()])
    
    x_min, x_max = all_x.min(), all_x.max()
    y_min, y_max = all_y.min(), all_y.max()

    for ax, data_name in zip(axes, data_dict.keys()):
        ax.set_title('Hexbin map: '+ data_name)
        ax.set_ylabel('y')
        ax.set_xlabel('x')
        hb = ax.hexbin(x=data_dict[data_name][:, 0], y=data_dict[data_name][:, 1], cmap='viridis', mincnt=1)
    
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        fig.colorbar(hb, ax=ax, orientation='vertical')
    
    fig.tight_layout()
    
    return fig


def plot_boxplots_df(data: pd.DataFrame, num_columns: int = 4, f_size: tuple = (16, 9), custom_title: str = None, boxplot_kwargs: dict = dict()) -> plt.Figure:
    """
    Plots boxplots for each column(feature) of a given pd.Dataframe

    Accepts:
        - 'data' (pd.DataFrame): the dataset where each column is a feature
        - 'num_columns' (int): number of columns in the subplot grid
        - 'f_size' (tuple): the size of the fig (width, height)
        - 'single_ax' (bool): tells whether each dataset has its own axis or all hists are plotted on the same axis
        - `custom_title` (str): a custom title for the plot (None for some default title)
        - `boxplot_kwargs` (dict): additional arguments to pass to `data.boxplot`

    Returns:
        - `plt.Figure`: a Matplotlib figure with the boxplots
    """
    num_rows = int(np.ceil(len(data.columns)/num_columns))
    
    fig, axes = plt.subplots(ncols=num_columns, nrows=num_rows, figsize=f_size)
    axes = axes.flatten()
    
    if custom_title is None:
        fig.suptitle(f'Boxplots')
    else:
        fig.suptitle(custom_title)
    
    for i, column in enumerate(data.columns):
        if boxplot_kwargs is None:
            data.boxplot(column=[column], ax=axes[i])
        else:
            data.boxplot(column=[column], ax=axes[i], **boxplot_kwargs)
        axes[i].set_title(f'Boxplot of {column}')
    fig.tight_layout()
    return fig


def plot_boxplots_compare_df(df1_name: str, df2_name: str, df1: pd.DataFrame, df2: pd.DataFrame, num_columns: int = 4,
                            f_size: tuple = (16, 9), custom_title: str = None, boxplot_kwargs: dict = dict()) -> plt.Figure:
    """
    Plots boxplots for each column(feature) of two datasets side by side

    Accepts:
        - 'df1_name' (str): the name (prefix) of the first dataset
        - 'df2_name' (str): the name (prefix) of the second dataset
        - 'df1' (pd.DataFrame): the first dataset where each column is a feature
        - 'df2' (pd.DataFrame): the second dataset where each column is a feature
        - 'num_columns' (int): number of columns in the subplot grid
        - 'f_size' (tuple): the size of the fig (width, height)
        - `custom_title` (str): a custom title for the plot (None for some default title)
        - `boxplot_kwargs` (dict): additional arguments to pass to `data.boxplot`

    Returns:
        - `plt.Figure`: a Matplotlib figure with the boxplots
    """
    num_rows = int(np.ceil(len(df1.columns)/num_columns))
    
    fig, axes = plt.subplots(ncols=num_columns, nrows=num_rows, figsize=f_size)
    axes = axes.flatten()
    
    df1_renamed = df1.rename(columns={col: f"{df1_name}_{col}" for col in df1.columns})
    df2_renamed = df2.rename(columns={col: f"{df2_name}_{col}" for col in df2.columns})
    df_merged = pd.concat([df1_renamed, df2_renamed], axis=1)
    
    if custom_title is None:
        fig.suptitle(f'Boxplots')
    else:
        fig.suptitle(custom_title)
    
    for i, column in enumerate(df1.columns):
        if boxplot_kwargs is None:
            df_merged.boxplot(column=[df1_name+'_'+column, df2_name+'_'+column], ax=axes[i])
        else:
            df_merged.boxplot(column=[df1_name+'_'+column, df2_name+'_'+column], **boxplot_kwargs)
        axes[i].set_title(f'Boxplot of {column}')
    fig.tight_layout()
    return fig

# a helper function to determine whether to plot a KDE or Histogram for the given pd.DataFrame column
def is_continuous(series):
    return pd.api.types.is_numeric_dtype(series) and series.nunique() > 10 


def plot_univariate_analysis(df1: pd.DataFrame, df2: pd.DataFrame, df1_name: str = 'Dataset 1',
        df2_name: str = 'Dataset 2', num_columns=3, custom_title='Univariate Analysis', f_size: tuple = (18, 21),
        kde_kwargs: dict = dict(), hist_kwargs: dict = dict()) -> plt.Figure:
    """
    Plots a univariate analysis for each column of the two datasets first separately and then both on a single axis.
    KDE is used for continuous data and Histogram for discrete data (discrete <= 10 unique values see is_continuous function).

    Accepts:
        - 'df1' (pd.DataFrame): the first dataset where each column is a feature
        - 'df2' (pd.DataFrame): the second dataset where each column is a feature
        - 'df1_name' (str): the name (prefix) of the first dataset
        - 'df2_name' (str): the name (prefix) of the second dataset
        - 'num_columns' (int): number of columns in the subplot grid
        - 'f_size' (tuple): the size of the fig (width, height)
        - `custom_title` (str): a custom title for the plot (None for some default title)
        - `kde_kwargs` (dict): additional arguments to pass to `.plot(kind='kde'...)`
        - `hist_kwargs` (dict): additional arguments to pass to `.plot(kind='kde'...)`

    Returns:
        - `plt.Figure`: a Matplotlib figure with the univariate analysis of the two datasets
    """
    num_rows = int(np.ceil(len(df1.columns)*3/num_columns))
    
    fig, axes = plt.subplots(ncols=num_columns, nrows=num_rows, figsize=f_size)
    axes = axes.flatten()
    
    df1_renamed = df1.rename(columns={col: f"{df1_name}_{col}" for col in df1.columns})
    df2_renamed = df2.rename(columns={col: f"{df2_name}_{col}" for col in df2.columns})
    df_merged = pd.concat([df1_renamed, df2_renamed], axis=1)
    
    if custom_title is None:
        fig.suptitle(f'Univariate Analysis')
    else:
        fig.suptitle(custom_title)
    
    i = 0
    for column in df1.columns:
        if is_continuous(df_merged[df1_name + '_' + column]) and is_continuous(df_merged[df2_name + '_' + column]):
            df1[[column]].plot(kind="kde", ax=axes[i], **kde_kwargs)
            axes[i].set_title(f'KDE: {column} ({df1_name} only)')
            i += 1
            df2[[column]].plot(kind="kde", ax=axes[i], **kde_kwargs)
            axes[i].set_title(f'KDE: {column} ({df2_name} only)')
            i += 1
            df_merged[[df1_name + '_' + column, df2_name + '_' + column]].plot(kind="kde", ax=axes[i], **kde_kwargs)
            axes[i].set_title(f'KDE: {column} (both datasets)')
            i += 1
        else:
            df1[[column]].plot(kind="hist", ax=axes[i], **hist_kwargs)
            axes[i].set_title(f'Histogram: {column} ({df1_name} only)')
            i += 1
            
            df2[[column]].plot(kind='hist', ax=axes[i], **hist_kwargs)
            axes[i].set_title(f'Histogram: {column} ({df2_name} only)')
            i += 1
            
            df_merged[[df1_name + '_' + column, df2_name + '_' + column]].plot(kind="hist", ax=axes[i], **hist_kwargs)
            axes[i].set_title(f'Histogram: {column} (both datasets)')
        
    for ax in axes:
        ax.set_xlabel('Values')
        
    fig.tight_layout()
    return fig


def plot_legs_bar(df1: pd.DataFrame, df2: pd.DataFrame, df1_name: str = 'Dataset 1', df2_name: str = 'Dataset 2',
        custom_title: str = 'Barchart', f_size: tuple = (16,6), bar_kwargs: dict = dict()) -> plt.Figure:
    """
    A simple barchart that shows the information for legs count in the final states of each dataset

    Accepts:
        - 'df1' (pd.DataFrame): the first dataset where each column is a feature
        - 'df2' (pd.DataFrame): the second dataset where each column is a feature
        - 'df1_name' (str): the name (prefix) of the first dataset
        - 'df2_name' (str): the name (prefix) of the second dataset
        - 'f_size' (tuple): the size of the fig (width, height)
        - `custom_title` (str): a custom title for the plot (None for some default title)
        - `bar_kwargs` (dict): additional arguments to pass to `.bar(...)'

    Returns:
        - `plt.Figure`: a Matplotlib figure with the barchart with 3 axes (1 for the first dataframe, 1 for the second
        dataframe and 1 for both of them combined on a single axis)
    """
    fig, axes = plt.subplots(ncols=3, nrows=1, figsize=f_size)
    axes = axes.flatten()
    
    df1_counts = df1.apply(lambda x: x.sum()) 
    df2_counts = df2.apply(lambda x: x.sum()) 
    
    fig.suptitle(custom_title)
    
    axes[0].bar(df1_counts.index, df1_counts.values, **bar_kwargs)
    axes[0].set_title(f'Legs statistics: {df1_name}')
    axes[0].set_ylabel('times touched the ground')
    
    axes[1].bar(df2_counts.index, df2_counts.values, **bar_kwargs)
    axes[1].set_title(f'Legs statistics: {df2_name}')
    axes[1].set_ylabel('times touched the ground')
    
    bars1 = axes[2].bar(df1_counts.index, df1_counts.values, **bar_kwargs)
    bars2 = axes[2].bar(df2_counts.index, df2_counts.values, **bar_kwargs)
    axes[2].set_title('Legs statistics: (both)')
    axes[2].legend([bars1, bars2], [df1_name, df2_name]) 
    axes[1].set_ylabel('times touched the ground')
    
    fig.tight_layout()
    return fig
