import matplotlib.pyplot as plt
import datetime
from pandas.plotting import register_matplotlib_converters
import seaborn as sns
import numpy as np
import os
from statsmodels.tsa.stattools import adfuller

sns.set()
register_matplotlib_converters()

logo_dir = 'Logos'
Bitcoin_logo = os.path.join(logo_dir, "Bitcoin_logo.png")
crypto_logo = os.path.join(logo_dir, "cryptocurrency.png")
stocks_logo = os.path.join(logo_dir, "stocks.png")
metals_logo = os.path.join(logo_dir, "metals.png")
google_logo = os.path.join(logo_dir, "google.png")

# Function for plotting a graph where the x-axis is the date
def draw_xdate(data, ax = None, ylabel = None, im_path = None,
               years = range(2014,2020), months = [1,7], size=(12,7),
               text = None, title = None):
    """
    data: data to plot
    ax: plot on existing axes
    ylabel: to show vertically near the y-axis
    im_path: image for pyplot.figure background (only if ax is None)
    years: years to plot
    months: months to plot
    size: size of the pyplot.figure (only if ax is None)
    text: text to add somhwere in the axes (such as plot info)
    title: axes title
    """
    
    if ax is None:
        # create new figure
        fig, ax = plt.subplots(figsize=size)
        fig.tight_layout()

    ax.plot(data['date'], data.drop('date', axis=1))
    if ylabel is not None:
        ax.set_ylabel(ylabel,fontsize=15)
    
    # set up x-axis
    ax.set_xlabel('Date',fontsize=15)
    ax.set_xticks([datetime.date(i,j,1) for i in years for j in months])
    ax.set_xticklabels([datetime.date(i,j,1).strftime('%b %Y')
                        for i in years for j in months])
    
    if text is not None:
        # add text as annotation
        ax.annotate(text, xy=(0.1, 0.8), xycoords='axes fraction',
                    xytext=(0.4, 0.8), textcoords='axes fraction',
                    fontsize=16, color='red')
    
    if title is not None:
        ax.set_title(title, fontsize=25)
    
    if im_path is not None:
        # set background
        im = plt.imread(im_path)
        x0,x1 = ax.get_xlim()
        y0,y1 = ax.get_ylim()
        ax.imshow(im, extent=[x0, x1, y0, y1], aspect='auto', alpha=.4)
        # ax.imshow(im)#, 200, 50, zorder=3, alpha=.2)
    
    return ax

# Function for plotting both prediction and true-value on the same plot
def draw_pred_res(data, dataset_name = '', ax=None, im_path=None):
   
    # infering years to plot from data
    start, end = data['date'].iloc[0].year, data['date'].iloc[-1].year
    plot_years = list(range(start, end+1))
    
    # august is the dirst month in the test data
    plot_months = [8]
    title = dataset_name + ' Prediction'

    mae= np.absolute(np.array(data['price'].values-data['pred'].values)).mean()
    ax = draw_xdate(data, years=plot_years, ax=ax, im_path=im_path,
                    months=plot_months, ylabel='Closing Price ($)',
                    text=f"Mean Absolute Error ={mae:.4f}", title=title)
    ax.legend(['Actual Price', 'Predicted Price'])
    return ax

# function for plotting the training error in the training epochs
def plot_training_error(train_res):
    """
    train_res: output of model.fit (Keras style)
    """
    fig, ax = plt.subplots(figsize=(4,4))
    ax.plot(train_res.epoch, train_res.history['loss'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Model Loss')
    ax.set_title('Training Error')
    fig.tight_layout()
    plt.show()

def plot_adf(data):
    """
    data: df of dfs
    """
    plt.plot(data[:,:,0].flatten())
    result = adfuller(data[:,:,0].flatten())
    plt.title('P-Value for Closing Crice: %f' % result[1])
    plt.show()
    
    plt.plot(data[:,:,1].flatten())
    result = adfuller(data[:,:,1].flatten())
    plt.title('P-Value for Trading Volume: %f' % result[1])
    plt.show()

# function for plotting boxplot of errors to compare between models
def plot_boxplot(preds, names):
    """
    preds: array of arrays containing mae for each iteration, for each model
    names: name of models (same lengnth as preds)
    """

    min_len = min(len(x) for x in preds)
    preds = [x[:min_len] for x in preds]

    fig, ax = plt.subplots()
    ax.boxplot(preds, widths=0.75)
    ax.set_xticklabels(names)
    ax.set_title(f'Bitcoin Test Set ({min_len} runs)')
    ax.set_ylabel('Mean Absolute Error (MAE)',fontsize=12)
    fig.tight_layout()
    plt.show()

def plot_train_test_data(train_data, test_data, im=None):
        start = train_data['date'].iloc[0].year
        end = test_data['date'].iloc[-1].year
        plot_years = list(range(start,end+1))
        ax = draw_xdate(train_data, years=plot_years, im_path=im)
        ax = draw_xdate(test_data, years=plot_years, ax=ax)
        cols = train_data.drop('date', axis=1).columns
        ax.legend([c +' Train' for c in cols] + [c +' Test' for c in cols])
        plt.show()

# plot heatmap for the pearson correlation between the data items
def plot_pearson_correlation(train_data):
    """
    train_data: the training data df
    """
    corr = train_data.drop(columns='date').corr(method='pearson')
    fig, ax = plt.subplots()
    sns.heatmap(corr, ax=ax, cbar=False, annot=True)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
