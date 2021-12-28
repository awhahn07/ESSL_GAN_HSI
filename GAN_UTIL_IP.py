import os
import csv
import datetime
import matplotlib.pyplot as plt
from pandas import read_csv as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from IP_DATA import IP_DS
'''
Creates directory for test. Returns the dir path to GAN RUN. 
Calls README_MAKE and writer_make 
'''
def make_logger(model):
    date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    path = ".\\Test_{}".format(date)
    if not os.path.exists(path):
        os.mkdir(path, mode = 0o777)
    readme_make(model,path)
    log_path,acc_path,class_path=writer_make(path,date)
    path_dict={"path":path ,"log_path":log_path,
               "acc_path":acc_path,"class_path":class_path}
    return path_dict
'''Create TXT file with Model Parameters of D and G '''

def readme_make(model,path): 
        def txt_print(string):
            with open(read_path,'a') as file:
                file.write(string+'\n')                
        read_path = os.path.join(path,'README.txt')
        model_params = [x for x in model.__dict__.items()]        
        with open(read_path,'a') as file:
            file.write('PARAMETERS:\n')
            for line in model_params:
                file.write(str(line)+'\n')
            file.write('DISCRIMINATOR MODEL: \n')    
        model.discriminator.summary(print_fn=txt_print)
        with open(read_path,'a') as file:
            file.write('GENERATOR MODEL:\n')
        model.generator.summary(print_fn=txt_print)        

'''Create CSV files with headers to log loss and accuracy 
values in main GAN RUN'''        
def writer_make(path,date):    
    log_path = os.path.join(path,'hsi_loss_log_{}'.format(date)+'.csv')
    acc_path = os.path.join(path,'acc_log_{}'.format(date)+'.csv')
    class_path = os.path.join(path,'classifier_loss_{}'.format(date)+'.csv')
    with open(log_path,'a') as log, open(acc_path,'a') as acc:
        Fieldnames1 = ['Epoch','Global_Step','Disc_Loss',
                       'Gen_Loss','Real_Loss','Fake_Loss']
        Fieldnames2 = ['Epoch','Global_Step','p_DS','p_DU','p_GP','p_GE']
        writer1 = csv.DictWriter(log,Fieldnames1)
        writer1.writeheader()
        writer2 = csv.DictWriter(acc,Fieldnames2)
        writer2.writeheader()    
    return log_path,acc_path,class_path
        
'''Write to log'''
def log(data,path):
    with open(path, 'a') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data)
                
'''Create and Save a loss plot'''                 
def loss_plots(path):
    df = pd(path)
    print(df.head())   
    fig,ax = plt.subplots(2,1)
    fig.set_size_inches(8,6)
    ax[0].plot(df['Global_Step'],df['Disc_Loss'])
    ax[0].plot(df['Global_Step'],df['Gen_Loss'])
    ax[0].set(title='D vs G Loss',xlabel='Global Steps',ylabel='Loss')
    ax[0].set_autoscaley_on(0)
    ax[0].legend()
    ax[1].plot(df['Global_Step'],df['Real_Loss'])
    ax[1].plot(df['Global_Step'],df['Fake_Loss'])
    ax[1].plot(df['Global_Step'],df['Disc_Loss'])
    ax[1].set(title='D_Loss, Real and Synthetic Outputs',
              xlabel='Global Steps',ylabel='Loss')
    ax[1].set_autoscaley_on(0)
    ax[1].legend()
    fig.subplots_adjust(hspace=0.5)
    fig.savefig(path[:-4]+'.png')
    plt.close(fig=fig)

def acc_plots(path):
    df = pd(path)
    print(df.head())  
    fig,ax = plt.subplots(2,1)
    fig.set_size_inches(8,8)
    ax[0].plot(df['p_DS'])
    ax[0].plot(df['p_DU'])
    ax[0].set(title='Discriminator Performance',xlabel='Global Steps',
      ylabel='Accuracy')
    ax[0].set_autoscaley_on(0)
    ax[0].legend()
    ax[1].plot(df['p_GP'])
    ax[1].plot(df['p_GE'])
    ax[1].set(title='Generator Performance',xlabel='Global Steps',
      ylabel='Accuracy')
    ax[1].set_autoscaley_on(0)
    ax[1].legend()
    fig.subplots_adjust(hspace=0.5)
    fig.savefig(path[:-4]+'.png') 
    plt.close(fig=fig)
    
def make_dataset(data='',batch=100,random_state=42,split_size=0.3,
                 pca_n=None,remove_null=False,categories=9):
    DataSet = IP_DS()    
    if remove_null == True:
        Data,Lab = DataSet.remove_null()
        num_class=categories+1
        print('Using Scaled Dataset, null removed')
    else:
        num_class=categories+2
        if data == 'raw':
            Data,Lab = DataSet.raw()
            print('Using raw Dataset')
        elif data == 'PCA':
            Data,Lab = DataSet.PCA_n(n_components=pca_n)
            if pca_n == None:
                pca_n = len(Data[1])
            print('Using PCA Dataset, features = {}'.format(pca_n))
        else:
            Data,Lab = DataSet.scaled()
            print('Using Scaled Dataset')
    X_train, X_test, y_train, y_test = train_test_split(
            Data, Lab, test_size=split_size, random_state=random_state)
    train_y = tf.keras.utils.to_categorical(y_train,num_classes=num_class)
    train_X = tf.data.Dataset.from_tensor_slices(X_train)
    train_y = tf.data.Dataset.from_tensor_slices(train_y)
    train_ds = tf.data.Dataset.zip(
            (train_X,train_y)).shuffle(len(X_train)).batch(batch)
    test_X = tf.data.Dataset.from_tensor_slices(X_test)
    y_test = tf.keras.utils.to_categorical(y_test,num_classes=num_class)
    test_y = tf.data.Dataset.from_tensor_slices(y_test)  
    test_ds = tf.data.Dataset.zip(
            (test_X,test_y)).shuffle(len(X_test)).batch(batch)
    class_weights = DataSet.get_weights()
    return train_ds,test_ds,class_weights  

      