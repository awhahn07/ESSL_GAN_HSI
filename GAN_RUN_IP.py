import tensorflow as tf
import time
from GAN_IP_MODEL import GanModel
import GAN_UTIL_IP as util
from keras.utils.generic_utils import Progbar
import numpy as np
tf.enable_eager_execution()

def main(batch_size=100,epochs=10,
         gamma=1,use_class_weights=True):    
    
    EPOCHS = epochs    
    Remove_Null = True
    target_length = 200 
    if Remove_Null==True:
        nb_class = 17    ## include syn class, K+1
    else: 
        nb_class = 18    ## include syn class, K+1
    '''Make DS'''
    train_ds,test_ds,class_weights=util.make_dataset(
            data='scaled',batch=batch_size,remove_null=Remove_Null,
            random_state=np.random.randint(1000),categories=16)       
    test_ds.repeat()
    '''Determine whether to weight classes for imbalanced datasets'''
    if use_class_weights==True:
        class_weights=class_weights
    else: 
        class_weights=1
    '''Instantiate a GAN Model'''
    GAN = GanModel(batch_size=batch_size,gamma=gamma,
                   nb_class=nb_class,target_length=target_length,
                   class_weights=use_class_weights)
    '''Run logging utility, returns dictionary with test directory path
    loss log path, accuracy path, and classifier path'''
    path = util.make_logger(GAN)
    '''Use Tensorflows native global step creator'''
    global_step = tf.train.get_or_create_global_step()
    '''Begin Training Loop'''
    for epoch in range(EPOCHS):
        start = time.time()
        test_iter = test_ds.make_one_shot_iterator()
        next_test = test_iter.get_next()
        print('Epoch %d/%d'%(epoch,EPOCHS))
        if Remove_Null:
            progbar=Progbar(target=7124)
        else:
            progbar=Progbar(target=14718)
        
        for i,(images,labels) in enumerate(train_ds):
            gen_loss,disc_loss,r_loss,f_loss = GAN.train_step(
                    images,labels,global_step,class_weights)
            if (global_step.numpy() % 20) == 0:
                acc = GAN.test(next_test)
                data_acc = [epoch+1,global_step.numpy(),acc['p_DS'],
                            acc['p_DU'],acc['p_GP'],acc['p_GE']]
                util.log(data_acc,path['acc_path'])
                data_loss = [epoch+1, global_step.numpy(), disc_loss.numpy(),
                             gen_loss.numpy(),r_loss.numpy(),f_loss.numpy()]
                util.log(data_loss,path['log_path'])
            progbar.update(i*batch_size)
            
        GAN.generate_and_save_images(epoch,path['path'])
        GAN.discriminator.save_weights(
                path['path']+'/d_weight',save_format='h5')
        GAN.generator.save_weights(
                path['path']+'/g_weight',save_format='h5')
        print ('Time taken for epoch {} is {} sec'.format(
                epoch + 1,time.time()-start))
    util.loss_plots(path['log_path']) 
    util.acc_plots(path['acc_path'])
    
if __name__ == '__main__':    
    tf.enable_eager_execution()
    print('Entering Training')
    main(batch_size=50,gamma=1,epochs=10, use_class_weights=True)
    print('Finish Execution')    
