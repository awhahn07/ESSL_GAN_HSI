import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io as io
from tensorflow import keras as k

class GanModel(object):    
    def __init__(self,binary=False,batch_size=100,noise_dim=100,
                 gamma=1,nb_class=18,target_length=None,
                 class_weights=True, glr=5e-4, dlr=5e-4): 
        self.batch_size = batch_size
        self.noise_dim = noise_dim
        self.target_length = target_length
        assert(target_length != None)
        self.gamma = gamma  #aka Gamma in thesis
        self.class_weights = class_weights
        self.nb_class = nb_class  #include fake class            
        self.gen_learning_rate = glr
        self.disc_learning_rate = dlr
        self.make_discriminator_model()
        self.make_generator_model()
        self.gen_optimizer()
        self.disc_optimizer()
            
    def make_generator_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(1024, use_bias=False, 
                input_shape=(self.noise_dim+self.nb_class,),
                activation='tanh'))
        model.add(tf.keras.layers.Dense(6400,activation='tanh'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Reshape((50,128)))
        model.add(tf.keras.layers.UpSampling1D(size=2))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv1D(
                64, 5, padding='same',activation='tanh'))
        model.add(tf.keras.layers.UpSampling1D(size=2))
        model.add(tf.keras.layers.Conv1D(
                1, 5, padding='same',activation='tanh'))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(self.target_length))
        model.add(tf.keras.layers.Reshape([self.target_length,1]))
        self.generator = model
        self.generator.summary()
    
    def make_discriminator_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv1D(
                64, 5, padding='same', activation='relu',
                input_shape=[self.target_length,1], name = 'CONV1'))
        model.add(tf.keras.layers.Conv1D(
                64, 5, padding='valid', activation='relu',name = 'CONV2'))
        model.add(tf.keras.layers.MaxPool1D())
        model.add(tf.keras.layers.Conv1D(
                64, 5, padding='valid', activation='relu',name = 'CONV3'))
        model.add(tf.keras.layers.Conv1D(
                64, 5, padding='valid', activation='relu',name = 'CONV4'))
        model.add(tf.keras.layers.MaxPool1D())
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(1024,name = 'DEN1'))
        model.add(tf.keras.layers.Dense(self.nb_class))          
        self.discriminator=model
        self.discriminator.summary()
    
    def generator_vector(self):
        ''' Generate vector of random ints in range of all valid TRUE classes,
        i.e. Classes C_0...C_n. Fake class is C_n+1  '''
        gen_lab=np.random.randint(0,self.nb_class-1,size=self.batch_size)
        gen_lab = k.utils.to_categorical(gen_lab, num_classes=self.nb_class)
        '''Generate noise vector of size batch_size,noise_dim'''
        noise = np.random.normal(
                size=[self.batch_size,self.noise_dim]).astype('float32')
        '''Return labels encoded into noise'''
        return np.append(noise,gen_lab,axis=1),gen_lab
    
    def generator_loss(self,generated_output,lab,weights=1):
        gen_loss = tf.losses.softmax_cross_entropy(
                lab,generated_output, label_smoothing=0.1,weights=1)
        return gen_loss
 
    def discriminator_loss(self, real_output, generated_output,
                           train_lab,weights=1):
        '''Compute real loss'''
        real_loss = tf.losses.softmax_cross_entropy(
                train_lab,real_output,label_smoothing=0.1,weights=weights)
        '''Compute synthetic loss'''
        gen_loss_labels = tf.ones(self.batch_size)*(self.nb_class-1)
        gen_loss_labels = k.utils.to_categorical(gen_loss_labels,
                                                 num_classes=self.nb_class)
        generated_loss = tf.losses.softmax_cross_entropy(
                gen_loss_labels,generated_output,label_smoothing=0.1)
        total_loss = real_loss + self.gamma*generated_loss
        return total_loss,real_loss,generated_loss
        
    def make_class_weights(self,labels,class_weights):
        if class_weights==None:
            return 1
        ind = [np.argmax(lab) for lab in labels]
        weights = [class_weights[i] for i in ind]    
        return weights  
    
    def gen_optimizer(self):
        self.generator_optimizer = tf.train.AdamOptimizer(
                self.gen_learning_rate)
        print("Gen Optimizer Learning Rate: {}".format(
                self.gen_learning_rate))
        
    def disc_optimizer(self):
        self.discriminator_optimizer = tf.train.AdamOptimizer(
                self.disc_learning_rate)
        print("Disc Optimizer Learning Rate: {}".format(
                self.disc_learning_rate))
        
    def train_step(self,images,labels,global_step,class_weights=None):
        ''' Generate a vector of size([Batch_Size,Noise_Dim+Num_Classes]). 
        This vector has random noise of size Noise_Dim with a Sparse Vector
        that cooresponds to a randomly selected label appended. Each vector 
        Z fed into Generator G has a size of Noise_Dim + Class Size.
        Then it is batched into Batch_Size
        ''' 
        Batch_Noise_Labeled,syn_labels = self.generator_vector()
        ''' This is the Tensorflow implementation to control at lower levels
        the forward propogation and subsequent backprop of gradients.
        The Training step is implemented in a context manager with statement, 
        wherethe gradients of D and G are stored in "GradientTapes" '''
        real_loss_weights = self.make_class_weights(labels,class_weights)
        syn_loss_weights = self.make_class_weights(syn_labels,class_weights)
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:    
            '''Generate Images from G network'''
            generated_images = self.generator(
                    Batch_Noise_Labeled, training=True) 
            '''Evaluate both Real and Synthetic images, D(x) and D(G(z))'''
            real_output = self.discriminator(images, training=True)
            generated_output = self.discriminator(
                    generated_images, training=True) 
            '''Compute Losses using Categorical Crossentropy'''
            gen_loss = self.generator_loss(generated_output,syn_labels,
                                           weights=syn_loss_weights)
            disc_loss,real_loss,fake_loss = self.discriminator_loss(
                    real_output, generated_output, labels,
                    weights=real_loss_weights)
        '''calculate gradients'''    
        gradients_of_generator = gen_tape.gradient(
                gen_loss, self.generator.variables)
        gradients_of_discriminator = disc_tape.gradient(
                disc_loss, self.discriminator.variables)
        '''Using Adam optimizer, backpropogate gradients'''
        self.generator_optimizer.apply_gradients(
                zip(gradients_of_generator, self.generator.variables),
                    global_step=global_step)
        self.discriminator_optimizer.apply_gradients(
                zip(gradients_of_discriminator,
                    self.discriminator.variables),
                    global_step=global_step)
        return gen_loss,disc_loss,real_loss,fake_loss
    
    def generate_and_save_images(self,epoch,path):
        gen_vec = np.random.normal(size=[self.nb_class-1,100])
        lab_enc = k.utils.to_categorical(np.arange(self.nb_class-1),
                                         num_classes=self.nb_class)
        gen_vec=np.append(gen_vec,lab_enc,axis=1)
        predictions = self.generator.predict(gen_vec)
        pred = np.squeeze(predictions)
        fig,ax = plt.subplots(6,3,sharex=True,sharey=True)
        fig.set_size_inches(10,15)  
        for i,axes in enumerate(ax.flat):
            if i < len(predictions):
                axes.plot(pred[i, :])
                axes.set_title('Spectrum {}'.format(i))
            axes.axis('on')
            axes.grid(True)
        #fig.show()
        fig.savefig(path+'/image_at_epoch_{}.png'.format(epoch))
        plt.close(fig=fig)
        
    def generate_and_save_signals(self,runs,epoch,path):
        for run in range(runs):
            g = self.generator(self.generator_vector())
            lab = self.gen_labels()
            Data = np.squeeze(g)
            lab = np.squeeze(lab)
            lab_index=[]
            c = self.nb_class-1
            for j in range(c):
                ind = [i for i, x in enumerate(lab) if x==j]
                lab_index.append(ind)    
            Segment = [Data[lab_index[i]] for i in range(c)]
            if run == 0:
                Signals = Segment
            else:
                Signals = [np.append(Signals[ind],Segment[ind],axis=0) 
                    for ind in range(c)]
        mdict={'Signals':Signals}
        io.savemat(os.path.join(
                path,'signal_at_epoch_{}'.format(epoch)),mdict)
        return Signals
    
    def test(self,dataset_element):
        im,lab = dataset_element
        '''Create a generator Vector with Labels'''
        gv,l = self.generator_vector()
        '''Calcualte G(z), aka generated images batch'''
        gz = self.generator.predict(gv)
        '''Calculate D(x), from real Images'''
        dx = self.discriminator.predict(im.numpy())
        '''Calculate D(G(z)) from gen ims'''
        dgz = self.discriminator.predict(gz)
        '''Create label vector of all K+1 labels, 
        i.e., labels vector indicating all generated imagery'''
        syn_lab = np.zeros_like(l)
        syn_lab[:,-1]=1
        '''Convert from logits to probabilites for 
        D(X) and D(G(Z)) and extract values'''
        syn_pred = tf.nn.softmax(dgz)
        true_pred = tf.nn.softmax(dx)
        syn_pred = syn_pred.numpy()
        true_pred = true_pred.numpy()
        
        acc = tf.keras.metrics.categorical_accuracy
        mean = tf.math.reduce_mean
        '''Compute mean accuracy for the following:
            p_{DS}: Accuracy of D in determining True 
                    images true classes
            p_{GP}: G getting D to predict its desired class 
            p_{DU}: How often D predicts G(z) is synthetic 
            p_{GE}: How often D predicts G to be real but
                    NOT the intended class: 
                    G_c = (!G_precision & D_false_real)'''
        gp = acc(l,dgz)
        dts = acc(syn_lab,dgz)
        dfr = (dts.numpy()==0).astype(int)
        gp_not = (gp.numpy()==0).astype(int)
        p_DS = mean(acc(lab,true_pred)).numpy()
        p_DU = mean(dts).numpy()
        p_GP = mean(gp).numpy()
        p_GE = np.mean(dfr & gp_not)
        
        return {'p_DS':p_DS,'p_GP':p_GP,'p_DU':p_DU,'p_GE':p_GE} 
        