import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import load_model
from keras_preprocessing.image import load_img
from keras_preprocessing.image import img_to_array
from keras.preprocessing import image
from tensorflow.keras.layers import Dense, Activation,Dropout,Conv2D, MaxPooling2D, Flatten,Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model, Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import seaborn as sns
sns.set_style('darkgrid')
import io
import base64
import random
from flask import Flask, render_template, redirect, request, url_for, send_file


app = Flask(__name__, template_folder='templates', static_folder='static')

app.config["IMAGE_UPLOADS"] = "static/images/"
model_path = ("CNN-pisang-87.5.h5")
model = load_model(model_path)

@app.route("/", methods= ["GET"])
def index():
    return render_template('index.html')

@app.route("/", methods= ["POST", "GET"])
def klasifikasi():
    image = request.files['image']
    img_path = (os.path.join(app.config["IMAGE_UPLOADS"], image.filename))
    image.save(img_path)

    labels = ['Pisang Matang', 'Pisang Mentah', 'Pisang Sangat Matang']
    image = tf.keras.utils.load_img(img_path, target_size=(200, 200))
    x = tf.keras.utils.img_to_array(image)
    x = np.expand_dims(x, axis = 0)
    image = np.vstack([x])
    classes = model.predict(image, batch_size = 12)
        
    if (classes[0][0] == 1):
        print("Gambar ini masuk ke dalam kelas", labels[0])
    elif (classes[0][1] == 1):
        print("Gambar ini masuk ke dalam kelas", labels[1])
    elif (classes[0][2] == 1):
        print("Gambar ini masuk ke dalam kelas", labels[2])

    prediksi = labels[np.argmax(classes)]
        # klasifikasi = '{:2.0f}%'.format(100 * np.max(prediksi))

    return render_template('index.html', img_path = img_path,
                           klasifikasi = '{:2.0f}%'.format(100 * np.argmax(prediksi)),
                           prediksi = prediksi)

@app.route("/model", methods=['POST', 'GET'])
def laporan():
    df = None
    lengths = {}
    augmented = None
    ori = None
    # encoded_image = None 
    plot_filename = None
    cm = None
    clr = None

    if request.method == 'POST':
        a = request.form['epoch']
        epochs = int(a)

        # baca data
        sdir=r'static/pisang'

        filepaths=[]
        labels=[]
        classlist=os.listdir(sdir)
        for klass in classlist:
            classpath=os.path.join(sdir,klass)
            if os.path.isdir(classpath):
                flist=os.listdir(classpath)
                for f in flist:
                    fpath=os.path.join(classpath,f)
                    filepaths.append(fpath)
                    labels.append(klass)                   
        Fseries= pd.Series(filepaths, name='filepaths')
        Lseries=pd.Series(labels, name='labels')    
        df=pd.concat([Fseries, Lseries], axis=1)
            
        # split data
        train_split=.7
        test_split=.15
        dummy_split=test_split/(1-train_split)
        train_df, dummy_df=train_test_split(df, train_size=train_split, shuffle=True, random_state=1)
        test_df, valid_df=train_test_split(dummy_df, train_size=dummy_split, shuffle=True, random_state=1)
        lengths = {
            'train_df': len(train_df),
            'test_df': len(test_df),
            'valid_df': len(valid_df)
        }

        height=200
        width=200
        channels=3
        batch_size=2

        img_shape=(height, width, channels)
        img_size=(height, width)
        length=len(test_df)
        test_batch_size=sorted([int(length/n) for n in range(1,length+1) if length % n ==0 and length/n<=80],reverse=True)[0]  
        test_steps=int(length/test_batch_size)


        gen=ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        train_datagen = ImageDataGenerator(rescale=1/255.)
        train_data = train_datagen.flow_from_dataframe(train_df,
                                                       x_col='filepaths',
                                                       y_col='labels',
                                               target_size=img_size,
                                               batch_size=batch_size,
                                               class_mode='categorical',
                                               shuffle=False)
        
        train_gen=gen.flow_from_dataframe( train_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical',
                                            color_mode='rgb', shuffle=False, batch_size=batch_size)

        validgen=ImageDataGenerator(
            rescale=1./255
        )
        valid_gen=validgen.flow_from_dataframe( valid_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical',
                                            color_mode='rgb', shuffle=True, batch_size=batch_size)

        testgen=ImageDataGenerator(
                rescale=1./255
        )
        test_gen=testgen.flow_from_dataframe( test_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical',
                                            color_mode='rgb', shuffle=False, batch_size=test_batch_size)

        classes=list(train_gen.class_indices.keys())

        class_count=len(classes)

        images, labels = train_data.next()
        augmented_images, augmented_labels = train_gen.next()

        # Original vs Augmented
        random_number = random.randint(0, 1)
        plt.imshow(images[random_number])
        plt.title(f"Original image")
        plt.axis(False)
        ori = 'static/ori.png'
        plt.savefig(ori)
        plt.figure()
        plt.imshow(augmented_images[random_number])
        plt.title(f"Augmented image")
        plt.axis(False)
        augmented = 'static/aug.png'
        plt.savefig(augmented)

        model_name='CNN'
        # print("Building model with", base_model)

        model = tf.keras.Sequential([
                    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
                    # This is the first convolution
                    tf.keras.layers.Conv2D(filters=32, padding='same',input_shape=img_shape, kernel_size=(3,3), activation='relu', strides=1),
                    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

                    # second convoluiton
                    tf.keras.layers.Conv2D(filters=64, padding='same',input_shape=img_shape, kernel_size=(3,3), activation='relu', strides=1),
                    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

                    # third convolution
                    tf.keras.layers.Conv2D(filters=128, padding='same',input_shape=img_shape, kernel_size=(3,3), activation='relu', strides=1),
                    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dropout(rate=0.3),
                    tf.keras.layers.Dense(class_count, activation='softmax')
        ])

        # feature extraction
        # def extract():
        #     img_path = 'static/pisang/matang/matang1.jpg'

        #     def get_img_array(img_path, target_size):
        #         img = keras.utils.load_img(
        #             img_path, target_size=target_size)
        #         array = keras.utils.img_to_array(img)
        #         array = np.expand_dims(array, axis=0)
        #         return array

        #     img_tensor = get_img_array(img_path, target_size=(200, 200))

        #     layer_outputs = []
        #     layer_names = []
        #     for layer in model.layers:
        #         if isinstance(layer, (layers.Conv2D, layers.MaxPooling2D)):
        #             layer_outputs.append(layer.output)
        #             layer_names.append(layer.name)
        #     activation_model = keras.Model(inputs=model.input, outputs=layer_outputs)
        #     activations = activation_model.predict(img_tensor)
        #     first_layer_activation = activations[0]
        #     print(first_layer_activation.shape)

        #     images_per_row = 16
        #     for layer_name, layer_activation in zip(layer_names, activations):
        #         n_features = layer_activation.shape[-1]
        #         size = layer_activation.shape[1]
        #         n_cols = n_features // images_per_row
        #         display_grid = np.zeros(((size + 1) * n_cols - 1,
        #                                 images_per_row * (size + 1) - 1))
        #         for col in range(n_cols):
        #             for row in range(images_per_row):
        #                 channel_index = col * images_per_row + row
        #                 channel_image = layer_activation[0, :, :, channel_index].copy()
        #                 if channel_image.sum() != 0:
        #                     channel_image -= channel_image.mean()
        #                     channel_image /= channel_image.std()
        #                     channel_image *= 64
        #                     channel_image += 128
        #                 channel_image = np.clip(channel_image, 0, 255).astype("uint8")
        #                 display_grid[
        #                     col * (size + 1): (col + 1) * size + col,
        #                     row * (size + 1) : (row + 1) * size + row] = channel_image
        #         scale = 1. / size
        #         plt.figure(figsize=(scale * display_grid.shape[1],
        #                             scale * display_grid.shape[0]))
        #         plt.title(layer_name)
        #         plt.grid(False)
        #         plt.axis("off")
        #         plt.imshow(display_grid, aspect="auto", cmap="viridis")

        #         static_folder = os.path.join(app.root_path, "static")
        #         save_path = os.path.join(static_folder, f"{layer_name}.png")
        #         plt.savefig(save_path, format='png')
                
        #         image_stream = io.BytesIO()
        #         plt.savefig(image_stream, format='png')
        #         image_stream.seek(0)
                
        #         encoded_image = base64.b64encode(image_stream.getvalue()).decode('utf-8')
                
        #         return encoded_image

        # training
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.001), loss='categorical_crossentropy', metrics='accuracy')

        model.summary()

        mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)  
        early_stopping = EarlyStopping(monitor='val_loss', patience=3)
        history=model.fit(x=train_gen, epochs=epochs, validation_data=valid_gen,callbacks=[early_stopping, mc])

        # grafik
        def print_in_color(txt_msg,fore_tupple,back_tupple):
            #prints the text_msg in the foreground color specified by fore_tupple with the background specified by back_tupple 
            #text_msg is the text, fore_tupple is foregroud color tupple (r,g,b), back_tupple is background tupple (r,g,b)
            rf,gf,bf=fore_tupple
            rb,gb,bb=back_tupple
            msg='{0}' + txt_msg
            mat='\33[38;2;' + str(rf) +';' + str(gf) + ';' + str(bf) + ';48;2;' + str(rb) + ';' +str(gb) + ';' + str(bb) +'m' 
            print(msg .format(mat), flush=True)
            print('\33[0m', flush=True) # returns default print color to back to black
                

        def tr_plot(tr_data, start_epoch):
            #Plot the training and validation data
            tacc=tr_data.history['accuracy']
            tloss=tr_data.history['loss']
            vacc=tr_data.history['val_accuracy']
            vloss=tr_data.history['val_loss']
            Epoch_count=len(tacc)+ start_epoch
            Epochs=[]
            for i in range (start_epoch ,Epoch_count):
                Epochs.append(i+1)   
            index_loss=np.argmin(vloss)#  this is the epoch with the lowest validation loss
            val_lowest=vloss[index_loss]
            index_acc=np.argmax(vacc)
            acc_highest=vacc[index_acc]
            plt.style.use('fivethirtyeight')
            sc_label='best epoch= '+ str(index_loss+1 +start_epoch)
            vc_label='best epoch= '+ str(index_acc + 1+ start_epoch)
            fig,axes=plt.subplots(nrows=1, ncols=2, figsize=(10,6))
            axes[0].plot(Epochs,tloss, 'r', label='Training loss')
            axes[0].plot(Epochs,vloss,'g',label='Validation loss' )
            axes[0].scatter(index_loss+1 +start_epoch,val_lowest, s=150, c= 'blue', label=sc_label)
            axes[0].set_title('Training and Validation Loss')
            axes[0].set_xlabel('Epochs')
            axes[0].set_ylabel('Loss')
            axes[0].legend()
            axes[1].plot (Epochs,tacc,'r',label= 'Training Accuracy')
            axes[1].plot (Epochs,vacc,'g',label= 'Validation Accuracy')
            axes[1].scatter(index_acc+1 +start_epoch,acc_highest, s=150, c= 'blue', label=vc_label)
            axes[1].set_title('Training and Validation Accuracy')
            axes[1].set_xlabel('Epochs')
            axes[1].set_ylabel('Accuracy')
            axes[1].legend()
            plt.tight_layout
            plot_filename = 'static/plot.png'
            plt.savefig(plot_filename)
            #plt.style.use('fivethirtyeight')

        def print_info( test_gen, preds, print_code, save_dir, subject ):
            class_dict=test_gen.class_indices
            labels= test_gen.labels
            file_names= test_gen.filenames 
            error_list=[]
            true_class=[]
            pred_class=[]
            prob_list=[]
            new_dict={}
            error_indices=[]
            y_pred=[]
            for key,value in class_dict.items():
                new_dict[value]=key             # dictionary {integer of class number: string of class name}
            # store new_dict as a text fine in the save_dir
            classes=list(new_dict.values())     # list of string of class names
            dict_as_text=str(new_dict)
            dict_name= subject + '-' +str(len(classes)) +'.txt'  
            dict_path=os.path.join(save_dir,dict_name)    
            with open(dict_path, 'w') as x_file:
                x_file.write(dict_as_text)    
            errors=0      
            for i, p in enumerate(preds):
                pred_index=np.argmax(p)        
                true_index=labels[i]  # labels are integer values
                if pred_index != true_index: # a misclassification has occurred
                    error_list.append(file_names[i])
                    true_class.append(new_dict[true_index])
                    pred_class.append(new_dict[pred_index])
                    prob_list.append(p[pred_index])
                    error_indices.append(true_index)            
                    errors=errors + 1
                y_pred.append(pred_index)    
            if print_code !=0:
                if errors>0:
                    if print_code>errors:
                        r=errors
                    else:
                        r=print_code           
                    msg='{0:^28s}{1:^28s}{2:^28s}{3:^16s}'.format('Filename', 'Predicted Class' , 'True Class', 'Probability')
                    print_in_color(msg, (0,255,0),(55,65,80))
                    for i in range(r):                
                        split1=os.path.split(error_list[i])                
                        split2=os.path.split(split1[0])                
                        fname=split2[1] + '/' + split1[1]
                        msg='{0:^28s}{1:^28s}{2:^28s}{3:4s}{4:^6.4f}'.format(fname, pred_class[i],true_class[i], ' ', prob_list[i])
                        print_in_color(msg, (255,255,255), (55,65,60))
                        #print(error_list[i]  , pred_class[i], true_class[i], prob_list[i])               
                else:
                    msg='With accuracy of 100 % there are no errors to print'
                    print_in_color(msg, (0,255,0),(55,65,80))
            if errors>0:
                plot_bar=[]
                plot_class=[]
                for  key, value in new_dict.items():        
                    count=error_indices.count(key) 
                    if count!=0:
                        plot_bar.append(count) # list containg how many times a class c had an error
                        plot_class.append(value)   # stores the class 
                fig=plt.figure()
                fig.set_figheight(len(plot_class)/3)
                fig.set_figwidth(10)
                plt.style.use('fivethirtyeight')
                for i in range(0, len(plot_class)):
                    c=plot_class[i]
                    x=plot_bar[i]
                    plt.barh(c, x, )
                    plt.title( ' Errors by Class on Test Set')
            y_true= np.array(labels)        
            y_pred=np.array(y_pred)
            if len(classes)<= 30:
                # create a confusion matrix 
                cm = confusion_matrix(y_true, y_pred )        
                length=len(classes)
                if length<8:
                    fig_width=5
                    fig_height=5
                else:
                    fig_width= int(length * .5)
                    fig_height= int(length * .5)
                plt.figure(figsize=(fig_width, fig_height))
                sns.heatmap(cm, annot=True, vmin=0, fmt='g', cmap='Blues', cbar=False)       
                plt.xticks(np.arange(length)+.5, classes, rotation= 90)
                plt.yticks(np.arange(length)+.5, classes, rotation=0)
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                plt.title("Confusion Matrix")
                cm = 'static/cm.png'
                plt.savefig(cm)
            clr = classification_report(y_true, y_pred, target_names=classes)
            return clr
      
        # save model
        tr_plot(history,0)
        save_dir=r'pp/'
        subject='pisang'
        acc=model.evaluate( test_gen, batch_size=test_batch_size, verbose=1, steps=test_steps, return_dict=False)[1]*100
        msg=f'accuracy on the test set is {acc:5.2f} %'
        print_in_color(msg, (0,255,0),(55,65,80))
        save_id=str (model_name +  '-' + subject +'-'+ str(acc)[:str(acc).rfind('.')+3] + '.h5')
        save_loc=os.path.join(save_dir, save_id)
        model.save(save_loc)

        print_code=0
        preds=model.predict(test_gen) 
        clr = print_info( test_gen, preds, print_code, save_dir, subject )

        plot_filename = 'static/plot.png'
        cm = 'static/cm.png'
        # encoded_image = extract()
        augmented = 'static/aug.png'
        ori = 'static/ori.png'
        
    return render_template('model.html', df = df, lengths = lengths, augmented = augmented, ori = ori, plot_filename = plot_filename, cm = cm, clr = clr)

app.run(debug=True)