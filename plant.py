from PIL import Image
import numpy as np
from skimage import transform
import os
import cv2
import streamlit as st
import tensorflow
import openai
from keras.models import load_model
import pandas as pd
import shutil

def predtext(inp):
    promt = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.\n\nHuman: Hello, who are you?\nAI: I am an AI created by OpenAI. How can I help you today?\nHuman: write about plant disease"+ str(inp)
    openai.api_key = "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"
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=promt,
    temperature=0.9,
    max_tokens=150,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0.6,
    stop=[" Human:", " AI:"]
    )
    a = str(response['choices'][0]['text'])
    return a[5:]

def predfer(inp):
    promt = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.\n\nHuman: Hello, who are you?\nAI: I am an AI created by OpenAI. How can I help you today?\nHuman: name the fertilizers udes for plant disease "+ str(inp)
    openai.api_key = "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"
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=promt,
    temperature=0.9,
    max_tokens=150,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0.6,
    stop=[" Human:", " AI:"]
    )
    a = str(response['choices'][0]['text'])
    return a[5:]

def predexp(inp):
    promt = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.\n\nHuman: Hello, who are you?\nAI: I am an AI created by OpenAI. How can I help you today?\nHuman: name one expert in "+ str(inp)
    openai.api_key = "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"
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=promt,
    temperature=0.9,
    max_tokens=150,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0.6,
    stop=[" Human:", " AI:"]
    )
    a = str(response['choices'][0]['text'])
    return a[5:]

def predictor(sdir, csv_path,  model_path, averaged=True, verbose=True):    
    # read in the csv file
    class_df=pd.read_csv(csv_path)    
    class_count=len(class_df['class'].unique())
    img_height=int(class_df['height'].iloc[0])
    img_width =int(class_df['width'].iloc[0])
    img_size=(img_width, img_height)    
    scale=class_df['scale by'].iloc[0] 
    image_list=[]
    # determine value to scale image pixels by
    try: 
        s=int(scale)
        s2=1
        s1=0
    except:
        split=scale.split('-')
        s1=float(split[1])
        s2=float(split[0].split('*')[1])
    path_list=[]
    paths=os.listdir(sdir)    
    for f in paths:
        path_list.append(os.path.join(sdir,f))
    if verbose:
        print (' Model is being loaded- this will take about 10 seconds')
    model=load_model(model_path)
    image_count=len(path_list) 
    image_list=[]
    file_list=[]
    good_image_count=0
    for i in range (image_count):        
        try:
            img=cv2.imread(path_list[i])
            img=cv2.resize(img, img_size)
            img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)            
            good_image_count +=1
            img=img*s2 - s1             
            image_list.append(img)
            file_name=os.path.split(path_list[i])[1]
            file_list.append(file_name)
        except:
            if verbose:
                print ( path_list[i], ' is an invalid image file')
    if good_image_count==1: # if only a single image need to expand dimensions
        averaged=True
    image_array=np.array(image_list)    
    # make predictions on images, sum the probabilities of each class then find class index with
    # highest probability
    preds=model.predict(image_array)    
    if averaged:
        psum=[]
        for i in range (class_count): # create all 0 values list
            psum.append(0)    
        for p in preds: # iterate over all predictions
            for i in range (class_count):
                psum[i]=psum[i] + p[i]  # sum the probabilities   
        index=np.argmax(psum) # find the class index with the highest probability sum        
        klass=class_df['class'].iloc[index] # get the class name that corresponds to the index
        prob=psum[index]/good_image_count  # get the probability average         
        # to show the correct image run predict again and select first image that has same index
        for img in image_array:  #iterate through the images    
            test_img=np.expand_dims(img, axis=0) # since it is a single image expand dimensions 
            test_index=np.argmax(model.predict(test_img)) # for this image find the class index with highest probability
            if test_index== index: # see if this image has the same index as was selected previously
                if verbose: # show image and print result if verbose=1
                    plt.axis('off')
                    plt.imshow(img) # show the image
                    print (f'predicted species is {klass} with a probability of {prob:6.4f} ')
                break # found an image that represents the predicted class      
        return klass, prob, img, None
    else: # create individual predictions for each image
        pred_class=[]
        prob_list=[]
        for i, p in enumerate(preds):
            index=np.argmax(p) # find the class index with the highest probability sum
            klass=class_df['class'].iloc[index] # get the class name that corresponds to the index
            image_file= file_list[i]
            pred_class.append(klass)
            prob_list.append(p[index])            
        Fseries=pd.Series(file_list, name='image file')
        Lseries=pd.Series(pred_class, name= 'species')
        Pseries=pd.Series(prob_list, name='probability')
        df=pd.concat([Fseries, Lseries, Pseries], axis=1)
        if verbose:
            length= len(df)
            print (df.head(length))
        return None,None,None,df

def pred(img):
    store_path=os.path.join("C:/Users/navad/Downloads/IDP 3-1", 'storage')
    if os.path.isdir(store_path):
        shutil.rmtree(store_path)
    os.mkdir(store_path)
    # input an image of letter A
    img_path = "res.jpg"
    img=cv2.imread(img_path,  cv2.IMREAD_REDUCED_COLOR_2)
    file_name=os.path.split(img_path)[1]
    dst_path=os.path.join(store_path, file_name)
    cv2.imwrite(dst_path, img)
    # check if the directory was created and image stored
    # print (os.listdir(store_path))
    csv_path = "class_dict.csv" # path to class_dict.csv
    model_path= "EfficientNetB3-plant disease-98.91.h5" # path to the trained model
    klass, prob, img, df =predictor(store_path, csv_path,  model_path, averaged=True, verbose=False) # run the classifier
    # msg=f' image is of plant disease {klass}  with a probability of {prob * 100: 6.2f} %'
    return klass , prob

image = Image.open(r'C:\Users\navad\Downloads\IDP 3-1\WhatsApp Image 2023-01-20 at 06.53.25.jpg') 
col1, col2 = st.columns([0.6, 0.4])
with col1:               
    st.markdown(""" <style> .font {
    font-size:35px ; font-family: 'Cooper Black'; color: black;} 
    </style> """, unsafe_allow_html=True)
    st.markdown(""" <style> .font1 {
    font-size:35px ; font-family: 'Times New Roman'; color: black;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">HACKATHON 2023</p>', unsafe_allow_html=True)
with col2:
    st.image(image,  width=380)
st.markdown('<p class="font1">Identify and Solve Disease in Plants/Crops</p>', unsafe_allow_html=True)
st.sidebar.markdown('<p class="font1">Welcome to the project </p>', unsafe_allow_html=True)
with st.sidebar.expander("About the App"):
     st.write("""
        Use this simple app to detection plate disease from the image,provides scientific explantion and remidies to cure disease . 
         \n  
     """)
im = Image.open(r"C:\Users\navad\Downloads\IDP 3-1\Screenshot 2023-01-20 064033.png")
with st.sidebar.expander("About the Dataset"):
     st.write("""
        The Current Project is avaliable with the following Classes of Deseases in Crops/Plants \n
        -Tomato \n 
        -Potato \n 
        -Apple \n 
        -Grape \n 
        -Bell Pepper \n 
        -Peach \n 
        -Strawberry \n
        -Cherry \n  
     """)
st.sidebar.image(im)

uploaded_file = st.file_uploader("", type=['jpg','png','jpeg'])
if uploaded_file is not None:
    st.title("Here is the image you've selected")
    image = Image.open(uploaded_file)
    st.image(image)
    b = image.save("res.jpg")
    a,p = pred("res.jpg")
    if int(p*100) > 80:
        st.subheader("Name of the disease :"+a)
        st.subheader("Accuracy :"+str(p*100))
        st.subheader("Description")
        st.write(predtext(a))
        st.subheader("Solution", anchor=None)
        st.write(predfer(a))
        # st.subheader("Expert")
        # st.write(predexp(a))
        st.subheader("Helpline Number: 155251")
    else:
        st.subheader("Invalid Input")
        st.subheader("Helpline Number: 155251")


    