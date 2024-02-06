from flask import Flask, request, jsonify
import h5py
import numpy as np
import tensorflow as tf
import matplotlib.image as img
import PIL
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.image as img
from keras.models import load_model
import keras.backend
import keras.utils as image
from keras.preprocessing.image import ImageDataGenerator
from keras import models
import urllib.request
from PIL import Image


app = Flask(__name__)
#model = load_model(r'C:\Users\ssalm\Downloads\best_model_101class1.hdf5' ,compile = False)
model = load_model(r'C:\Users\ssalm\Downloads\mobilenet_model.hdf5' ,compile = False)
@app.route('/api/model', methods=['POST'])
def index():
    data= request.get_json()
    print("--->",request);
    image_link = data["image"]
    urllib.request.urlretrieve(
    image_link,
   "g.jpg")
    a = [Image.open("g.jpg")]
    a[0].show()
    images=[r"g.jpg"]
  
    #img = [Image.open("g.jpg")]
    #images = image_link
    #images = [r'C:\Users\ssalm\OneDrive\Desktop\Calories Counter\python\env\spegit.jpg']
    for img in images:
            img = image.load_img(img, target_size=(224, 224))
            img = image.img_to_array(img)                    
            img = np.expand_dims(img, axis=0)         
            img = img / 255.   
            print(img.shape)                                   

            pred = model.predict(img)
            index = np.argmax(pred)
            #foods_sorted= ['Apple pie, 277 calories','Baklava, 334 calories','Bread pudding,306 calories in 1 cup','Burger, 354 calories per burger','Cheese plate, 339.2 per 92.65g','Cheesecake, 257 calories one peice 80g','Chicken curry, 296 calories per 269g','Chicken wings, 42 calories per wings','Chocolate cake, 352 calories per 95g','Club sandwich, 591 calories 1 sandwich 268g','Cup cakes, 131 calories in 1 cupcake (43 g)','Donuts, 245 calories in 1 medium size doughnut','Fish and chips, 710 calories per 228g','French fries, 365 calories per 117g','French toast, 149 calories 1 slices','Fried rice, 228 calories per 140g','Frozen yogurt, 159 calories per 100g','Garlic bread, 206 calories in 1 slices','Greek salad, 111 calories in 240g','Grilled cheese sandwich, 426 calories per 116 g serving','Hot and sour soup, 90 calories 1 cup','Hot dog, 151 calories in 1 hotdog','Ice cream, 137 calories in 1 cup','Macaroni, 310 calories in 249g','Pancakes, 227 calories in 100g',
            #'Pizza, 285 calories in 1 silice','Poutine, 233 calories in 100g','Samosa,91 calories in one samosa','Spaghetti , 113 calories in 100 grams','Spring rolls, 85 calories in one piece','Strawberry shortcake, 346 calories in 100g','Waffles, 218 calories in one round']

            foods_sorted=['Bread, 206 calories in 1 slices: 0',
                            'Cake, 352 calories per 95g: 1',
                            'Chicken wings, 42 calories per wings: 2',
                            'Club sandwich, 591 calories 1 sandwich 268g: 3',
                            'Hamburger, 354 calories per burger: 4',
                            'Ice cream, 137 calories in 1 cup: 5',
                            'Macaroni, 310 calories in 249g: 6',
                            'Pizza, 285 calories in 1 silice: 7',
                            'Samosa, 91 calories in one samosa: 8',
                            'Spaghetti , 113 calories in 100 grams: 9',
                            'Spring rolls, 85 calories in one piece: 10']   
            pred_value = foods_sorted[index]
            return {"models":pred_value}

    return jsonify({'result1' :data})
    
        

#images = [r'C:\Users\ssalm\OneDrive\Desktop\Calories Counter\python\env\spegit.jpg']

if __name__ == "__main__":
      app.run(port=3000,debug=True)
