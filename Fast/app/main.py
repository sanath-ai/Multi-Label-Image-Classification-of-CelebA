import tensorflow as tf
import matplotlib.image as imread
from fastapi import FastAPI, Body, Request, File, UploadFile, Form, Depends, BackgroundTasks 
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from keras.models import load_model
from keras.models import Sequential
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import numpy as np

COL = ['5_o_Clock_Shadow',
 'Arched_Eyebrows',
 'Attractive',
 'Bags_Under_Eyes',
 'Bald',
 'Bangs',
 'Big_Lips',
 'Big_Nose',
 'Black_Hair',
 'Blond_Hair',
 'Blurry',
 'Brown_Hair',
 'Bushy_Eyebrows',
 'Chubby',
 'Double_Chin',
 'Eyeglasses',
 'Goatee',
 'Gray_Hair',
 'Heavy_Makeup',
 'High_Cheekbones',
 'Male',
 'Mouth_Slightly_Open',
 'Mustache',
 'Narrow_Eyes',
 'No_Beard',
 'Oval_Face',
 'Pale_Skin',
 'Pointy_Nose',
 'Receding_Hairline',
 'Rosy_Cheeks',
 'Sideburns',
 'Smiling',
 'Straight_Hair',
 'Wavy_Hair',
 'Wearing_Earrings',
 'Wearing_Hat',
 'Wearing_Lipstick',
 'Wearing_Necklace',
 'Wearing_Necktie',
 'Young']

SIZE = 150


def get_model(model_path):
    model = load_model(model_path)
    return model

def pred(img_dir , model):
    img = imread.imread(img_dir)

    img = image.load_img(img_dir, target_size=(SIZE,SIZE,3))
    img = image.img_to_array(img)
    img = img/255.
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    prd = model.predict(img)      
    #pred_dict = {COL[i]: prd[0][i] for i in range(len(prd[0]))}
    prd = prd[0]
    return prd
    
def best_pred(dic):
    best_prd = []
    ser = list(dic.items())
    for i in range(len(ser)):
        if ser[i][1] > 0.75 :
            best_prd.append((ser[i][0] , ser[i][1]))
    
    return best_prd


app = FastAPI()
templates = Jinja2Templates(directory="app/htmldir/")
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/home/{user_name}", response_class=HTMLResponse)
def write_home(request: Request, user_name: str):
    return templates.TemplateResponse("home.html", {"request": request, "username": user_name})


@app.api_route("/submitform" ,methods = ["GET","POST"], response_class=HTMLResponse)
async def handle_form(request: Request , assignment_file: UploadFile = File(...)):
    
    print(assignment_file.filename)
    content_assignment = await assignment_file.read()
    print(content_assignment)
    with open("app/static/my_file.jpg", "wb") as im:
        im.write(content_assignment)
        file_path = "app/static/my_file.jpg"
        model_path = "app/vgg0.h5"
        model = get_model(model_path)
        prd = pred(file_path , model)
        prd = list(prd)
        print(prd)

        return templates.TemplateResponse("img.html", {"request": request , "col" : COL  , "predictions":prd }  )
