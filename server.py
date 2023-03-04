import os
from flask import Flask, request
import json
import predict
app = Flask(__name__)
@app.route('/',methods=["GET","POST"])
def classify():
    if request.method=="POST":
        f=request.files["hello"] # request a image is just a images that is stored in f
        print(f)
        prediction=predict.pred(f.filename)
        print(prediction)
        if prediction!=None:
            return (prediction)
    return "Image Cannot Be Analyzed"
if __name__ == "__main__":
    app.run(host='0.0.0.0')