# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 08:56:21 2021

@author: saake
"""

import numpy as np
from flask import Flask, request, jsonify, render_template, redirect, url_for, abort, send_from_directory
import pickle
import os

app = Flask(__name__, template_folder='templates')  
app.config['UPLOAD_PATH'] = 'C:/Users/mansu/Desktop/AIS/uploads'
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_file():
    # uploaded_file = request.files['file']
    # uploaded_file.save(os.path.join('./', uploaded_file.filename))
    # if uploaded_file.filename != '':
    return redirect('test')

@app.route('/upload/<filename>')
def upload(filename):
    return send_from_directory(app.config['UPLOAD_PATH'], filename)

@app.route('/test')
def test():
    return render_template('test.html')

@app.route('/predict',methods=['POST'])
def predict():
    season = {'Winter': 1,'Spring': 2, 'Summer': 3, 'Fall': 4}
    int_features = [x for x in request.form.values()]
    int_features[-1] = season[int_features[-1]]
     
    final_features = int_features[1:]
    final_features = np.array([int(x) for x in final_features])
    
    print(final_features)
    
    
    final_features= final_features.reshape(1,-1)
    print(model)
    output = model[int_features[0]].predict(final_features) 
    
    #output = 76
    return render_template('test.html', prediction_text='You should buy {} pounds'.format(output))

