from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import Sequential
from keras.layers import Dense
from django.shortcuts import render, redirect
import os
from django.http import JsonResponse
def CollegePredictorView(request):
    if request.method == 'GET':
        return render(request, 'CollegePredictor.html', )
    if request.method == "POST":
        gre = request.POST['gre']
        toefl = request.POST['toefl']
        u_rating = request.POST['u_rating']
        sop = request.POST['sop']
        lor = request.POST['lor']
        cgpa = request.POST['cgpa']
        r_exp = request.POST['r_exp']
        df = pd.read_csv(os.path.join('CollegePredictor/Admission_Predict.csv'))
        df.drop(columns=['Serial No.'], inplace=True)
        X = df.iloc[:, 0:-1]
        y = df.iloc[:, -1]
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, y, test_size=0.2, random_state=1)
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        global r2_result
        global model
        model = Sequential()
        model.add(Dense(7, activation='relu', input_dim=7))
        model.add(Dense(7, activation='relu'))
        model.add(Dense(7, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer='Adam')
        history = model.fit(X_train_scaled, Y_train, epochs=100, validation_split=0.2)
        y_pred = model.predict(X_test_scaled)
        r2_result = r2_score(Y_test, y_pred)
        input_data = (float(gre), float(toefl), float(u_rating), float(sop), float(lor), float(cgpa), float(r_exp))
        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        global prediction
        prediction = model.predict(input_data_reshaped)
        prediction = str(prediction)
        prediction = prediction[2:len(prediction)-2]
        index = prediction.find('.')
        prediction = prediction[:index]
        r = str(r2_result)[2:]
        var = int(r)/100000000000000
        r = str(var)
        res = {'prediction':prediction, 'r2_result': r, 'gre':gre,'toefl':toefl,'u_rating':u_rating,'sop':sop,'lor':lor,'cgpa':cgpa,'r_exp':r_exp}
        return render(request, 'predictorresult.html',res)



def chartView(request):
    finalrep = {'answer': 0, 'nochance':0}
    finalrep['answer'] = prediction
    intpre = 100.00 - float(prediction)
    finalrep['nochance'] = intpre
    return JsonResponse({'prediction':finalrep}, safe=False)