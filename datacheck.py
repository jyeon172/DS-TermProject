import pandas as pd
import numpy as np


def decode_symp(enc):
    raw = []

    for i in range(len(enc)):
        s = ""
        if (enc.iloc[i, 0] == 1):
            s = s + ',Fever'
        if (enc.iloc[i, 1] == 1):
            s = s + ',Tiredness'
        if (enc.iloc[i, 2] == 1):
            s = s + ',Dry-Cough'
        if (enc.iloc[i, 3] == 1):
            s = s + ',Difficulty-in-Breathing'
        if (enc.iloc[i, 4] == 1):
            s = s + ',Sore-Throat'
        if (enc.iloc[i, 5] == 1):
            s = 'None-Symptom'

        raw.append(s)

    return raw


def decode_expSymp(enc):
    raw = []

    for i in range(len(enc)):
        s = ""
        if (enc.iloc[i, 0] == 1):
            s = s + ',Pains'
        if (enc.iloc[i, 1] == 1):
            s = s + ',Nasal-Congestion'
        if (enc.iloc[i, 2] == 1):
            s = s + ',Runny-Nose'
        if (enc.iloc[i, 3] == 1):
            s = s + ',Diarrhea'
        if (enc.iloc[i, 4] == 1):
            s = 'None_Experiencing'

        raw.append(s)

    return raw


def decode_age(enc):
    raw = []

    for i in range(len(enc)):
        if (enc.iloc[i, 0] == 1):
            age = 1
        elif (enc.iloc[i, 1] == 1):
            age = 11
        elif (enc.iloc[i, 2] == 1):
            age = 21
        elif (enc.iloc[i, 3] == 1):
            age = 26
        elif (enc.iloc[i, 4] == 1):
            age = 61

        raw.append(age)

    return raw


def decode_gender(enc):
    raw = []

    for i in range(len(enc)):
        if (enc.iloc[i, 0] == 1):
            g = 'Female'
        elif (enc.iloc[i, 1] == 1):
            g = 'Male'
        elif (enc.iloc[i, 2] == 1):
            g = 'Transgender'

        raw.append(g)
    return raw


def decode_severity(enc):
    raw = []

    for i in range(len(enc)):
        if (enc.iloc[i, 0] == 1):
            s = 'Mild'
        elif (enc.iloc[i, 1] == 1):
            s = 'Moderate'
        elif (enc.iloc[i, 2] == 1):
            s = 'None'
        elif (enc.iloc[i, 3] == 1):
            s = 'Severe'

        raw.append(s)
    return raw


def decode_contact(enc):
    raw = []

    for i in range(len(enc)):
        if (enc.iloc[i, 0] == 1):
            c = 'Dont-Know'
        elif (enc.iloc[i, 1] == 1):
            c = 'No'
        elif (enc.iloc[i, 2] == 1):
            c = 'Yes'

        raw.append(c)
    return raw


data = pd.read_csv('Cleaned-Data.csv')

# symptom age gender severity contact country

symp = data.iloc[:, [0, 1, 2, 3, 4, 5]]
expSymp = data.iloc[:, [6, 7, 8, 9, 10]]
age = data.iloc[:, [11, 12, 13, 14, 15]]
gender = data.iloc[:, [16, 17, 18]]
severity = data.iloc[:, [19, 20, 21, 22]]
contact = data.iloc[:, [23, 24, 25]]

# make raw data
raw_symp = decode_symp(symp)
raw_expSymp = decode_expSymp(expSymp)
raw_age = decode_age(age)
raw_gender = decode_gender(gender)
raw_severity = decode_severity(severity)
raw_contact = decode_contact(contact)

# input data
data['Symptoms'] = raw_symp
data['Experiencing-Symptoms'] = raw_expSymp
data['Age'] = raw_age
data['Gender'] = raw_gender
data['Severity'] = raw_severity
data['Contact'] = raw_contact

# drop onehot data
drop_list = ['Fever', 'Tiredness', 'Dry-Cough', 'Difficulty-in-Breathing', 'Sore-Throat', 'None_Symptom',
             'Pains', 'Nasal-Congestion', 'Runny-Nose', 'Diarrhea', 'None_Experiencing',
             'Age_0-9', 'Age_10-19', 'Age_20-24', 'Age_25-59', 'Age_60+',
             'Gender_Female', 'Gender_Male', 'Gender_Transgender',
             'Severity_Mild', 'Severity_Moderate', 'Severity_None', 'Severity_Severe',
             'Contact_Dont-Know', 'Contact_No', 'Contact_Yes']

data = data.drop(drop_list, axis=1, inplace=True)

print(data)
print('==dtypes=====')
print(data.dtypes)
print('==describe=====')
print(data.describe())
print('==info=====')
print(data.info())
