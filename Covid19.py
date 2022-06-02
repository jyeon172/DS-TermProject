"""
Data Science
Term Project
Helps self-diagnosis of COVID-19 through body symptoms
202035509 KIM YEEUN
202035518 NOH HYUNGJU
202035521 PARK JEONGYEON
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import warnings

warnings.filterwarnings('ignore')

pd.options.display.max_columns = None
pd.options.display.width = None

'''
Data Print Functions
'''

# Functions for output of initial data
def do_printInfo(dt): 
    print(dt)
    print()
    
    # To determine the column name, missing values by column, and data type by column that exist in the initial data
    print(dt.info())
    print()
    print(dt.isna().sum())
    print()
    
    # It is used to summarize major statistics according to numeric columns of initial data.
    print(dt.describe())
    print()
    
    # Items by Features
    print(dt['Country'].unique())
    print()
    print(dt['Symptoms'].unique())
    print()
    print(dt['Experiencig_Symptoms'].unique())
    print()
    print(dt['Age'].unique())
    print()
    print(dt['Gender'].unique())
    print()
    print(dt['Severity'].unique())
    print()
    print(dt['Contact'].unique())
    print()


'''
Data PreProcessing Functions
'''


def change_wrong_name(dt):  # Change 'None-Sympton' in original data to 'None Symptom'
    for i in range(len(dt)):
        s = dt['Symptoms'][i]
        if s == 'None-Sympton':
            dt['Symptoms'][i] = 'None-Symptom'

    return dt


def chk_wrong_age_and_treat(dt):  # treating wrong data
    for i in range(len(dt)):
        a = dt['Age'][i]
        if a >= 150 or a < 0:  # age is wrong data
            dt['Age'][i] = np.NAN  # make it NaN

    # fill NaN data
    dt.fillna(axis=0, method='ffill', limit=2, inplace=True)
    dt.fillna(axis=0, method='bfill', limit=2, inplace=True)

    return dt


def get_counts(dt_origin):  # getting counts of Symptoms and ExpSympts in each row
    if 'Country' in dt_origin.columns.tolist(): # if 'Country' feature is not dropped from data, drop it
        dt_origin = dt_origin.drop(labels='Country', axis=1)

    symp_cnt = []
    expSymp_cnt = []

    for i in range(len(data)):  # in every row
        symp = len(dt_origin['Symptoms'][i].split(','))  # count the number of symptoms
        if 'None-Symptom' in dt_origin['Symptoms'][i].split(','): # because 'None-Symptom' means the number of symtom is 0, so minus 1 from counted data (1)
            symp = symp - 1
        symp_cnt.append(symp)

        expSymp = len(dt_origin['Experiencig_Symptoms'][i].split(','))  # count the number of ExpSympts
        if 'None_Experiencing' in dt_origin['Experiencig_Symptoms'][i].split(','): # Same with 'None-Symptom'
            expSymp = expSymp - 1
        expSymp_cnt.append(expSymp)

    dt_cnt = pd.DataFrame(symp_cnt, columns=['Count_Symptoms'])  # make new dataframe
    dt_cnt = dt_cnt.assign(Count_Experiencing_Symptoms=expSymp_cnt)  # add ExpSymp data

    dt_joined = dt_origin.join(dt_cnt)  # join new dataframe with data

    return dt_joined


def get_condition(dt_origin):  # get severity level(=condition): none=0, mild=1, moderate=2, severe=3
    df = dt_origin.copy()
    severity_columns = df.filter(like='Severity_').columns  # get all Severity columns
    df['Severity_None'].replace({1: '0', 0: '-1'}, inplace=True)
    df['Severity_Mild'].replace({1: '1', 0: '-1'}, inplace=True)
    df['Severity_Moderate'].replace({1: '2', 0: '-1'}, inplace=True)
    df['Severity_Severe'].replace({1: '3', 0: '-1'}, inplace=True)
    df['Condition'] = df[severity_columns].values.tolist()  # data type = list

    def removing(list1):
        list1 = set(list1)  # make list1(String)
        list1.discard('-1')  # delete '-1' in list1
        a = ''.join(list1)
        return a

    df['Condition'] = df['Condition'].apply(removing)  # to change data type list => string
    df['Condition'] = df['Condition'].apply(pd.to_numeric)  # string to float

    # df = df.join(df['Condition']) #join Condition data
    df.drop(severity_columns, axis=1, inplace=True)  # drop all Severity columns except 'Condition'

    return df

# one-hot encoding code resource: https://steadiness-193.tistory.com/99
def one_hot(dt, data_idx, prefix): # dt: entire DataFrame to encode, data_idx: index of data to encode, prefix: feature prefix string to add after encoding
    all_ele = []
    data_col = dt.iloc[:, data_idx]  # get data column using index

    for i in data_col:  # get all elements of data
        all_ele.extend(i.split(','))

    ele = pd.unique(all_ele)  # make data elemtns unique
    zero_matrix = np.zeros((len(data_col), len(ele)))  # make zero table for one-hot
    dumnie = pd.DataFrame(zero_matrix, columns=ele)

    for i, elem in enumerate(data_col):  # update one-hot table 1 for each element
        index = dumnie.columns.get_indexer(elem.split(',')) # get index of dumnie data
        dumnie.iloc[i, index] = 1

    dt = dt.iloc[:, data_idx:]  # drop data before encoding
    data_joined = dt.join(dumnie.add_prefix(prefix))  # join one-hot encoding dataframe

    print('One-hot Encoding Success')
    return data_joined


def one_hot_age(dt):  # one-hot encoding for 'Age' Column
    data_col = dt['Age']
    cols = ['0-9', '10-19', '20-24', '25-59', '60+'] # groups of age

    zero_matrix = np.zeros((len(data_col), len(cols)))
    dumnie = pd.DataFrame(zero_matrix, columns=cols) # make empty dataframe

    for i in range(len(data_col)): # if 'Age' data is in particular age group, make the value of dataframe[age row, group col] = 1
        if data_col[i] < 10: # group 0-9
            dumnie.iloc[i, 0] = 1
        elif data_col[i] < 20: # group 10-19
            dumnie.iloc[i, 1] = 1
        elif data_col[i] < 25: # group 20-24
            dumnie.iloc[i, 2] = 1
        elif data_col[i] < 60: # group 24-59
            dumnie.iloc[i, 3] = 1
        elif data_col[i] >= 60: # group 60+
            dumnie.iloc[i, 4] = 1

    dt = dt.drop(labels='Age', axis=1) # drop Age (before encoding data)
    data_joined = dt.join(dumnie.add_prefix('Age_')) # add encoding dataframe

    print('One-hot Encoding Success')
    return data_joined


def do_oneHot(dt):  # do one-hot encoding in entire dataframe
    dt = one_hot(dt, dt.columns.get_loc('Symptoms'), 'Symptoms_')
    dt = one_hot(dt, dt.columns.get_loc('Experiencig_Symptoms'), 'ExpSympt_')
    dt = one_hot_age(dt)  # Column 'Age' need to grouping, so use different one-hot-encoding function
    dt = one_hot(dt, dt.columns.get_loc('Gender'), 'Gender_')
    dt = one_hot(dt, dt.columns.get_loc('Severity'), 'Severity_')
    dt = one_hot(dt, dt.columns.get_loc('Contact'), 'Contact_')

    # in one_hot def, index of Contact is 0 so that data.iloc[:, data_idx:] is not work
    dt_onehot = dt.drop(labels='Contact', axis=1)

    dt_onehot.to_csv('./data/after_oneHot.csv', index=False)  # save the result in the csv file

    return dt_onehot


def get_Conditions(dt):  # merge 'Severity_' columns in String
    print('make heatmap start')
    df = dt.copy()
    severity_columns = df.filter(like='Severity_').columns
    
    df['Severity_None'].replace({1: 'None', 0: 'No'}, inplace=True)
    df['Severity_Mild'].replace({1: 'Mild', 0: 'No'}, inplace=True)
    df['Severity_Moderate'].replace({1: 'Moderate', 0: 'No'}, inplace=True)
    df['Severity_Severe'].replace({1: 'Severe', 0: 'No'}, inplace=True)
    df['Condition'] = df[severity_columns].values.tolist()  # df['Condition'] is a list ex. ['No', 'Mild', 'No', 'No']

    def removing(list1):  # change df['Condition'] to string
        list1 = set(list1)
        list1.discard("No")
        a = ''.join(list1)
        return a

    df['Condition'] = df['Condition'].apply(removing)
    df.drop(severity_columns, axis=1, inplace=True)

    sns.countplot(df['Condition'])
    plt.show()

    return df


def get_score(dt):  # get Symptoms score using Symptoms_sum and ExpSymps_sum
    df = dt.copy()
    idx_syp_start = dt.columns.get_loc('Symptoms_Fever') # the index of first feature of Symptoms_
    idx_syp_end = dt.columns.get_loc('Symptoms_None-Symptom') # the index of last feature of Symptoms_
    idx_expSyp_start = dt.columns.get_loc('ExpSympt_Pains') # the index of first feature of ExpSympt_
    idx_expSyp_end = dt.columns.get_loc('ExpSympt_None_Experiencing') # the index of last feature of ExpSympt_
    
    # the score is number of symptoms and experiencing symptoms
    df['Symptoms_Score'] = df.iloc[:, idx_syp_start:idx_syp_end].sum(axis=1) + df.iloc[:, idx_expSyp_start:idx_expSyp_end].sum(axis=1)

    return df

# code resource: https://www.kaggle.com/code/harshaggarwal7/covid-19-symptom-analysis?scriptVersionId=40251034&cellId=36
def get_htmp_after_encod(dt): # dt: dataframe to make heatmap
    df = dt.copy()
    df = get_score(df)
    df = get_condition(df)

    from pylab import rcParams
    rcParams['figure.figsize'] = 13, 18
    corrmat = df.corr() # make correlation matrix
    cols = corrmat.index
    cm = np.corrcoef(df[cols].values.T) # get correlation coefficient of correlation matrix
    
    sns.set(font_scale=1.25) # set heatmap plot
    sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',
                annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values) # make heatmap
    plt.title('Heatmap of All data')
    plt.show()

    return corrmat  # return correlation matrix data


def do_PCA_partial(data_origin):  # PCA just for partial (drop severity:one-hot and Condition: string)
    dt = data_origin.copy()
    severity_columns = dt.filter(like='Severity_').columns
    dt.drop(severity_columns, axis=1, inplace=True) # drop severity because they are one-hot encoded columns
    dt.drop('Condition', axis=1, inplace=True) # drop condition because it is string

    scaler = preprocessing.StandardScaler()
    train_df_scaled = scaler.fit_transform(dt) # data scaling

    pca = PCA(n_components=2)
    df = pca.fit_transform(train_df_scaled)
    pca_df = pd.DataFrame(df, columns=['P1', 'P2'])

    res_ratio = pca.explained_variance_ratio_
    print('Variance ratio after PCA is {}'.format(res_ratio))

    return pca_df


def do_PCA_all(data_origin):  # PCA for all data (with severity, drop Condition:string)

    df = data_origin.copy()
    df_x = df.drop('Condition', axis=1)  # to apply PCA
    df_y = df['Condition']  # the results

    x = df_x.values
    x = preprocessing.StandardScaler().fit_transform(x)  # scaling data

    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(x)  # do PCA
    df_pca = pd.DataFrame(data=data_pca, columns=['principal component 1',
                                                  'principal component 2'])  # restore PCA result

    print('===PCA Result=====')  # print PCA result
    print(df_pca)

    df_final = pd.concat([df_pca, df_y], axis=1)  # join PCA result and 'Condition'
    print('===PCA Result with Condition=====')
    print(df_final)

    res_ratio = pca.explained_variance_ratio_
    print('Variance ratio after PCA is {}'.format(res_ratio))

    return df_pca


def do_preprocessing(dt_oh):
    data_cond = get_Conditions(dt_oh)  # data with Condition (type: string)
    cor = get_htmp_after_encod(dt_oh)  # getting correlation matrix
    data_co = get_condition(dt_oh)  # data with Condition (type: int)

    print('===Result of Data with Condition(String)=====')
    print(data_cond)
    print('===Result of Data with Condition(Integer)=====')
    print(data_co)
    print('===Result of Correlation=====')
    print(cor)

    symp_pca_1 = do_PCA_partial(dt_oh[dt_oh.filter(like='Symptoms_').columns].join(data_co['Condition']))
    expSymp_pca_1 = do_PCA_partial(dt_oh[dt_oh.filter(like='ExpSympt_').columns].join(data_co['Condition']))

    symp_pca_2 = do_PCA_all(dt_oh[dt_oh.filter(like='Symptoms_').columns].join(data_co['Condition']))
    expSymp_pca_2 = do_PCA_all(dt_oh[dt_oh.filter(like='ExpSympt_').columns].join(data_co['Condition']))

    print('===Result of PCA of Symptoms without Severity=====')
    print(symp_pca_1)
    print('===Result of PCA of Experiencing Symptoms without Severity=====')
    print(expSymp_pca_1)

    print('===Result of PCA of Symptoms with Severity=====')
    print(symp_pca_2)
    print('===Result of PCA of Experiencing Symptoms with Severity=====')
    print(expSymp_pca_2)


'''
Visualization Functions
'''

# Function to count the number of symptoms
def get_symptom_count(the_list):
    return sum(the_list.values)

# Functions for different visualizations
def do_visualization():
    # countplot
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    sns.countplot(data_oh['Severity_None'], ax=ax[0])
    sns.countplot(data_oh['Severity_Mild'], ax=ax[1])
    sns.countplot(data_oh['Severity_Moderate'], ax=ax[2])
    sns.countplot(data_oh['Severity_Severe'], ax=ax[3])

    plt.suptitle('Number of patients by severity (0.0 = Not Applicable, 1.0 = Applicable)', fontsize=20)
    fig.tight_layout()
    plt.show()

    # barplot
    # code resource : https://www.kaggle.com/code/sanjanabhute03/uslclustering-project
    indicators = ['Symptoms_Fever', 'Symptoms_Tiredness', 'Symptoms_Dry-Cough', 'Symptoms_Difficulty-in-Breathing',
                  'Symptoms_Sore-Throat', 'ExpSympt_Pains', 'ExpSympt_Nasal-Congestion',
                  'ExpSympt_Runny-Nose', 'ExpSympt_Diarrhea', 'Age_0-9', 'Age_10-19', 'Age_20-24', 'Age_25-59',
                  'Age_60+',
                  'Gender_Male', 'Gender_Female', 'Gender_Transgender']
    target_columns = ['Severity_None']

    severity = ['Severity_Mild', 'Severity_Moderate', 'Severity_Severe', 'Severity_None']

    risk = data_oh[severity]
    features = data_oh[indicators]
    targets = data_oh[target_columns]

    plt.figure(figsize=(10, 10))
    temp = []

    for i in indicators:
        temp.append(sum(features[i].values))
    temp_df = pd.DataFrame({"Indicator": indicators, "Occurence_Count": temp})

    sns.barplot(data=temp_df, x="Occurence_Count", y="Indicator")
    plt.title("Symptom, age, gender distribution", fontsize=25)
    plt.show()

    # piechart
    plt.figure(figsize=(11, 11))
    plt.pie(data=temp_df, x="Occurence_Count", labels=temp_df["Indicator"])
    plt.show()

    # countplot - by severity

    for i in range(len(severity)):
        ind = indicators.copy()
        sev = severity[i]
        ind.append(sev)

        features['Total_Symptom'] = features[indicators].apply(get_symptom_count, axis=1)
        feats = data_oh[ind]
        feats['Total_Symptom'] = feats[indicators].apply(get_symptom_count, axis=1)

        plt.figure(figsize=(10, 10))
        ax1 = sns.countplot(data=feats, x='Total_Symptom', hue=sev)
        plt.xlabel("Total symptom occurence on someone")
        for p in ax1.patches:
            height = p.get_height()
            ax1.text(p.get_x() + p.get_width() / 2., height + 5, height, ha='center', size=9)

        plt.show()


'''
Analysis & Evaluation Functions
'''


def do_analysis(df, indicators, target_columns):
    # Set features and targets
    features = df[indicators]
    targets = df[target_columns]

    features['Total_Symptom'] = features[indicators].apply(get_symptom_count, axis=1)

    # 1: Random Forest Classifier, 2: Logistic Regression, 3: Decision Tree Classifier
    algo = [1, 2, 3]

    # Analysis for each Severity with 3 kinds of algorithm
    # Severity - Severity_None, Severity_Mild, Severity_Moderate, Severity_Severe
    # Analysis - Random Forest Classifier, Logistic Regression, Decision Tree Classifier
    for i in range(len(target_columns)):
        for j in range(len(algo)):
            print("====== For {} ======".format(target_columns[i]))
            target(features, targets, target_columns[i], algo[j])
            print()


def target(fs, ts, t, algo):
    d = fs
    d[t] = ts[t].values  # set current target feature

    # print("== print data ====")
    # print(d)
    # print()

    x = d.drop([t, 'Total_Symptom'], axis=1)
    x = PCA(n_components=3).fit_transform(x)
    y = d[t]

    # print("== print x ====")
    # print(x)
    # print("== print y ====")
    # print(y)

    do_split(x, y, algo)


def do_split(x, y, algo):
    # divide into test and train dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=.3)

    k_fold = KFold(n_splits=10, shuffle=True, random_state=0)  # K-Fold (for k=10)

    if algo == 1:  # Random Forest Classifier
        do_RFC(x_train, x_test, y_train, y_test, k_fold)
    elif algo == 2:  # Logistic Regression
        do_LR(x_train, x_test, y_train, y_test, k_fold)
    elif algo == 3:  # Decision Tree Classifier
        do_DTC(x_train, x_test, y_train, y_test, k_fold)


def do_RFC(x_train, x_test, y_train, y_test, k_fold):
    # Random Forest Classifier
    print("<< Random Forest Classifier >>")
    rfc = RandomForestClassifier()
    rfc.fit(x_train, y_train)
    print("> Random Forest Classifier score: {}".format(rfc.score(x_test, y_test)))  # Print result

    # Print confusion matrix for Random Forest Classifier model
    print("> Confusion Matrix")
    do_cm(rfc, x_test, y_test)

    # Find the best estimator using Grid Search
    params = {
        "max_depth": [15, 20, 25],
        "n_estimators": [27, 30, 33],
        "criterion": ["gini", "entropy"],
    }
    rf_reg = GridSearchCV(rfc, params, cv=10, n_jobs=10)
    rf_reg.fit(x_train, y_train)
    print("> The best estimator using GridSearch: {}".format(rf_reg.best_estimator_))

    # Set model with best estimator
    rfc_tune = rf_reg.best_estimator_
    rfc_tune.fit(x_train, y_train)
    print("> Random Forest Classifier tuned score: {}".format(rfc_tune.score(x_test, y_test)))  # Print result

    # Print confusion matrix for tuned Random Forest Classifier model
    print("> Confusion Matrix using tuned")
    do_cm(rfc_tune, x_test, y_test)

    # Do evaluation with k-fold (k=10)
    do_evaluation(rfc_tune, x_test, y_test, k_fold)


def do_LR(x_train, x_test, y_train, y_test, k_fold):
    # Logistic Regression
    print("<< Logistic Regression >>")
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    print("> Logistic Regression score: {}".format(lr.score(x_test, y_test)))  # Print result

    # Print confusion matrix for Logistic Regression
    print("> Confusion Matrix")
    do_cm(lr, x_test, y_test)

    # Find the best estimator using Grid Search
    params = {
        "penalty": ['l1', 'l2', 'elasticnet', 'none'],
        "solver": ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    }
    lr_reg = GridSearchCV(lr, params, cv=10, n_jobs=10)
    lr_reg.fit(x_train, y_train)
    print("> The best estimator using GridSearch: {}".format(lr_reg.best_estimator_))

    # Set model with best estimator
    lr_tune = lr_reg.best_estimator_
    lr_tune.fit(x_train, y_train)
    print("> Logistic Regression tuned score: {}".format(lr_tune.score(x_test, y_test)))  # Print result

    # Print confusion matrix for tuned Logistic Regression
    print("> Confusion Matrix using tuned")
    do_cm(lr_tune, x_test, y_test)

    # Do evaluation with k-fold (k=10)
    do_evaluation(lr_tune, x_test, y_test, k_fold)


def do_DTC(x_train, x_test, y_train, y_test, k_fold):
    # Decision Tree Classifier
    print("<< Decision Tree Classifier >>")
    dtc = DecisionTreeClassifier()
    dtc.fit(x_train, y_train)
    print("> Decision Tree Classifier score: {}".format(dtc.score(x_test, y_test)))  # Print result

    # Print confusion matrix for Decision Tree Classifier
    print("> Confusion Matrix")
    do_cm(dtc, x_test, y_test)

    # Find the best estimator using Grid Search
    params = {
        "criterion": ["gini", "entropy"],
        "max_depth": [15, 20, 25],
    }
    dtc_reg = GridSearchCV(dtc, params, cv=10, n_jobs=10)
    dtc_reg.fit(x_train, y_train)
    print("> The best estimator using GridSearch: {}".format(dtc_reg.best_estimator_))

    # Set model with best estimator
    dtc_tune = dtc_reg.best_estimator_
    dtc_tune.fit(x_train, y_train)
    print("> Decision Tree Classifier tuned score: {}".format(dtc_tune.score(x_test, y_test)))  # Print result

    # Print confusion matrix for tuned Decision Tree Classifier
    print("> Confusion Matrix using tuned")
    do_cm(dtc_tune, x_test, y_test)

    # Do evaluation with k-fold (k=10)
    do_evaluation(dtc_tune, x_test, y_test, k_fold)


def do_cm(model, x_test, y_test):
    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)  # Set confusion matrix
    # sns.heatmap(cm, annot=True)
    # plt.show()
    print(cm)  # Print result


def do_evaluation(tune, x_test, y_test, kf):
    # Do evaluation with K-Fold (k=10)
    scores = cross_val_score(tune, x_test, y_test, cv=kf, n_jobs=1, scoring="accuracy")
    # Print result
    i = 0
    print("< KFold Accuracy >")
    for score in scores:
        i = i + 1
        print("> When {}, Accuracy: {}".format(i, score))
    print("> Cross validation mean score: {}".format(scores.mean()))  # Print mean score


'''
Data Information
'''

# read data
data = pd.read_csv('./data/Raw_data_final.csv')

# understanding basic data information - technical statistics
do_printInfo(data)

'''
Data PreProcessing
'''

data = change_wrong_name(data)  # change wrong file name 'None-Sympton' to 'None-Symptom'
data = chk_wrong_age_and_treat(data)  # treat wrong data
data.dropna(how='any', inplace=True)  # drop rows with dirty data (except Age error)
data.drop('Country', axis=1, inplace=True)  # drop Column 'Country'
print(data)
print()

data1 = get_counts(data)
print(data1['Count_Symptoms'])
print()

do_oneHot(data)  # one-hot encoding

data_oh = pd.read_csv('./data/after_oneHot.csv')
do_preprocessing(data_oh)

'''
Visualization
'''

do_visualization()

'''
Data Analysis & Evaluation
'''
# Set indicators and target features
indicators_of_data = ['Symptoms_Fever', 'Symptoms_Tiredness', 'Symptoms_Dry-Cough', 'Symptoms_Difficulty-in-Breathing',
                      'Symptoms_Sore-Throat', 'ExpSympt_Pains', 'ExpSympt_Nasal-Congestion',
                      'ExpSympt_Runny-Nose', 'ExpSympt_Diarrhea', 'Age_0-9', 'Age_10-19', 'Age_20-24', 'Age_25-59',
                      'Age_60+', 'Gender_Male', 'Gender_Female', 'Gender_Transgender']
target_cols_of_data = ['Severity_None', 'Severity_Mild', 'Severity_Moderate', 'Severity_Severe']

do_analysis(data_oh, indicators_of_data, target_cols_of_data)

