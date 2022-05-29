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
Data PreProcessing Functions
'''


def change_wrong_name(data):  # Change 'None-Sympton' in original data to 'None Symptom'
    for i in range(len(data)):
        s = data['Symptoms'][i]
        if (s == 'None-Sympton'):
            data['Symptoms'][i] = 'None-Symptom'

    return data


def chk_wrong_age_and_treat(data):  # treating wrong data
    for i in range(len(data)):
        a = data['Age'][i]
        if (a >= 150 or a < 0):  # age is wrong data
            data['Age'][i] = np.NAN  # make it NaN

    # fill NaN data
    data.fillna(axis=0, method='ffill', limit=2, inplace=True)
    data.fillna(axis=0, method='bfill', limit=2, inplace=True)

    return data


def get_counts(data):  # getting counts of Symptoms and ExpSympts in each row
    if ('Country' in data.columns.tolist()):
        data = data.drop(labels='Country', axis=1)

    symp_cnt = []
    expSymp_cnt = []

    for i in range(len(data)):  # in every row
        symp = len(data['Symptoms'][i].split(','))  # count the number of symptoms
        if ('None-Symptom' in data['Symptoms'][i].split(',')):
            symp = symp - 1
        symp_cnt.append(symp)

        expSymp = len(data['Experiencig_Symptoms'][i].split(','))  # count the number of ExpSympts
        if ('None_Experiencing' in data['Experiencig_Symptoms'][i].split(',')):
            expSymp = expSymp - 1
        expSymp_cnt.append(expSymp)

    data_cnt = pd.DataFrame(symp_cnt, columns=['Count_Symptoms'])  # make new dataframe
    data_cnt = data_cnt.assign(Count_Experiencing_Symptoms=expSymp_cnt)  # add ExpSymp data

    data_joined = data.join(data_cnt)  # join new dataframe with data

    return data_joined


def get_condition(data):  # get severity level(=condition): none=0, mild=1, moderate=2, severe=3
    df = data.copy()
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

    df['Condition'] = df['Condition'].apply(removing)  ##to change data type list => string
    df['Condition'] = df['Condition'].apply(pd.to_numeric)  # string to float

    # df = df.join(df['Condition']) #join Condition data
    df.drop(severity_columns, axis=1, inplace=True)  # drop all Severy columns except 'Condition'

    return df


def get_htmp_before_encoding(data):  # make heatmap using count of symptoms and expSympts and Condition
    print('make heatmap start')
    data = get_counts(data)
    print('get counts success')
    data = get_condition(data)
    print('get condition success')

    df = data[['Count_Symptoms', 'Count_Experiencing_Symptoms', 'Condition']]
    # df = preprocessing.StandardScaler().fit_transform(df) #scaling data

    df = df.corr()
    print(df)

    htmp = sns.heatmap(df, vmin=-5, vmax=5, cbar=True)
    plt.title('Heatmap of Count Symptoms, Experencing Symptoms, and Severity Level')
    plt.show()
    print('make heatmap success')


def one_hot(data, data_idx, prefix):  # one-hot encoding
    all_ele = []
    data_col = data.iloc[:, data_idx]  # get data column using index

    for i in data_col:  # get all elements of data
        all_ele.extend(i.split(','))

    ele = pd.unique(all_ele)  # make data elemtns unique
    zero_matrix = np.zeros((len(data_col), len(ele)))  # make zero table for one-hot
    dumnie = pd.DataFrame(zero_matrix, columns=ele)

    for i, elem in enumerate(data_col):  # update one-hot table 1 for each element
        index = dumnie.columns.get_indexer(elem.split(','))
        dumnie.iloc[i, index] = 1

    data = data.iloc[:, data_idx:]  # drop data before encoding
    data_joined = data.join(dumnie.add_prefix(prefix))  # join one-hot encoding dataframe

    print('One-hot Encoding Success')
    return data_joined


def one_hot_age(data):  # one-hot encoding for 'Age' Column
    data_col = data['Age']
    cols = ['0-9', '10-19', '20-24', '25-59', '60+']

    zero_matrix = np.zeros((len(data_col), len(cols)))
    dumnie = pd.DataFrame(zero_matrix, columns=cols)

    for i in range(len(data_col)):
        if (data_col[i] < 10):
            dumnie.iloc[i, 0] = 1
        elif (data_col[i] < 20):
            dumnie.iloc[i, 1] = 1
        elif (data_col[i] < 25):
            dumnie.iloc[i, 2] = 1
        elif (data_col[i] < 60):
            dumnie.iloc[i, 3] = 1
        elif (data_col[i] >= 60):
            dumnie.iloc[i, 4] = 1

    data = data.drop(labels='Age', axis=1)
    data_joined = data.join(dumnie.add_prefix('Age_'))

    print('One-hot Encoding Success')
    return data_joined


def do_oneHot(data):  # do one-hot encoding in entire dataframe
    data = one_hot(data, data.columns.get_loc('Symptoms'), 'Symptoms_')
    data = one_hot(data, data.columns.get_loc('Experiencig_Symptoms'), 'ExpSympt_')
    data = one_hot_age(data)  # Column 'Age' need to grouping, so use different one-hot-encoding function
    data = one_hot(data, data.columns.get_loc('Gender'), 'Gender_')
    data = one_hot(data, data.columns.get_loc('Severity'), 'Severity_')
    data = one_hot(data, data.columns.get_loc('Contact'), 'Contact_')

    # in one_hot def, index of Contact is 0 so that data.iloc[:, data_idx:] is not work
    data = data.drop(labels='Contact', axis=1)

    data.to_csv('./data/after_oneHot.csv', index=False)  # save the result in the csv file


def get_Conditions(data):  # merge 'Severity_' columns in String
    print('make heatmap start')
    df = data.copy()
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


def get_score(data):  # get Symptoms score using Symptoms_sum and ExpSymps_sum
    df = data.copy()
    idx_syp_start = data.columns.get_loc('Symptoms_Fever')
    idx_syp_end = data.columns.get_loc('Symptoms_None-Symptom')
    idx_expSyp_start = data.columns.get_loc('ExpSympt_Pains')
    idx_expSyp_end = data.columns.get_loc('ExpSympt_None_Experiencing')
    df['Symptoms_Score'] = df.iloc[:, idx_syp_start:idx_syp_end].sum(axis=1) + df.iloc[:,
                                                                               idx_expSyp_start:idx_expSyp_end].sum(
        axis=1)
    # df = df.join(df['Symptoms_Score'])

    return df


def get_htmp_after_encod(data):
    df = data.copy()
    df = get_score(df)
    df = get_condition(df)

    from pylab import rcParams
    rcParams['figure.figsize'] = 13, 18
    corrmat = df.corr()
    cols = corrmat.index
    cm = np.corrcoef(df[cols].values.T)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',
                     annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
    plt.title('Heatmap of All data')
    plt.show()
    print('make heatmap success')

    return corrmat  # return correlation matrix data


def do_PCA_partial(data):  # PCA just for partial (drop severity:one-hot and Condition: string)
    dt = data.copy()
    severity_columns = dt.filter(like='Severity_').columns
    dt.drop(severity_columns, axis=1, inplace=True)
    dt.drop('Condition', axis=1, inplace=True)

    scaler = preprocessing.StandardScaler()
    train_df_scaled = scaler.fit_transform(dt)

    pca = PCA(n_components=2)
    df = pca.fit_transform(train_df_scaled)
    pca_df = pd.DataFrame(df, columns=['P1', 'P2'])

    res_ratio = pca.explained_variance_ratio_
    print('Variance ratio after PCA is {}'.format(res_ratio))

    return pca_df


def do_PCA_all(data):  # PCA for all data (with severity, drop Condition:string)

    df = data.copy()
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

    res_ratio = pca.explained_variance_ratio_
    print('Variance ratio after PCA is {}'.format(res_ratio))
    print()

    return df_pca


def do_preprocessing(data):
    data_cond = get_Conditions(data_oh)  # data with Condition (type: string)
    cor = get_htmp_after_encod(data_oh)  # getting correlation matrix
    data_co = get_condition(data_oh)  # data with Condition (type: int)

    symp_pca_1 = do_PCA_partial(data_oh[data_oh.filter(like='Symptoms_').columns].join(data_co['Condition']))
    expSymp_pca_1 = do_PCA_partial(data_oh[data_oh.filter(like='ExpSympt_').columns].join(data_co['Condition']))

    symp_pca_2 = do_PCA_all(data_oh[data_oh.filter(like='Symptoms_').columns].join(data_co['Condition']))
    expSymp_pca_2 = do_PCA_all(data_oh[data_oh.filter(like='ExpSympt_').columns].join(data_co['Condition']))


'''
Visualization Functions
'''


def get_symptom_count(the_list):
    return sum(the_list.values)


'''
Analysis & Evaluation Functions
'''


def do_analysis(df):
    # Set indicators and target features
    indicators = ['Symptoms_Fever', 'Symptoms_Tiredness', 'Symptoms_Dry-Cough', 'Symptoms_Difficulty-in-Breathing',
                  'Symptoms_Sore-Throat', 'ExpSympt_Pains', 'ExpSympt_Nasal-Congestion',
                  'ExpSympt_Runny-Nose', 'ExpSympt_Diarrhea', 'Age_0-9', 'Age_10-19', 'Age_20-24', 'Age_25-59',
                  'Age_60+', 'Gender_Male', 'Gender_Female', 'Gender_Transgender']
    target_columns = ['Severity_None', 'Severity_Mild', 'Severity_Moderate', 'Severity_Severe']
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


def do_cm(model, xTest, yTest):
    y_pred = model.predict(xTest)
    cm = confusion_matrix(yTest, y_pred)  # Set confusion matrix
    # sns.heatmap(cm, annot=True)
    # plt.show()
    print(cm)  # Print result


def do_evaluation(tune, xTest, yTest, kF):
    # Do evaluation with K-Fold (k=10)
    scores = cross_val_score(tune, xTest, yTest, cv=kF, n_jobs=1, scoring="accuracy")
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
print(data)
print()

# understanding basic data information - technical statistics
print(data.info())
print()
print(data.isna().sum())
print()
print(data.describe())
print()
print(data['Country'].unique())
print()
print(data['Symptoms'].unique())
print()
print(data['Experiencig_Symptoms'].unique())
print()
print(data['Age'].unique())
print()
print(data['Gender'].unique())
print()
print(data['Severity'].unique())
print()
print(data['Contact'].unique())
print()

'''
Data PreProcessing
'''

data = change_wrong_name(data)  # change wrong file name 'None-Sympton' to 'None-Symptom'
data = chk_wrong_age_and_treat(data)  # treat wrong data
data.dropna(how='any', inplace=True)  # drop rows with dirty data (except Age error)
data.drop('Country', axis=1, inplace=True)  # drop Column 'Country'
print(data)

data1 = get_counts(data)
print(data1['Count_Symptoms'])

do_oneHot(data)  # one-hot encoding

data_oh = pd.read_csv('./data/after_oneHot.csv')
do_preprocessing(data_oh)

'''
Visualization
'''

# countplot
fig, ax = plt.subplots(1, 4, figsize=(20, 5))
sns.countplot(data_oh['Severity_None'], ax=ax[0])
sns.countplot(data_oh['Severity_Mild'], ax=ax[1])
sns.countplot(data_oh['Severity_Moderate'], ax=ax[2])
sns.countplot(data_oh['Severity_Severe'], ax=ax[3])

plt.suptitle('Number of patients by severity (0.0 = Not Applicable, 1.0 = Applicable)', fontsize=20)
fig.tight_layout()

# barplot
indicators = ['Symptoms_Fever', 'Symptoms_Tiredness', 'Symptoms_Dry-Cough', 'Symptoms_Difficulty-in-Breathing',
              'Symptoms_Sore-Throat', 'ExpSympt_Pains', 'ExpSympt_Nasal-Congestion',
              'ExpSympt_Runny-Nose', 'ExpSympt_Diarrhea', 'Age_0-9', 'Age_10-19', 'Age_20-24', 'Age_25-59', 'Age_60+',
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

# piechart
plt.figure(figsize=(11, 11))
plt.pie(data=temp_df, x="Occurence_Count", labels=temp_df["Indicator"])
plt.show()

# countplot - by severity

indicators2 = ['Symptoms_Fever', 'Symptoms_Tiredness', 'Symptoms_Dry-Cough', 'Symptoms_Difficulty-in-Breathing',
               'Symptoms_Sore-Throat', 'ExpSympt_Pains', 'ExpSympt_Nasal-Congestion',
               'ExpSympt_Runny-Nose', 'ExpSympt_Diarrhea', 'Age_0-9', 'Age_10-19', 'Age_20-24', 'Age_25-59', 'Age_60+',
               'Gender_Male',
               'Gender_Female', 'Gender_Transgender', 'Severity_None']
indicators3 = ['Symptoms_Fever', 'Symptoms_Tiredness', 'Symptoms_Dry-Cough', 'Symptoms_Difficulty-in-Breathing',
               'Symptoms_Sore-Throat', 'ExpSympt_Pains', 'ExpSympt_Nasal-Congestion',
               'ExpSympt_Runny-Nose', 'ExpSympt_Diarrhea', 'Age_0-9', 'Age_10-19', 'Age_20-24', 'Age_25-59', 'Age_60+',
               'Gender_Male',
               'Gender_Female', 'Gender_Transgender', 'Severity_Mild']
indicators4 = ['Symptoms_Fever', 'Symptoms_Tiredness', 'Symptoms_Dry-Cough', 'Symptoms_Difficulty-in-Breathing',
               'Symptoms_Sore-Throat', 'ExpSympt_Pains', 'ExpSympt_Nasal-Congestion',
               'ExpSympt_Runny-Nose', 'ExpSympt_Diarrhea', 'Age_0-9', 'Age_10-19', 'Age_20-24', 'Age_25-59', 'Age_60+',
               'Gender_Male',
               'Gender_Female', 'Gender_Transgender', 'Severity_Moderate']
indicators5 = ['Symptoms_Fever', 'Symptoms_Tiredness', 'Symptoms_Dry-Cough', 'Symptoms_Difficulty-in-Breathing',
               'Symptoms_Sore-Throat', 'ExpSympt_Pains', 'ExpSympt_Nasal-Congestion',
               'ExpSympt_Runny-Nose', 'ExpSympt_Diarrhea', 'Age_0-9', 'Age_10-19', 'Age_20-24', 'Age_25-59', 'Age_60+',
               'Gender_Male',
               'Gender_Female', 'Gender_Transgender', 'Severity_Severe']

features['Total_Symptom'] = features[indicators].apply(get_symptom_count, axis=1)
feats = data_oh[indicators2]
feats['Total_Symptom'] = feats[indicators].apply(get_symptom_count, axis=1)

plt.figure(figsize=(10, 10))
ax1 = sns.countplot(data=feats, x='Total_Symptom', hue='Severity_None')
plt.xlabel("Total symptom occurence on someone")
for p in ax1.patches:
    height = p.get_height()
    ax1.text(p.get_x() + p.get_width() / 2., height + 5, height, ha='center', size=9)

plt.show()

feats2 = data_oh[indicators3]
feats2['Total_Symptom'] = feats2[indicators].apply(get_symptom_count, axis=1)

plt.figure(figsize=(10, 10))
ax2 = sns.countplot(data=feats2, x='Total_Symptom', hue='Severity_Mild')
plt.xlabel("Total symptom occurence on someone")
for p in ax2.patches:
    height = p.get_height()
    ax2.text(p.get_x() + p.get_width() / 2., height + 5, height, ha='center', size=9)
plt.show()

feats3 = data_oh[indicators4]
feats3['Total_Symptom'] = feats3[indicators].apply(get_symptom_count, axis=1)

plt.figure(figsize=(10, 10))
ax3 = sns.countplot(data=feats3, x='Total_Symptom', hue='Severity_Moderate')
plt.xlabel("Total symptom occurence on someone")
for p in ax3.patches:
    height = p.get_height()
    ax3.text(p.get_x() + p.get_width() / 2., height + 5, height, ha='center', size=9)
plt.show()

feats4 = data_oh[indicators5]
feats4['Total_Symptom'] = feats4[indicators].apply(get_symptom_count, axis=1)

plt.figure(figsize=(10, 10))
ax4 = sns.countplot(data=feats4, x='Total_Symptom', hue='Severity_Severe')
plt.xlabel("Total symptom occurence on someone")
for p in ax4.patches:
    height = p.get_height()
    ax4.text(p.get_x() + p.get_width() / 2., height + 5, height, ha='center', size=9)
plt.show()


'''
Data Analysis & Evaluation
'''

do_analysis(data_oh)
