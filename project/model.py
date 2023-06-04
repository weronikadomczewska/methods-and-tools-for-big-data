import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
# from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


def show_confusion_matrix(Y_test, Y_pred, labels, title):
    cm = confusion_matrix(Y_test, Y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=labels)
    disp.plot()
    plt.title(title)
    plt.show()

def show_roc(df, target_column, classifier, X_test, y_test, title):
    # Separate features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    classes = [0.0, 1.0]
    
    # Predict probabilities
    y_score = classifier.predict_proba(X_test)
    y_score = y_score[:, 1]  # Keep only the probabilities for the positive class
    
    # Binarize the target labels
    y_test_binarized = label_binarize(y_test, classes=classes)
    y_test_binarized = np.squeeze(y_test_binarized)
    
    # Compute ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized, y_score)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # print(y_test_binarized.shape, y_score.shape)
    
    # Plot ROC curve
    plt.figure()
    plt.plot(fpr[1], tpr[1], color='darkorange', lw=2, label='Class 1 (AUC = %0.2f)' % roc_auc[1])
    plt.plot(fpr[0], tpr[0], color='navy', lw=2, label='Class 0 (AUC = %0.2f)' % roc_auc[0])
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC - {title}')
    plt.legend(loc="lower right")
    plt.show()

def apply_classification_to_dataset(dataset, algorithm_name):
    X = dataset.iloc[:,1:]
    y = dataset.iloc[:,0]
    # classes=np.unique(y)
    classes = [0.0, 1.0]
    target_column = "PCOS(Y/N)"

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    df_train = pd.concat([X_train, y_train], axis=1)
    class_0 = df_train[df_train[target_column] == 0]
    class_1 = df_train[df_train[target_column] == 1]
    class_0_sampled = class_0.sample(n=len(class_1), random_state=42)
    df_train_balanced = pd.concat([class_0_sampled, class_1])
    X_train_balanced = df_train_balanced.drop(target_column, axis=1)
    y_train_balanced = df_train_balanced[target_column]

    title = ""
    if algorithm_name == "dt":
        clf = DecisionTreeClassifier(random_state=0)
        title = "Decision Tree Classifier"
    elif algorithm_name == "rf":
        clf = RandomForestClassifier(n_estimators=300,  criterion='gini', max_depth=10, min_samples_split=2, min_samples_leaf=1, 
                                  min_weight_fraction_leaf=0.0, max_features='log2', max_leaf_nodes=None, min_impurity_decrease=0.0
                                  , bootstrap=True,oob_score=False, n_jobs=1, random_state=None,
                                  verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)
        title = "Random Forest CLassifier"
    elif algorithm_name == "knn":
        clf = KNeighborsClassifier()
        title = "K-Nearest Neighbors Classifier"
    elif algorithm_name == "nb":
        clf = GaussianNB()
        title = "Naive Bayes Classifier"

    clf.fit(X_train_balanced, y_train_balanced)
    Y_pred = clf.predict(X_test)

    # confusion matrix
    show_confusion_matrix(y_test, Y_pred, classes, title)

    # Calculated stats
    target_names = [str(x) for x in classes]
    print(target_names)
    print(classification_report(y_test, Y_pred, target_names=target_names))

    # # roc
    show_roc(dataset, "PCOS(Y/N)", clf, X_test, y_test, title)

    

# reading data from file
pcos_type_0_datapath = "archive\PCOS_data_without_infertility.xlsx"
pcos_type_0_data = pd.read_excel(pcos_type_0_datapath, sheet_name="Full_new")

# print(pcos_type_0_data.info())
# print(pcos_type_0_data.isnull().sum())

# dropping unnecessary features
# it was necessary to convert inches to centimeters in dataset
# there is one record with fast food info missing
# there are two blood test results that are type object
pcos_type_0_data = pcos_type_0_data.drop(["Sl. No", "Patient File No.", "Marraige Status(Yrs)", "No. of aborptions", "Waist(inch)", "Hip(inch)", "Unnamed: 46"], axis=1)
pcos_type_0_data["AMH(ng/mL)"] = pd.to_numeric(pcos_type_0_data["AMH(ng/mL)"], errors="coerce")
pcos_type_0_data["II    beta-HCG(mIU/mL)"] = pd.to_numeric(pcos_type_0_data["II    beta-HCG(mIU/mL)"], errors="coerce")
pcos_type_0_data.fillna(pcos_type_0_data["Fast food (Y/N)"].median(), inplace=True)
print(pcos_type_0_data["PCOS(Y/N)"].value_counts())

# print(pcos_type_0_data.info())
# print(pcos_type_0_data.isnull().sum())


pcos_type_0_data = pcos_type_0_data.rename(columns=lambda x: x.strip())


sight_features = ["PCOS(Y/N)", "Age(yrs)", "Weight(kg)", "Height(cm)", "BMI", 
                  "Hip(cm)", "Waist(cm)", "Waist:Hip Ratio", "Weight gain(Y/N)", 
                  "hair growth(Y/N)", "Skin darkening(Y/N)", "Hair loss(Y/N)",
                  "Pimples(Y/N)", "Fast food (Y/N)", "Reg.Exercise(Y/N)", 
                  "Cycle(R/I)", "Cycle length(days)", "Pregnant(Y/N)"]

vitals = ["PCOS(Y/N)", "Blood Group", "Pulse rate(bpm)", "RR(breaths/min)", "Hb(g/dl)",
          "I   beta-HCG(mIU/mL)", "II    beta-HCG(mIU/mL)", "FSH(mIU/mL)",
          "LH(mIU/mL)", "FSH/LH", "TSH(mIU/L)", "AMH(ng/mL)", "PRL(ng/mL)",
          "Vit D3 (ng/mL)", "PRG(ng/mL)", "BP _Systolic(mmHg)", "BP _Diastolic(mmHg)", 
          "Follicle No. (L)", "Follicle No. (R)", "Avg. F size (L)(mm)", "Avg. F size (R)(mm)", 
          "Endometrium (mm)"]


sight_features_data = pcos_type_0_data[sight_features]
vitals_data = pcos_type_0_data[vitals]

# apply_classification_to_dataset(sight_features_data, "nb")
# apply_classification_to_dataset(sight_features_data, "rf")
# apply_classification_to_dataset(sight_features_data, "knn")
# apply_classification_to_dataset(sight_features_data, "dt")

# apply_classification_to_dataset(vitals_data, "nb")
# apply_classification_to_dataset(vitals_data, "rf")
# apply_classification_to_dataset(vitals_data, "knn")
# apply_classification_to_dataset(vitals_data, "dt")

# apply_classification_to_dataset(pcos_type_0_data, "nb")
# apply_classification_to_dataset(pcos_type_0_data, "rf")
# apply_classification_to_dataset(pcos_type_0_data, "knn")
# apply_classification_to_dataset(pcos_type_0_data, "dt")


 
