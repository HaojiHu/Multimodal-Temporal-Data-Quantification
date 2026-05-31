import pandas as pd
import numpy as np
from variants_for_ablation_study import NoMixtureNoClusteredMI, NoClusteredMI
from scipy.io import arff, loadmat
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, mutual_info_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from normalize_clustered_mi import NormalizedClusteredMI
import mixed


path = './data/feature_selection/'


files = ['COIL20.mat', 'ORL.mat', 'Yale.mat', 'warpPIE10P.mat', 'Isolet.mat', 'TOX_171.mat', 'USPS.mat', 'CLL_SUB_111.mat']
rand_seed = 42

win_data_rf_ross = []
win_data_svm_ross = []
win_data_lr_ross = []
tie_data_rf_ross = []
tie_data_svm_ross = []
tie_data_lr_ross = []

win_data_rf_mixture = []
win_data_svm_mixture = []
win_data_lr_mixture = []
tie_data_rf_mixture = []
tie_data_svm_mixture = []
tie_data_lr_mixture = []

win_data_rf_noclustering = []
win_data_svm_noclustering = []
win_data_lr_noclustering = []
tie_data_rf_noclustering = []
tie_data_svm_noclustering = []
tie_data_lr_noclustering = []

rf_ross = 0
svm_ross = 0
lr_ross = 0

Tie_rf_ross = 0
Tie_svm_ross = 0
Tie_lr_ross = 0

rf_mixture = 0
svm_mixture = 0
lr_mixture = 0

Tie_rf_mixture = 0
Tie_svm_mixture = 0
Tie_lr_mixture = 0

rf_noclustering = 0
svm_noclustering = 0
lr_noclustering = 0

Tie_rf_noclustering = 0
Tie_svm_noclustering = 0
Tie_lr_noclustering = 0

total = 0

def symbol2id(disc):
    res = []
    symbol_dict = {}
    idx = 1
    for i, s in enumerate(disc):
        if s not in symbol_dict:
            symbol_dict[s] = idx
        symbol_id = symbol_dict[s]
        res.append(symbol_id)

    return res



for file in files:
    file_name = path + file
    print('---------------------------------')
    print(file)

    if '.arff' in file_name:
        arff_file = arff.loadarff(file_name)
        dfs = pd.DataFrame(arff_file[0])
        class_column = 'Class'
    elif '.mat' in file_name:
        mat = loadmat(file_name)
        dfs = pd.DataFrame(np.concatenate( [mat['X'], mat['Y']], axis=1))
        class_column = dfs.columns[-1]
    # dfs = pd.read_excel(file_name)
    # print(dfs)

    print('#instance: ', len(dfs), '#feature: ', len(dfs.columns)-1, ' #class', len(dfs[class_column].unique()))
    print(len(dfs), ' ', len(dfs.columns)-1, ' ', len(dfs[class_column].unique()))

    def category2id(category):
        cat = category.unique()
        cat2id = {}
        id = 1
        for c in cat:
            cat2id[c] = id
            id += 1

        cat_id = []
        for c in category:
            cat_id.append(cat2id[c])
        return np.array(cat_id)

    category = category2id(dfs[class_column])
    feature_importance1 = []
    feature_importance_noclustering = []
    feature_importance_ross = []
    feature_importance_mix = []
    columns = []
    for col in dfs.columns:
        if col == class_column:
            continue
        feature = np.array(list(map(float, dfs[col].tolist())))
        # mi = NMI(category, feature)
        mi = NormalizedClusteredMI(category, feature)
        mi_ross = mutual_info_classif(np.expand_dims(feature, axis=1), category, n_neighbors=3)
        mi_noclustering = NoClusteredMI(category, feature)

        # print(col, ': ', round(mi, 5), ' vs ', round(mi1[0],5))
        # n = len(discs[:size - tau])
        x = np.asarray(category)
        y = np.asarray(feature)
        mi_mixture = mixed.Mixed_KSG(category, feature)




        feature_importance1.append(round(mi, 5))
        feature_importance_ross.append(round(mi_ross[0],5)) # for sklearner package
        feature_importance_mix.append(round(mi_mixture,5))
        feature_importance_noclustering.append(round(mi_noclustering, 5))
        columns.append(col)

    def run_randomForest(X_train, X_test, y_train, y_test):
        clf = RandomForestClassifier(n_estimators=100, random_state=rand_seed, n_jobs=-1)
        # clf = LogisticRegression(random_state=0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        # print('Accuracy on test set: ')
        # print(accuracy_score(y_test, y_pred))
        return round(accuracy_score(y_test, y_pred), 4)

    def run_SVM(X_train, X_test, y_train, y_test):

        clf = SVC(kernel='linear', C=1.0, random_state=rand_seed)  # Linear kernel

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        # print('Accuracy on test set: ')
        # print(accuracy_score(y_test, y_pred))
        return round(accuracy_score(y_test, y_pred), 4)


    def run_LR(X_train, X_test, y_train, y_test):

        clf = LogisticRegression(solver='newton-cg', max_iter=1000, random_state=rand_seed)

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        # print('Accuracy on test set: ')
        # print(accuracy_score(y_test, y_pred))
        return round(accuracy_score(y_test, y_pred), 4)


    X = dfs.drop(class_column, axis=1).to_numpy()
    y = category2id(dfs[class_column])
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)
    #
    # run_randomForest(X_train, X_test, y_train, y_test)

    sort_idx1 = np.argsort(feature_importance1)[::-1]
    sort_idx_ross = np.argsort(feature_importance_ross)[::-1]
    sort_idx_mixture = np.argsort(feature_importance_mix)[::-1]
    sort_idx_noclustering = np.argsort(feature_importance_noclustering)[::-1]
    # importance1 = feature_importance1[sort_idx1][:topk]
    # importance2 = feature_importance2[sort_idx2][:topk]

    # print(columns)
    def print_index(sort_idx):
        res = '['
        for i, idx in enumerate(sort_idx):
            if i == 0:
                res += str(idx)
            else:
                res += ',' + str(idx)
        res += ']'
        return res


    topk = 3
    res1 = print_index(sort_idx1[:topk])
    res_ross = print_index(sort_idx_ross[:topk])

    print(res1, ' vs ', res_ross)
    # print(importance1, ' vs ', importance2)


    def get_top_feature(df, topk, sort_idx):
        topk_columns = []
        for i in range(topk):
            topk_columns.append(columns[sort_idx[i]])

        X = df[topk_columns].to_numpy()
        return X

    acc_list = ''
    acc_list_ross = ''
    acc_list_mixture = ''
    acc_list_noclustering = ''
    svm_list = ''
    svm_list_ross = ''
    svm_list_mixture = ''
    svm_list_noclustering = ''
    lr_list = ''
    lr_list_ross = ''
    lr_list_mixture = ''
    lr_list_noclustering = ''

    win_rf_ross = 0
    win_svm_ross = 0
    win_lr_ross = 0
    tie_rf_ross = 0
    tie_svm_ross = 0
    tie_lr_ross = 0

    win_rf_mixture = 0
    win_svm_mixture = 0
    win_lr_mixture = 0
    tie_rf_mixture = 0
    tie_svm_mixture = 0
    tie_lr_mixture = 0

    win_rf_noclustering = 0
    win_svm_noclustering = 0
    win_lr_noclustering = 0
    tie_rf_noclustering = 0
    tie_svm_noclustering = 0
    tie_lr_noclustering = 0
    # for topk in range(5, 30, 5):
    feature_num = []

    for topk in range(20, 110, 20):
        total += 1
        feature_num.append(topk)

        X = get_top_feature(dfs, topk, sort_idx1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)
        accu = run_randomForest(X_train, X_test, y_train, y_test)
        acc_list += str(accu) + '  '

        X = get_top_feature(dfs, topk, sort_idx_ross)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)
        accu_ross = run_randomForest(X_train, X_test, y_train, y_test)
        acc_list_ross += str(accu_ross) + '  '

        X = get_top_feature(dfs, topk, sort_idx_mixture)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)
        accu_mixture = run_randomForest(X_train, X_test, y_train, y_test)
        acc_list_mixture += str(accu_mixture) + '  '

        X = get_top_feature(dfs, topk, sort_idx_noclustering)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)
        accu_noclustering = run_randomForest(X_train, X_test, y_train, y_test)
        acc_list_noclustering += str(accu_noclustering) + '  '

        if accu > accu_ross:
            win_rf_ross += 1
            rf_ross += 1
        elif accu == accu_ross:
            tie_rf_ross += 1
            Tie_rf_ross += 1

        if accu > accu_mixture:
            win_rf_mixture += 1
            rf_mixture += 1
        elif accu == accu_mixture:
            tie_rf_mixture += 1
            Tie_rf_mixture += 1

        if accu > accu_noclustering:
            win_rf_noclustering += 1
            rf_noclustering += 1
        elif accu == accu_noclustering:
            tie_rf_noclustering += 1
            Tie_rf_noclustering += 1

        X = get_top_feature(dfs, topk, sort_idx1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)
        accu = run_SVM(X_train, X_test, y_train, y_test)
        svm_list += str(accu) + '  '

        X = get_top_feature(dfs, topk, sort_idx_ross)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)
        accu_ross = run_SVM(X_train, X_test, y_train, y_test)
        svm_list_ross += str(accu_ross) + '  '

        X = get_top_feature(dfs, topk, sort_idx_mixture)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)
        accu_mixture = run_SVM(X_train, X_test, y_train, y_test)
        svm_list_mixture += str(accu_mixture) + '  '

        X = get_top_feature(dfs, topk, sort_idx_noclustering)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)
        accu_noclustering = run_SVM(X_train, X_test, y_train, y_test)
        svm_list_noclustering += str(accu_noclustering) + '  '

        if accu > accu_ross:
            win_svm_ross += 1
            svm_ross += 1
        elif accu == accu_ross:
            tie_svm_ross += 1
            Tie_svm_ross += 1

        if accu > accu_mixture:
            win_svm_mixture += 1
            svm_mixture += 1
        elif accu == accu_mixture:
            tie_svm_mixture += 1
            Tie_svm_mixture += 1

        if accu > accu_noclustering:
            win_svm_noclustering += 1
            svm_noclustering += 1
        elif accu == accu_noclustering:
            tie_svm_noclustering += 1
            Tie_svm_noclustering += 1

        X = get_top_feature(dfs, topk, sort_idx1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)
        accu = run_LR(X_train, X_test, y_train, y_test)
        lr_list += str(accu) + '  '

        X = get_top_feature(dfs, topk, sort_idx_ross)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)
        accu_ross = run_LR(X_train, X_test, y_train, y_test)
        lr_list_ross += str(accu_ross) + '  '

        X = get_top_feature(dfs, topk, sort_idx_mixture)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)
        accu_mixture = run_LR(X_train, X_test, y_train, y_test)
        lr_list_mixture += str(accu_mixture) + '  '

        X = get_top_feature(dfs, topk, sort_idx_noclustering)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)
        accu_noclustering = run_LR(X_train, X_test, y_train, y_test)
        lr_list_noclustering += str(accu_noclustering) + '  '

        if accu > accu_ross:
            win_lr_ross += 1
            lr_ross += 1
        elif accu == accu_ross:
            tie_lr_ross += 1
            Tie_lr_ross += 1

        if accu > accu_mixture:
            win_lr_mixture += 1
            lr_mixture += 1
        elif accu == accu_mixture:
            tie_lr_mixture += 1
            Tie_lr_mixture += 1

        if accu > accu_noclustering:
            win_lr_noclustering += 1
            lr_noclustering += 1
        elif accu == accu_noclustering:
            tie_lr_noclustering += 1
            Tie_lr_noclustering += 1

    print('feature number: ', feature_num)
    print('RF Ross: ', acc_list_ross)
    print('RF Mixture: ', acc_list_mixture)
    print('RF NoClu.: ', acc_list_noclustering)
    print('RF proposed: ', acc_list)
    print('SVM Ross: ', svm_list_ross)
    print('SVM Mixture: ', svm_list_mixture)
    print('SVM NoClu: ', svm_list_noclustering)
    print('SVM proposed: ', svm_list)
    print('LR Ross: ', lr_list_ross)
    print('LR Mixture: ', lr_list_mixture)
    print('LR NoClu: ', lr_list_noclustering)
    print('LR proposed: ', lr_list)

    if win_rf_ross + tie_rf_ross >= 3:
        win_data_rf_ross.append(file)
    if win_svm_ross + tie_svm_ross >= 3:
        win_data_svm_ross.append(file)
    if win_lr_ross + tie_lr_ross >= 3:
        win_data_lr_ross.append(file)

    if tie_rf_ross >= 3:
        tie_data_rf_ross.append(file)
    if tie_rf_ross >= 3:
        tie_data_svm_ross.append(file)
    if tie_lr_ross >= 3:
        tie_data_lr_ross.append(file)

    if win_rf_mixture + tie_rf_mixture >= 3:
        win_data_rf_mixture.append(file)
    if win_svm_mixture + tie_svm_mixture >= 3:
        win_data_svm_mixture.append(file)
    if win_lr_mixture + tie_lr_mixture >= 3:
        win_data_lr_mixture.append(file)

    if tie_rf_mixture >= 3:
        tie_data_rf_mixture.append(file)
    if tie_rf_mixture >= 3:
        tie_data_svm_mixture.append(file)
    if tie_lr_mixture >= 3:
        tie_data_lr_mixture.append(file)

    if win_rf_noclustering + tie_rf_noclustering >= 3:
        win_data_rf_noclustering.append(file)
    if win_svm_noclustering + tie_svm_noclustering >= 3:
        win_data_svm_noclustering.append(file)
    if win_lr_noclustering + tie_lr_noclustering >= 3:
        win_data_lr_noclustering.append(file)

    if tie_rf_noclustering >= 3:
        tie_data_rf_noclustering.append(file)
    if tie_rf_noclustering >= 3:
        tie_data_svm_noclustering.append(file)
    if tie_lr_noclustering >= 3:
        tie_data_lr_noclustering.append(file)

print('Win over Ross in random forest: ', len(win_data_rf_ross), ' out of ', len(files))
print(win_data_rf_ross)

print('Win over Ross in SVM: ', len(win_data_svm_ross), ' out of ', len(files))
print(win_data_svm_ross)

print('Win over Ross in logistic regression: ', len(win_data_lr_ross), ' out of ', len(files))
print(win_data_lr_ross)

print('Tie over Ross in random forest: ', len(tie_data_rf_ross), ' out of ', len(files))
print(tie_data_rf_ross)

print('Tie over Ross in SVM: ', len(tie_data_svm_ross), ' out of ', len(files))
print(tie_data_svm_ross)

print('Tie over Ross in logistic regression: ', len(tie_data_lr_ross), ' out of ', len(files))
print(tie_data_lr_ross)

print('Win over Mixture in random forest: ', len(win_data_rf_mixture), ' out of ', len(files))
print(win_data_rf_mixture)

print('Win over Mixture in SVM: ', len(win_data_svm_mixture), ' out of ', len(files))
print(win_data_svm_mixture)

print('Win over Mixture in logistic regression: ', len(win_data_lr_mixture), ' out of ', len(files))
print(win_data_lr_mixture)

print('Tie over Mixture in random forest: ', len(tie_data_rf_mixture), ' out of ', len(files))
print(tie_data_rf_mixture)

print('Tie over Mixture in SVM: ', len(tie_data_svm_mixture), ' out of ', len(files))
print(tie_data_svm_mixture)

print('Tie over Mixture in logistic regression: ', len(tie_data_lr_mixture), ' out of ', len(files))
print(tie_data_lr_mixture)

print('Win over NoClustering in random forest: ', len(win_data_rf_noclustering), ' out of ', len(files))
print(win_data_rf_noclustering)

print('Win over NoClustering in SVM: ', len(win_data_svm_noclustering), ' out of ', len(files))
print(win_data_svm_noclustering)

print('Win over NoClustering in logistic regression: ', len(win_data_lr_noclustering), ' out of ', len(files))
print(win_data_lr_noclustering)

print('Tie over NoClustering in random forest: ', len(tie_data_rf_noclustering), ' out of ', len(files))
print(tie_data_rf_noclustering)

print('Tie over NoClustering in SVM: ', len(tie_data_svm_noclustering), ' out of ', len(files))
print(tie_data_svm_noclustering)

print('Tie over NoClustering in logistic regression: ', len(tie_data_lr_noclustering), ' out of ', len(files))
print(tie_data_lr_noclustering)


print('proposed vs ross: ')
print('win cases: ', rf_ross + svm_ross + lr_ross)
print('tie cases: ', Tie_rf_ross + Tie_svm_ross + Tie_lr_ross)
print('total cases: ', total*3)
print('win percentage: ', (rf_ross + svm_ross + lr_ross)/(total*3)) # 0.5333
print('tie percentage: ', (Tie_rf_ross + Tie_svm_ross + Tie_lr_ross)/(total*3)) # 0.1583
print('lose percentage: ', 1 - (rf_ross + svm_ross + lr_ross)/(total*3) - (Tie_rf_ross + Tie_svm_ross + Tie_lr_ross)/(total*3))

print('proposed vs mixture: ')
print('win cases: ', rf_mixture + svm_mixture + lr_mixture)
print('tie cases: ', Tie_rf_mixture + Tie_svm_mixture + Tie_lr_mixture)
print('total cases: ', total*3)
print('win percentage: ', (rf_mixture + svm_mixture + lr_mixture)/(total*3)) # 0.5333
print('tie percentage: ', (Tie_rf_mixture + Tie_svm_mixture + Tie_lr_mixture)/(total*3)) # 0.1583
print('lose percentage: ', 1 - (rf_mixture + svm_mixture + lr_mixture)/(total*3) - (Tie_rf_mixture + Tie_svm_mixture + Tie_lr_mixture)/(total*3))

print('proposed vs noclustering: ')
print('win cases: ', rf_noclustering + svm_noclustering + lr_noclustering)
print('tie cases: ', Tie_rf_noclustering + Tie_svm_noclustering + Tie_lr_noclustering)
print('total cases: ', total*3)
print('win percentage: ', (rf_noclustering + svm_noclustering + lr_noclustering)/(total*3)) # 0.5333
print('tie percentage: ', (Tie_rf_noclustering + Tie_svm_noclustering + Tie_lr_noclustering)/(total*3)) # 0.1583
print('lose percentage: ', 1 - (rf_noclustering + svm_noclustering + lr_noclustering)/(total*3) - (Tie_rf_noclustering + Tie_svm_noclustering + Tie_lr_noclustering)/(total*3))