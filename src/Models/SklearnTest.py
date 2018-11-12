from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier


def run_sklearn(x_train, y_train, x_test, y_test):
    print("Linear SVC")
    svc = LinearSVC(verbose=True)
    svc.fit(X=x_train, y=y_train)
    print(svc.coef_)
    mean = svc.score(x_test, y_test)
    mean_string = f"Mean accuracy: {mean}"
    print(mean_string)

    # Decision Tree
    print("Decision Tree")
    clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)
    scores = cross_val_score(clf, x_train, y_train, cv=5)
    print(scores.mean())

    # Random Forrest
    print("Random Forest")
    clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
    scores = cross_val_score(clf, x_train, y_train, cv=5)
    print(scores.mean())

    # Grid search
    svc = SVC(gamma='scale')
    parameters = {'kernel':('linear', 'rbf'), 'C': [1, 10]}
    print('Starting Gridsearch...')
    clf = GridSearchCV(svc, parameters, cv=5, verbose=True, n_jobs=5)
    clf.fit(x_train, y_train)
    sorted(clf.cv_results_.keys())
    clf.score(x_test, y_test)