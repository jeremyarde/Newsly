from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier


def run_sklearn(x_train, y_train, x_test, y_test, predict_text):
    print("Linear SVC")
    svc = LinearSVC(verbose=True)
    svc.fit(X=x_train, y=y_train)
    print(svc.coef_)
    mean = svc.score(x_test, y_test)
    mean_string = f"Mean accuracy: {mean}"
    print(mean_string)
    predict(svc, predict_text)

    # Decision Tree
    print("Decision Tree")
    dtc = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)
    scores = cross_val_score(dtc, x_train, y_train, cv=5)
    print(scores.mean())

    # Random Forrest
    print("Random Forest")
    rfc = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
    scores = cross_val_score(rfc, x_train, y_train, cv=5)
    print(scores.mean())

    # # Grid search
    # svc = SVC(gamma='scale')
    # parameters = {'kernel':('linear', 'rbf'), 'C': [1, 10]}
    # print('Starting Gridsearch...')
    # clf = GridSearchCV(svc, parameters, cv=5, verbose=True, n_jobs=5)
    # clf.fit(x_train, y_train)
    # sorted(clf.cv_results_.keys())
    # clf.score(x_test, y_test)

def predict(classifier, text):
    # vectorizer = TfidfVectorizer()
    # vectorized_text = vectorizer.transform([text])
    prediction = classifier.predict(text)

    print(f"Prediction results: {prediction}")
