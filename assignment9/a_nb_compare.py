from load_data import load_data
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score

X, y = load_data()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)
g_pred = gnb.predict(X_test)

# Multinomial Naive Bayes
mnb = MultinomialNB()
mnb.fit(X_train, y_train)
m_pred = mnb.predict(X_test)

print("GaussianNB Accuracy:", accuracy_score(y_test, g_pred))
print("MultinomialNB Accuracy:", accuracy_score(y_test, m_pred))