from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,mean_absolute_error,mean_squared_error
from sklearn.datasets import load_iris

iris = load_iris() #['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)'],


x = iris.data         # iris['data']
y = iris.target       # iris['target']

print(iris['target_names'])

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


Lin = LogisticRegression(random_state=0)

Lin.fit(x_train,y_train)

Pred_y = Lin.predict(x_test)

acc = accuracy_score(y_test,Pred_y)
print(acc)
print(mean_absolute_error(y_test,Pred_y))



print(mean_squared_error(y_test,Pred_y))
