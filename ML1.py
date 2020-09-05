from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.datasets import load_iris

iris = load_iris() #['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)'],


x = iris.data         # iris['data']
y = iris.target       # iris['target']



print(iris['target_names'])

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

Lin = LinearRegression()

Lin.fit(x_train,y_train)


print(Lin.coef_) # Here we find b1 of equation b0+b1*x

print(Lin.intercept_) # intercept (b0)

Pred_y = Lin.predict(x_test)

acc = r2_score(y_test,Pred_y)
print(acc)


print(mean_absolute_error(y_test,Pred_y))
print(mean_squared_error(y_test,Pred_y))
