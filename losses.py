def MSE(yhat, y):
    return sum((yhat - y)**2) / 2


MSE.dif = lambda yhat, y: yhat - y
