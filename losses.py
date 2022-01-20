def MSE(yhat, y):
    return sum((y - yhat)**2) / 2


MSE.dif = lambda yhat, y: y - yhat
