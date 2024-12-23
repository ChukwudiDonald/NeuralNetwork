import numpy as np
import matplotlib.pyplot as plt
import  torch.nn as nn
import torch
import matplotlib_inline.backend_inline as mpl_inline
mpl_inline.set_matplotlib_formats('svg')  # New method


N = 30
x = torch.randn(N,1)
y = x + torch.randn(N,1)/2

# plt.plot(x,y,'s')
# plt.show()

ANNreg = nn.Sequential(
    nn.Linear(1,1),
    nn.ReLU(),
    nn.Linear(1,1)
)

# print(ANNreg)

learningRate = 0.05
lossfun = nn.MSELoss()

optmizer = torch.optim.SGD(ANNreg.parameters(),lr=learningRate)

no_epochs = 250
losses = torch.zeros(no_epochs)

for epoch in range(no_epochs):
    yHat = ANNreg(x)

    loss = lossfun(yHat,y)
    losses[epoch] = loss

    optmizer.zero_grad()
    loss.backward()
    optmizer.step()


predictions =  ANNreg(x)

testloss = (predictions-y).pow(2).mean()

# plt.plot(losses.detach(),'o',markerfacecolor='w', linewidth=.1)
# plt.plot(no_epochs,testloss.detach(),"ro")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Final loss = %g" %testloss.item())
# plt.show()

plt.plot(x,y,'bo', label="RealData")
plt.plot(x,predictions.detach(),'rs',label='Predictions')
plt.title(f"prediction-data r={np.corrcoef(y.T,predictions.detach().T)[0,1]: .2f}")
plt.legend()
plt.show()
