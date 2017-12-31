import numpy as np
import torch
from torch.autograd import Variable
from alexnet import alexnet
import os
import cv2
import random
import labels
import sys

if not __name__ == '__main__':
    sys.exit(-1)

if len(sys.argv) != 2:
    print("Error in arguments")
    sys.exit(-1)

image_path = sys.argv[1]

cl_value = 1

class JustIdentity(torch.nn.Module):
    def __init__(self):
        super(JustIdentity, self).__init__()
        self.noise = torch.nn.Parameter(torch.rand(1, 3, 227, 227).double())

    def forward(self, x):
        x = torch.clamp(self.noise, -cl_value, cl_value) + x.double()
        return x.float()

class FoolLoss(torch.nn.Module):
    def __init__(self, index):
        super(FoolLoss, self).__init__()
        self.index = index
        self.layer = torch.nn.Softmax()
    def forward(self, x):
        x = self.layer.forward(x)
        return 1 - x[0][self.index]



model=alexnet(True)
# We don't require to fine-tune weights of the model
# So just turn of the gradient of the weights
for param in model.parameters():
    param.requires_grad = False

def reshape_image(image):
    image = image.astype(np.float32)
    return np.expand_dims(np.squeeze(np.swapaxes(np.expand_dims(image, axis=0), 3,0), axis=3), axis=0)

def reshape_back(the_image):
    # the_image is of [1, 3, H, W]
    # resize it to [H,W,3]
    the_image = np.expand_dims(the_image, axis=4) # [1, 3, H, W, 1]
    the_image = np.swapaxes(the_image, 1, 4) # [1, 1, H,W,3]
    the_image = np.squeeze(the_image, axis=0) # [1, H,W,3]
    the_image = np.squeeze(the_image, axis=0) # [H,W,3]
    return the_image

image = cv2.imread(image_path,1)
image = cv2.resize(image, dsize=(227,227))
logits = model.forward(Variable(torch.FloatTensor(torch.from_numpy(reshape_image(image))))).data.numpy()
my_label_str = labels.get_label(np.argmax(logits, axis=1)[0])
print("The class is ", my_label_str)

# Pick second most likely class to fool with
index_required = np.argsort(logits)[0][998]

the_variable = Variable(torch.FloatTensor(torch.from_numpy(reshape_image(image))))

# The loss function
criterion = FoolLoss(index_required)
just_identity = JustIdentity()
a = list(just_identity.parameters())[0].clone()


optimizer = torch.optim.Adam(just_identity.parameters(), lr=0.01)
for i in range(100):
    loss=criterion(model.forward(just_identity.forward(the_variable)))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("Loss", loss.data[0])

b = list(just_identity.parameters())[0].clone()
print(torch.equal(a.data, b.data))

# TODO: Add code to predict the other class