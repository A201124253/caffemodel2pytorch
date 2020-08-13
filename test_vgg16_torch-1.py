import torchvision.models as models
import torch
import collections
import numpy

model = models.vgg16()
# model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg16', pretrained=True)
# model.load_state_dict(torch.load('remove-fc-vgg16.caffemodel.pt'))
# print(model)

model_features = torch.nn.Sequential(collections.OrderedDict(zip(['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2', 'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3', 'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'pool4', 'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'pool5'], model.features)))
state_dict = torch.load('remove-fc-vgg16.caffemodel.pt')
model_features.load_state_dict({l : torch.from_numpy(numpy.array(v)).view_as(p) for k, v in state_dict.items() for l, p in model_features.named_parameters() if k in l})
print(model_features)