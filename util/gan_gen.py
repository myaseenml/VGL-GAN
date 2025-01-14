import numpy
import json
import torch
from torch.autograd import Variable
import models.dcgan as dcgan
import toml
import matplotlib.pyplot as plt


boundary_value = 5.12
nz = 32

imageSize = 64
ngf = 64
ngpu = 1
n_extra_layers = 0
features = len(json.load(open('GANTrain/index2str.json')))

generator = dcgan.DCGAN_G(imageSize, nz, features, ngf, ngpu, n_extra_layers)


    

def gan_generate(x,batchSize,nz,model_path):
    generator.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    latent_vector = torch.FloatTensor(x).view(batchSize,nz, 1,1)
    with torch.no_grad():
        levels = generator(Variable(latent_vector))
    print(levels.shape)
    levels.data = levels.data[:, :, :16, :56]
    im = levels.data.cpu().numpy()
    print(im.shape)
    im = numpy.argmax( im, axis = 1)
    #img = plt.imshow(im)
    #plt.imshave(img, 'mariolsi.png')
    #from IPython import embed
    #embed()
    return json.dumps(im[0].tolist())
    
    
    
batch_size = 32
nz = 32
model_path = "GANTrain/samples/netG_epoch_4999_7684.pth"

x = torch.FloatTensor(batch_size, nz, 1, 1).normal_(0, 1)

out = gan_generate(x,batch_size, nz, model_path)


