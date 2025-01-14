import torch
import torch.nn as nn
import torch.nn.parallel

class DCGAN_D2(nn.Module):
    def __init__(self, isize, nz, nc, ndf, ngpu, n_extra_layers=0):
        super(DCGAN_D, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential()
        # input is nc x isize x isize
        main.add_module('initial:conv:{0}-{1}'.format(nc, ndf),
                        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        main.add_module('initial:relu:{0}'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = isize / 2, ndf

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}:{1}:conv'.format(t, cndf),
                            nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}:{1}:batchnorm'.format(t, cndf),
                            nn.BatchNorm2d(cndf))
            main.add_module('extra-layers-{0}:{1}:relu'.format(t, cndf),
                            nn.LeakyReLU(0.2, inplace=True))

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid:{0}-{1}:conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid:{0}:batchnorm'.format(out_feat),
                            nn.BatchNorm2d(out_feat))
            main.add_module('pyramid:{0}:relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        # state size. K x 4 x 4
        main.add_module('final:{0}-{1}:conv'.format(cndf, 1),
                        nn.Conv2d(cndf, 1, 4, 1, 0, bias=False))
        self.main = main


    def forward(self, input):
        if self.ngpu > 1 and isinstance(input.data, torch.cuda.FloatTensor):
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else: 
            output = self.main(input)
            
        output = output.mean(0)
        return output.view(1)
        
        
class DCGAN_D(nn.Module):
    def __init__(self, isize, nz, nc, ndf, ngpu, n_extra_layers=0):
        super(DCGAN_D, self).__init__()
        self.ngpu = ngpu
        
        self.block1 = nn.Sequential(
            nn.Conv2d(17,64, kernel_size = (3,3), stride=1, padding=1),
            nn.LeakyReLU(0.2)
            
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(64,128, kernel_size = (2,2), stride=2, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2,stride=2)
            
        )
        
        self.block3 = nn.Sequential(
            nn.Conv2d(128,128, kernel_size = (2,2), stride=2, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2,stride=2)
            
        )        
        
        self.block4 = nn.Sequential(
            nn.Conv2d(128,256, kernel_size = (2,2), stride=2, padding=0),
            nn.LeakyReLU(0.2)
            
        )     
        
        self.block5 = nn.Sequential(
            nn.Conv2d(256,1, kernel_size = (2,2), stride=2, padding=0),
            nn.LeakyReLU(0.2)
            
        )  
        

        
    def forward(self, input):
        
        x = self.block1(input)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x) 
        x = self.block5(x)
        
        x = x.mean(0)
        return x.view(1)

    

class DCGAN_G2(nn.Module):
    def __init__(self, isize, nz, nc, ngf, ngpu, n_extra_layers=0):
        super(DCGAN_G, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf//2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential()
        # input is Z, going into a convolution
        main.add_module('initial:{0}-{1}:convt'.format(nz, cngf),
                        nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False))
        main.add_module('initial:{0}:batchnorm'.format(cngf),
                        nn.BatchNorm2d(cngf))
        main.add_module('initial:{0}:relu'.format(cngf),
                        nn.ReLU(True))

        csize, cndf = 4, cngf
        while csize < isize//2:
            main.add_module('pyramid:{0}-{1}:convt'.format(cngf, cngf//2),
                            nn.ConvTranspose2d(cngf, cngf//2, 4, 2, 1, bias=False))
            main.add_module('pyramid:{0}:batchnorm'.format(cngf//2),
                            nn.BatchNorm2d(cngf//2))
            main.add_module('pyramid:{0}:relu'.format(cngf//2),
                            nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}:{1}:conv'.format(t, cngf),
                            nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}:{1}:batchnorm'.format(t, cngf),
                            nn.BatchNorm2d(cngf))
            main.add_module('extra-layers-{0}:{1}:relu'.format(t, cngf),
                            nn.ReLU(True))

        main.add_module('final:{0}-{1}:convt'.format(cngf, nc),
                        nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        main.add_module('final:{0}:tanh'.format(nc),
                        nn.ReLU())#nn.Softmax(1))    #Was TANH nn.Tanh())#
        self.main = main

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else: 
            output = self.main(input)

        #print (output[0,:,0,0])
        #exit()
        return output 
        
        
class DCGAN_G(nn.Module):
    def __init__(self, isize, nz, nc, ngf, ngpu, n_extra_layers=0):
        super(DCGAN_G, self).__init__()
        self.ngpu = ngpu
        
        self.linear1 = nn.Linear(32, 128)
        self.LR = nn.LeakyReLU(0.2)
        self.linear2 = nn.Linear(128,512)
        self.LR = nn.LeakyReLU(0.2)
        self.linear3 = nn.Linear(512, 2048)
        self.LR = nn.LeakyReLU(0.2)
        self.linear4 = nn.Linear(2048,4096)        
        self.LR = nn.LeakyReLU(0.2)
        
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(256,256, kernel_size = (3,3), stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)               
        )
        
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(256,256, kernel_size = (3,3), stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)               
        )
        
        self.block3 = nn.Sequential(
            nn.ConvTranspose2d(256,128, kernel_size = (2,2), stride=2, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)               
        )
        
        self.block4 = nn.Sequential(
            nn.ConvTranspose2d(128,64, kernel_size = (2,2), stride=2, padding=0),
            #nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)               
        )

        self.block5 = nn.Sequential(
            nn.ConvTranspose2d(64,32, kernel_size = (3,3), stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2)               
        )

        self.block6 = nn.Sequential(
            nn.ConvTranspose2d(32,16, kernel_size = (4,4), stride=2, padding=1),
            #nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2)               
        )
        
        self.block7 = nn.Sequential(
            nn.ConvTranspose2d(16,16, kernel_size = (4,4), stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2)               
        )
               
        self.conv2d = nn.Conv2d(16,17, kernel_size = (3,3), stride=1, padding=1)
        
        self.LR = nn.LeakyReLU(0.2)
        
    def forward(self, input):
        x = input.view(input.size(0),-1)
        x = self.linear1(x)
        x = self.LR(x)
        x = self.linear2(x)
        x = self.LR(x)  
        x = self.linear3(x)
        x = self.LR(x)  
        x = self.linear4(x)
        x = self.LR(x)  
        x = x.reshape(input.size(0),256,4,4)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.conv2d(x)
        x = self.LR(x)
        
        return x

###############################################################################
class DCGAN_D_nobn(nn.Module):
    def __init__(self, isize, nz, nc, ndf, ngpu, n_extra_layers=0):
        super(DCGAN_D_nobn, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential()
        # input is nc x isize x isize
        # input is nc x isize x isize
        main.add_module('initial:conv:{0}-{1}'.format(nc, ndf),
                        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        main.add_module('initial:relu:{0}'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = isize / 2, ndf

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}:{1}:conv'.format(t, cndf),
                            nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}:{1}:relu'.format(t, cndf),
                            nn.LeakyReLU(0.2, inplace=True))

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid:{0}-{1}:conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid:{0}:relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        # state size. K x 4 x 4
        main.add_module('final:{0}-{1}:conv'.format(cndf, 1),
                        nn.Conv2d(cndf, 1, 4, 1, 0, bias=False))
        self.main = main


    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else: 
            output = self.main(input)
            
        output = output.mean(0)
        return output.view(1)

class DCGAN_G_nobn(nn.Module):
    def __init__(self, isize, nz, nc, ngf, ngpu, n_extra_layers=0):
        super(DCGAN_G_nobn, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf//2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential()
        main.add_module('initial:{0}-{1}:convt'.format(nz, cngf),
                        nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False))
        main.add_module('initial:{0}:relu'.format(cngf),
                        nn.ReLU(True))

        csize, cndf = 4, cngf
        while csize < isize//2:
            main.add_module('pyramid:{0}-{1}:convt'.format(cngf, cngf//2),
                            nn.ConvTranspose2d(cngf, cngf//2, 4, 2, 1, bias=False))
            main.add_module('pyramid:{0}:relu'.format(cngf//2),
                            nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}:{1}:conv'.format(t, cngf),
                            nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}:{1}:relu'.format(t, cngf),
                            nn.ReLU(True))

        main.add_module('final:{0}-{1}:convt'.format(cngf, nc),
                        nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        main.add_module('final:{0}:tanh'.format(nc),
                        nn.Softmax())#Tanh())
        self.main = main

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input,  range(self.ngpu))
        else: 
            output = self.main(input)
        return output 