import torch.nn as nn
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.datasets as datasets

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device is", device, torch.cuda.is_available())
def expanded_sigmoid(x):
    absv= torch.abs(x)
    sign = torch.div(x,torch.add(absv,0.000001))
    ret = torch.log(torch.add(absv, 1))
    retu = torch.mul(ret,sign)
    return(retu)

def t_sin(x):
    retu = torch.sin(x)
    return(retu)


class Generator(nn.Module):
    
    def __init__(self,nz,out_length,features):
        
        super(Generator, self).__init__()
        self.nz = nz
        self.batch_len = out_length
        self.features = features
        self.second_hard_t = nn.Hardtanh(0,1)
        self.multihead_attn = nn.MultiheadAttention(1,1, batch_first=True)
        self.orig_comp = nn.Sequential(
            nn.Linear(self.nz, 1000),
            nn.SELU(),
            nn.Linear(1000,2500),
            nn.SELU(),
            nn.Linear(2500,self.features*self.batch_len)
        )
        self.conv_out = nn.Sequential(
            nn.ConvTranspose2d(1, 2, (4), stride=4),
            nn.SELU(),
            nn.ConvTranspose2d(2, 2, (4), stride=(4)),
            nn.SELU(),
            nn.Conv2d(2,4,(512 - (self.batch_len + 20),512 - (self.features + 20)),stride=1,padding=(8),padding_mode='circular'),
            nn.SELU(),
            nn.Conv2d(4,16,(16),stride=1,),
            nn.SELU(),
            nn.Conv2d(16,8,(16),stride=1,),
            nn.Conv2d(8,1,(8),stride=1,),
            
            )
        
    def forward(self, input):
        out = self.orig_comp(input)
        out = torch.sin(out)
        out = torch.add(out,1)
        out = torch.divide(out,2)
        out = out.reshape(out.shape[0], 1,self.batch_len, self.features)
        return(out)

class Standard_Discriminator(nn.Module):
    def __init__(self,batch_len,features):
        super(Standard_Discriminator, self).__init__()
        self.batch_len = batch_len
        self.features = features
        self.multihead_attn = nn.MultiheadAttention(1,1, batch_first=True)
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1,4,(1,self.features),stride=1,padding=(0,self.features),padding_mode='circular'),
            nn.Conv2d(4,2,(self.batch_len,1),stride=1,padding=(self.batch_len,0),padding_mode='circular'),
            nn.Conv2d(2,1,(self.batch_len,self.features),stride=1,padding=2,padding_mode='circular'),
            nn.Conv2d(1,1,(self.batch_len,self.features),stride=1,padding=2,padding_mode='circular'),
            nn.Flatten(start_dim=1),
            )
        self.lin_layer = nn.Sequential(
            nn.Linear((self.batch_len + self.features), int((self.batch_len + self.features)/10)),
            nn.ReLU(),
            )
        self.lin_layer_final = nn.Sequential(
            nn.Linear(int((self.batch_len + self.features)/10),10),
            nn.ReLU(),
            nn.Linear(10,1)
            )
        self.feature_wise = nn.Sequential(
            nn.Conv2d(1,32,(1),stride=1,padding=(0),padding_mode='circular', bias=True),
            nn.Conv2d(32,128,(1,self.features-2),stride=1,padding=(0),padding_mode='circular'),
            nn.Conv2d(128,1,(1,3),stride=1,padding=(0),dilation=1,padding_mode='circular'),
            nn.Flatten()
            )
        self.batch_wise = nn.Sequential(
            nn.Conv2d(1,128,(1),stride=1,padding=(0),padding_mode='circular', bias=True),
            nn.Conv2d(128,1,(self.batch_len,1),stride=1,padding=(0),padding_mode='circular'),
            nn.Flatten()
            
            )
    def forward(self, input):
        inp = input
        out_features = self.feature_wise(inp)
        new_inp,_ = torch.sort(inp,dim=2)
        out_batches = self.batch_wise(new_inp)
        comb = torch.cat((out_features, out_batches), dim=1)[:,:,None]
        comb = self.lin_layer(comb[:,:,0])[:,:,None]
        comb,_ = self.multihead_attn(comb, comb, comb)
        comb = torch.flatten(comb, start_dim=1)
        out = self.lin_layer_final(comb)
        out = torch.sin(out)
        out = torch.add(out,1)
        out = torch.divide(out,2)
        return(out)
   
class Reidentification_Discriminator(nn.Module):
    def __init__(self,batch_len,features):
        super(Reidentification_Discriminator, self).__init__()
        self.batch_len = batch_len
        self.features = features        
        self.multihead_attn = nn.MultiheadAttention(1,1, batch_first=True)
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1,4,(1,self.features),stride=1,padding=(0,self.features),padding_mode='circular'),
            nn.Conv2d(4,2,(self.batch_len,1),stride=1,padding=(self.batch_len,0),padding_mode='circular'),
            nn.Conv2d(2,1,(self.batch_len,self.features),stride=1,padding=2,padding_mode='circular'),
            nn.Conv2d(1,1,(self.batch_len,self.features),stride=1,padding=2,padding_mode='circular'),
            nn.Flatten(start_dim=1),
            )
        self.lin_layer = nn.Sequential(
            nn.Linear((self.batch_len + self.features), int((self.batch_len + self.features)/10)),
            nn.ReLU(),
            )
        self.lin_layer_final = nn.Sequential(
            nn.Linear(int((self.batch_len + self.features)/10),self.batch_len),
            nn.ReLU(),
            )
        self.feature_wise = nn.Sequential(
            nn.Conv2d(1,32,(1),stride=1,padding=(0),padding_mode='circular', bias=True),
            nn.Conv2d(32,128,(1,self.features-2),stride=1,padding=(0),padding_mode='circular'),
            nn.Conv2d(128,1,(1,3),stride=1,padding=(0),dilation=1,padding_mode='circular'),
            nn.Flatten()
            )
        self.batch_wise = nn.Sequential(
            nn.Conv2d(1,128,(1),stride=1,padding=(0),padding_mode='circular', bias=True),
            nn.Conv2d(128,1,(self.batch_len,1),stride=1,padding=(0),padding_mode='circular'),
            nn.Flatten()
            
            )
    def forward(self, gen):
        inp2 = gen
        out_features_real = self.feature_wise(inp2)
        new_inp_real,_ = torch.sort(inp2,dim=2)
        out_batches_real = self.batch_wise(new_inp_real)
        comb = torch.cat((out_features_real, out_batches_real), dim=1)[:,:,None]
        comb = self.lin_layer(comb[:,:,0])[:,:,None]
        comb,_ = self.multihead_attn(comb, comb, comb)
        comb = torch.flatten(comb, start_dim=1)
        out = self.lin_layer_final(comb)
        return(out) 
    
class SinglePoint_Discriminator(nn.Module):
    def __init__(self,batch_len,features):
        super(SinglePoint_Discriminator, self).__init__()
        self.batch_len = batch_len
        self.features = features
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1,4,(1,self.features),stride=1,padding=(0,self.features),padding_mode='circular'),
            nn.Conv2d(4,2,(self.batch_len,1),stride=1,padding=(self.batch_len,0),padding_mode='circular'),
            nn.Conv2d(2,1,(self.batch_len,self.features),stride=1,padding=2,padding_mode='circular'),
            nn.Conv2d(1,1,(self.batch_len,self.features),stride=1,padding=2,padding_mode='circular'),
            nn.Flatten(start_dim=1),
            )
        self.lin_layer = nn.Sequential(
            nn.Linear(self.batch_len + self.features, self.batch_len),
            )
        self.feature_wise = nn.Sequential(
            nn.Conv2d(1,32,(1),stride=1,padding=(0),padding_mode='circular', bias=True),
            nn.Conv2d(32,128,(1,self.features-2),stride=1,padding=(0),padding_mode='circular'),
            nn.Conv2d(128,1,(1,3),stride=1,padding=(0),dilation=1,padding_mode='circular'),
            nn.Flatten()
            )
        self.batch_wise = nn.Sequential(
            nn.Conv2d(1,128,(1),stride=1,padding=(0),padding_mode='circular', bias=True),
            nn.Conv2d(128,1,(self.batch_len,1),stride=1,padding=(0),padding_mode='circular'),
            nn.Flatten()
            
            )
    def forward(self, gen):
        inp1 = gen
        out_features_gen = self.feature_wise(inp1)
        new_inp_gen,_ = torch.sort(inp1,dim=2)
        out_batches_gen = self.batch_wise(new_inp_gen)
        comb = torch.cat((out_features_gen, out_batches_gen), dim=1)
        out = self.lin_layer(comb)
        return(out) 
    
if __name__=='__main__':  
    batch_len = 129
    features = 376
    nz = 20
    batches=8
    gen = Generator(nz,batch_len,features).to(device)
    standard_disc = Reidentification_Discriminator(batch_len,features).to(device)
    
    
    gen_params = sum(p.numel() for p in gen.parameters())
    disc_params = sum(p.numel() for p in standard_disc.parameters())
    
    print("parameters for this pair", gen_params, disc_params)
    
    inp = torch.randn(batches,nz)[:,None, None,:].to(device)
    print("Generator input:", inp.shape)
    g_out = gen(inp)
    real = torch.rand((g_out.shape)).to(device)
    print("Output from Generator", g_out.shape)
    d_out = standard_disc(g_out)
    print("Discriminator shape is", d_out.shape)
    
