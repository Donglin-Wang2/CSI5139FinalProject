import os
from SinGAN.models import WDiscriminator
from config import get_arguments
import SinGAN.functions as functions
from SinGAN.training import *

parser = get_arguments()
parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
parser.add_argument('--input_name', help='input image name', required=True)
parser.add_argument('--mode', help='task to be done', default='train')
opt = parser.parse_args()
opt = functions.post_config(opt)





# D_state = torch.load('./TrainedModels/balloons/scale_factor=0.750000,alpha=10/0/netD.pth')

# Ds.load_state_dict(D_state)


for scale_num in range(7):
    print(scale_num)
    opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
    opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)
    Ds, Gs = init_models(opt)
    if scale_num != 0:
        Gs.load_state_dict(torch.load('./TrainedModels/balloons/scale_factor=0.750000,alpha=10/%d/netG.pth' % (scale_num-1)))
        Ds.load_state_dict(torch.load('./TrainedModels/balloons/scale_factor=0.750000,alpha=10/%d/netD.pth' % (scale_num-1)))
    