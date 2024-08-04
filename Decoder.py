from decoders import *
def convblock(in_, out_, ks, st, pad):
    return nn.Sequential(
        nn.Conv2d(in_, out_, ks, st, pad),
        nn.BatchNorm2d(out_),
        nn.ReLU()
    )
def convblock2(in_, out_, ks, st, pad):
    return nn.Sequential(
        nn.Conv2d(in_, out_, ks, st, pad),
        nn.BatchNorm2d(out_),
        nn.LeakyReLU(negative_slope=0.2, inplace=True)
    )

def Aux_decoders():
    vat_decoder = [VATDecoder(4, 64, xi=1e-6, eps=2.0) for _ in range(1)]  # 1e-6 2.0   2
    drop_decoder = [DropOutDecoder(4, 96, drop_rate=0.5, spatial_dropout=True) for _ in range(2)]  ## 0.5   True   6
    cut_decoder = [CutOutDecoder(4, 96, erase=0.4) for _ in range(2)]  # # 0.4  6
    context_m_decoder = [ContextMaskingDecoder(4, 96) for _ in range(1)]  # 2
    object_masking = [ObjectMaskingDecoder(4, 64) for _ in range(2)]  # 2
    feature_drop = [FeatureDropDecoder(4, 64) for _ in range(1)]  # 6
    feature_noise = [FeatureNoiseDecoder(4, 64, uniform_range=0.3) for _ in range(2)]  # # 0.3  6

    # Aux_decoders = nn.ModuleList([*vat_decoder, *drop_decoder, *cut_decoder,
    #                                *context_m_decoder, *object_masking, *feature_drop, *feature_noise])
    Aux_decoders = nn.ModuleList([*object_masking])
    return Aux_decoders

class Main_Decoder(nn.Module):
    def __init__(self):
        super(Main_Decoder, self).__init__()
        self.score1 = nn.Conv2d(64, 1, 1, 1, 0)
        self.score2 = nn.Conv2d(64, 1, 1, 1, 0)
        self.score = nn.Conv2d(2, 1, 1, 1, 0)
        self.sig = nn.Sigmoid()

        self.score = nn.Conv2d(2, 1, 1, 1, 0)
    def forward(self, x1, x2):
        score1 = self.score1(F.interpolate(x1, (384, 384), mode='bilinear', align_corners=True))
        score2 = self.score2(F.interpolate(x2, (384, 384), mode='bilinear', align_corners=True))

        score = self.score(torch.cat((score1 + torch.mul(score1, self.sig(score2)),
                                      score2 + torch.mul(score2, self.sig(score1))), 1))

        return score1, score2, score, self.sig(score)
