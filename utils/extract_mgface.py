import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


def extract_embedding(data_loaders, dataset, model, fm='arcface', level=4):
    model.eval()
    dataloader = data_loaders[dataset]
    labels = []

    with torch.no_grad():
        feature = []
        feature_patch = []
        mask_prediction = []
        names = []

        final_iter = tqdm(dataloader, desc='Embedding Data...')

        for idx, inp in enumerate(final_iter):
            input_img, target, name = inp[0], inp[1], inp[2]
            out = model(input_img.cuda())

            # global embedding cho stage 1
            fea = out['fea']
            
            # ===== mask prediction =====
            logits_mask = out['mask'] 
            # print(logits_mask)
            prob = torch.softmax(logits_mask, dim=1)
            mask_label = torch.argmax(prob, dim=1)

            
            mask_prediction.append(mask_label)

            # print(name)
            names.append(name)

            if fm == 'arcface':
                if   level == 4:
                    aux_f = out['embedding_44']   # (B, C, H, W)
                elif level == 8:
                    aux_f = out['embedding_88']
                elif level == 16:
                    aux_f = out['embedding_16']
                
            elif fm in ['sphereface', 'cosface', 'facenet']:
                aux_f = out['embedding']
            elif fm == 'mobileface':
                aux_f = out['7x7']
            else:
                raise ValueError(f'Unsupported fm: {fm}')

            # ===== crop upper-half =====
            # print(aux_f.shape)
            H = aux_f.size(2)
            upper_h = (H + 1) // 2
            upper_aux_f = aux_f[:, :, :upper_h, :]   # (B, C, upper_h, W)
            # print(upper_aux_f.shape)


            feature_patch.append(upper_aux_f.data)
            feature.append(fea.data)
            labels.append(target)



        labels = torch.cat(labels, dim=0).squeeze(-1)

        feature_patch = torch.cat(feature_patch, dim=0)   # (N, C, H_up, W)
        N, C, _, _ = feature_patch.size()
        feature_patch = feature_patch.view(N, C, -1)      # (N, C, R)

        feature = torch.cat(feature, dim=0)  # (N, C_emb)
        mask_prediction = torch.cat(mask_prediction, dim=0).squeeze(-1)

    feature = F.normalize(feature, p=2, dim=1)
    feature_patch = F.normalize(feature_patch, p=2, dim=1)

    names = [name for subpool in names for name in subpool]

    return feature, feature_patch, labels, mask_prediction, names