import os
import argparse
import torch
from tqdm import trange
from PIL import Image

from face_models.mobileface import *

from face_models.resnet import *
from face_models.net_cos import *
from face_models.facenet import InceptionResnetV1
from utils.emd import emd_similarity
from utils.metrics import get_metrics_rank
from utils.extract_features import extract_embedding
from data_loader.facedata_loader import get_face_dataloader


parser = argparse.ArgumentParser(
    description="Test DeepFace-EMD"
)


parser.add_argument("-method", type=str, default="apc",help="Methods: uniform, apc, and sc",)
parser.add_argument("-fm", type=str, default="sphereface",help="face model",)
parser.add_argument("-l", type=int, default=4,help="level of grid size",)
parser.add_argument('-mask', action='store_true', help="If True, masked on",)
parser.add_argument("-d", type=str, default="lfw", help="dataset",)
parser.add_argument("-data_folder", type=str, default="data", help="dataset dir: data_small or data",)

args = parser.parse_args()




def main():
    print("args = {}".format(args))
    data_dir = os.path.join(os.getcwd(), args.data_folder)
        
    print('dataset dir: {}'.format(data_dir))
    if args.fm == 'arcface':
        datasets = {'lfw':['lfw128','lfw128_masked']}
    if args.fm == 'facenet':
        datasets = {'lfw':['lfw128','lfw128_masked']}

    
    if args.mask:
        query_data = datasets[args.d][1]
    else:    
        query_data = datasets[args.d][0]

    gallery_data = datasets[args.d][0] 
    print('query data: {} - gallery: {}'.format(query_data, gallery_data))
    _, data_loaders = get_face_dataloader(16, data_dir=data_dir, fm=args.fm, num_workers=16)



    if args.fm == 'arcface':
        model_path =  'pretrained/resnet18_110.pth'
        print('model : {}'.format(model_path))
        model = resnet_face18(False, use_reduce_pool=False)
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove module.
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    elif args.fm == 'facenet':
        model_path = 'pretrained/20180402-114759-vggface2.pt'
        model = InceptionResnetV1()
        model.load_state_dict(torch.load(model_path))
        
    model.eval()
    model = model.cuda()




    feature_bank_mask, feature_bank_center_mask, avgpool_bank_center_mask, labels_mask, names_mask = extract_embedding(
        data_loaders, query_data, model, fm=args.fm, level=args.l
    )

    feature_bank_orig, feature_bank_center_orig, avgpool_bank_center_orig, labels_orig, names_orig = extract_embedding(
        data_loaders, gallery_data, model, fm=args.fm, level=args.l
    )

    # Query = mask + original
    feature_bank_query = torch.cat([feature_bank_orig, feature_bank_mask], dim=0)
    feature_bank_center_query = torch.cat([feature_bank_center_orig, feature_bank_center_mask], dim=0)
    avgpool_bank_center_query = torch.cat([avgpool_bank_center_orig, avgpool_bank_center_mask], dim=0)
    labels_query = torch.cat([labels_orig, labels_mask], dim=0)
    names_query = names_orig + names_mask


    # Gallery = original
    feature_bank_gallery = feature_bank_orig
    feature_bank_center_gallery = feature_bank_center_orig
    avgpool_bank_center_gallery = avgpool_bank_center_orig
    labels_gallery = labels_orig
    names_gallery = names_orig

    stages = [0, 100]
    overall_r1 = {k: 0.0 for k in stages}
    overall_rp = {k: 0.0 for k in stages}
    overall_mapr = {k: 0.0 for k in stages}

    N, C, _ = feature_bank_query.size()
    max_stage = max(stages)

    # Precompute same-name mask
    same_name_masks = []
    for q_name in names_query:
        same_mask = torch.tensor(
            [g_name == q_name for g_name in names_gallery],
            device=feature_bank_center_gallery.device,
            dtype=torch.bool
        )
        same_name_masks.append(same_mask)

    for idx in trange(N):
        anchor_center = feature_bank_center_query[idx]

        approx_sim, _, _, _ = emd_similarity( None, anchor_center, None, feature_bank_center_gallery, 0 )

        # Remove all gallery images with the same name as the query.
        same_name_mask = same_name_masks[idx]

        approx_sim = approx_sim.clone()
        approx_sim[same_name_mask] = -float("inf")

        approx_tops = torch.argsort(approx_sim, descending=True)

        if max_stage > 0:
            topk_eff = min(max_stage, approx_tops.numel())
            top_inds = approx_tops[:topk_eff]

            anchor = feature_bank_query[idx]
            feature_query = avgpool_bank_center_query[idx]
            feature_gallery = avgpool_bank_center_gallery[top_inds]

            score_top, _, _, _ = emd_similarity( anchor, feature_query, feature_bank_gallery[top_inds], feature_gallery, 1, method=args.method )

            rank_in_tops = torch.argsort(score_top, descending=True)

        for stage in stages:
            if stage == 0 or max_stage == 0:
                final_tops = approx_tops
            else:
                stage_eff = min(stage, top_inds.numel())
                reranked_top = top_inds[rank_in_tops][:stage_eff]
                used = torch.zeros( feature_bank_gallery.size(0), device=feature_bank_gallery.device, dtype=torch.bool )
                used[reranked_top] = True
                rest_tops = approx_tops[~used[approx_tops]]
                final_tops = torch.cat([reranked_top, rest_tops], dim=0)

            r1, rp, mapr = get_metrics_rank( final_tops.data.cpu(), labels_query[idx], labels_gallery )

            overall_r1[stage] += r1
            overall_rp[stage] += rp
            overall_mapr[stage] += mapr


        
    for i, stage in enumerate(stages):
        overall_r1[stage] /= float(N / 100)
        overall_rp[stage] /= float(N / 100)
        overall_mapr[stage] /= float(N / 100)
        print('[stage %d] acc=%f, RP=%f, MAP@R=%f' % (i+1, overall_r1[stage], overall_rp[stage], overall_mapr[stage]))

if __name__ == '__main__':
    main()


