import os
import argparse
import torch
from tqdm import trange


from face_models.resnet import *
from face_models.net_cos import *
from face_models.mobileface import *
from face_models.facenet import InceptionResnetV1Mask
from utils.emd import patch_similarity
from utils.metrics import get_metrics_rank
from utils.extract_mgface import extract_embedding
from data_loader.facedata_loader import get_face_dataloader



parser = argparse.ArgumentParser( description="Test MGFace" )

parser.add_argument("-fm", type=str, default="sphereface",help="face model",)
parser.add_argument("-l", type=int, default=4,help="level of grid size",)
parser.add_argument('-mask', action='store_true', help="If True, masked on",)
parser.add_argument("-data_folder", type=str, default="data", help="dataset dir: data_small or data",)
parser.add_argument("-d", type=str, default="lfw", help="dataset",)

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
    _, data_loaders = get_face_dataloader(batch_size = 16, data_dir=data_dir, fm=args.fm, num_workers=16)
    if args.fm == 'arcface':
        model_path = 'checkpoints/arcface_w_maskhead.pth'
        model = resnet_face18_mask(False, use_reduce_pool=False)
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
    elif args.fm == 'facenet':
        model_path = 'checkpoints/facenet_w_maskhead.pth'
        model = InceptionResnetV1Mask()
        model.load_state_dict(torch.load(model_path))
        
    model.eval()
    model = model.cuda()


    feature_mask, feature_patch_mask, labels_mask, mask_mask, names_mask = extract_embedding(data_loaders, query_data, model, fm=args.fm, level=args.l)
    feature_orig, feature_patch_orig, labels_orig, mask_orig, names_orig = extract_embedding(data_loaders, gallery_data, model, fm=args.fm, level=args.l)

    feature_patch_query = torch.cat([feature_patch_orig,feature_patch_mask], dim=0)
    feature_query = torch.cat([feature_orig,feature_mask], dim=0)
    mask_query = torch.cat([mask_orig,mask_mask], dim=0)
    labels_query = torch.cat([labels_orig,labels_mask], dim=0)
    names_query = names_orig + names_mask    

    feature_gallery, feature_patch_gallery, labels_gallery, _, names_gallery  = feature_orig, feature_patch_orig, labels_orig, mask_orig, names_orig
    

    overall_r1 = 0.0
    overall_rp = 0.0
    overall_mapr = 0.0

    N = feature_query.size(0)
    topk = 100

    # precompute same-name mask
    same_name_masks = []
    for q_name in names_query:
        same_mask = torch.tensor(
            [g_name == q_name for g_name in names_gallery],
            device=feature_gallery.device,
            dtype=torch.bool
        )
        same_name_masks.append(same_mask)

    for idx in trange(N):
        is_mask = int(mask_query[idx]) == 1

        # stage 1: global cosine
        anchor = feature_query[idx]                  # (C,)
        sim = feature_gallery @ anchor               # (Ng,)

        sim[same_name_masks[idx]] = -100.0

        if not is_mask:
            final_tops = torch.argsort(sim, descending=True)
        else:
            # Re-rank only top-k for query mask
            approx_tops = torch.argsort(sim, descending=True)
            top_inds = approx_tops[:topk]

            anchor_patch = feature_patch_query[idx]              # (C, R)
            gallery_patch_topk = feature_patch_gallery[top_inds] # (K, C, R)

            sim_patch = patch_similarity(anchor_patch, gallery_patch_topk)  # (K,)

            # Remove the same name in top-k
            top_same = same_name_masks[idx][top_inds]
            sim_patch[top_same] = -100.0

            rank_in_topk = torch.argsort(sim_patch, descending=True)
            reranked_topk = top_inds[rank_in_topk]

            final_tops = torch.cat([reranked_topk, approx_tops[topk:]], dim=0)

        r1, rp, mapr = get_metrics_rank(final_tops.data.cpu(), labels_query[idx], labels_gallery)
        overall_r1 += r1
        overall_rp += rp
        overall_mapr += mapr

    overall_r1 /= float(N / 100)
    overall_rp /= float(N / 100)
    overall_mapr /= float(N / 100)

    print('[Mask-Aware] acc=%f, RP=%f, MAP@R=%f' % (overall_r1, overall_rp, overall_mapr))

if __name__ == '__main__':
    main()


