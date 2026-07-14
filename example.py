# vis_mask_queries_stage1_vs_mgface.py

import os
import argparse

import torch
import torch.nn as nn
import torch.multiprocessing
from tqdm import trange
from PIL import Image, ImageDraw, ImageFont

from face_models.resnet import *
from face_models.facenet import InceptionResnetV1Mask

from utils.emd import patch_similarity
from utils.extract_mgface import extract_embedding
from data_loader.facedata_loader import get_face_dataloader


torch.multiprocessing.set_sharing_strategy("file_system")


parser = argparse.ArgumentParser(
    "Visualize Stage-1 vs MGFace Top-5 for masked queries only"
)

parser.add_argument("-fm", type=str, default="arcface", choices=["arcface", "facenet"])
parser.add_argument("-l", type=int, default=4, help="patch grid level")
parser.add_argument("-d", type=str, default="lfw", help="dataset")
parser.add_argument("-data_folder", type=str, default="data")

parser.add_argument("--topk", type=int, default=100)
parser.add_argument("--num_vis", type=int, default=50)
parser.add_argument("--start_idx", type=int, default=0)
parser.add_argument("--vis_dir", type=str, default="vis_stage1_vs_mgface")
parser.add_argument("--vis_image_size", type=int, default=128)

parser.add_argument("--vis_orig_folder", type=str, default="lfw-align-128-mgface")
parser.add_argument("--vis_masked_folder", type=str, default="lfw-align-128-masked-mgface")

args = parser.parse_args()


def tensor_to_int(x):
    if torch.is_tensor(x):
        return int(x.detach().cpu().item())
    return int(x)


def name_to_relpath(name):
    name = str(name)
    parts = name.split("_")

    if len(parts) < 3:
        raise ValueError(f"Invalid name format: {name}")

    identity = "_".join(parts[1:-1])
    image_idx = parts[-1]

    return os.path.join(identity, f"{identity}_{image_idx}.jpg")


def build_image_path(name, data_dir, image_folder):
    return os.path.join(data_dir, image_folder, name_to_relpath(name))


def load_image(path, image_size):
    img = Image.open(path).convert("RGB")
    img = img.resize(image_size)
    return img


def draw_text(draw, xy, text, fill=(0, 0, 0)):
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    draw.text(xy, text, fill=fill, font=font)


def save_top5_compare(
    query_path,
    stage1_paths,
    stage1_labels,
    mgface_paths,
    mgface_labels,
    query_label,
    mask_pred,
    save_path,
    image_size=(128, 128),
):
    """
    Row 1: Stage 1 global cosine only
    Row 2: MGFace with mask-gated routing

    If mask_pred == 1:
        MGFace = Stage 2 patch re-ranking
    If mask_pred == 0:
        MGFace = Stage 1 global ranking
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cell_w, cell_h = image_size
    gap = 8
    text_h = 42
    n_cols = 6
    n_rows = 2

    canvas_w = n_cols * cell_w + (n_cols + 1) * gap
    canvas_h = n_rows * (cell_h + text_h) + (n_rows + 1) * gap

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    if int(mask_pred) == 1:
        mgface_title = "MGFace | Mask -> Stage 2"
    else:
        mgface_title = "MGFace | Unmask -> Stage 1"

    rows = [
        ("Stage 1", stage1_paths, stage1_labels),
        (mgface_title, mgface_paths, mgface_labels),
    ]

    for row_idx, (row_title, top5_paths, top5_labels) in enumerate(rows):
        y = gap + row_idx * (cell_h + text_h + gap)

        x = gap
        query_img = load_image(query_path, image_size)
        canvas.paste(query_img, (x, y))

        draw.rectangle(
            [x, y, x + cell_w - 1, y + cell_h - 1],
            outline=(0, 0, 0),
            width=2,
        )

        draw_text(
            draw,
            (x, y + cell_h + 2),
            f"{row_title}\nQID: {query_label}",
            fill=(0, 0, 0),
        )

        for rank, (path, label) in enumerate(zip(top5_paths, top5_labels), start=1):
            x = gap + rank * (cell_w + gap)

            try:
                img = load_image(path, image_size)
            except Exception as e:
                print(f"[Warning] Cannot load image: {path}. Error: {e}")
                img = Image.new("RGB", image_size, color=(230, 230, 230))

            canvas.paste(img, (x, y))

            correct = int(label) == int(query_label)
            color = (0, 180, 0) if correct else (220, 0, 0)
            mark = "OK" if correct else "NO"

            draw.rectangle(
                [x, y, x + cell_w - 1, y + cell_h - 1],
                outline=color,
                width=4,
            )

            draw_text(
                draw,
                (x, y + cell_h + 2),
                f"Top-{rank} {mark}\nID: {label}",
                fill=color,
            )

    canvas.save(save_path)


def load_checkpoint_to_model(model, model_path, fm):
    print("Load checkpoint:", model_path)

    state = torch.load(model_path, map_location="cpu")

    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    if fm == "arcface":
        from collections import OrderedDict

        new_state = OrderedDict()
        for k, v in state.items():
            name = k[7:] if k.startswith("module.") else k
            new_state[name] = v

        model.load_state_dict(new_state, strict=True)

    else:
        model.load_state_dict(state, strict=True)

    return model


def build_model(fm):
    if fm == "arcface":
        model = resnet_face18_mask(False, use_reduce_pool=False)
        model_path = "checkpoints/arcface_w_maskhead.pth"

    elif fm == "facenet":
        model = InceptionResnetV1Mask()
        model_path = "checkpoints/facenet_w_maskhead.pth"

    else:
        raise ValueError(f"Unsupported model: {fm}")

    model = load_checkpoint_to_model(model, model_path, fm)
    model.eval()
    model = nn.DataParallel(model).cuda()

    return model


def main():
    print("args =", args)

    data_dir = os.path.join(os.getcwd(), args.data_folder)

    if args.fm == "arcface":
        datasets = {
            "lfw": ["lfw128", "lfw128_masked_test"]
        }

    elif args.fm == "facenet":
        datasets = {
            "lfw": ["lfw128", "lfw128_masked_test"]
        }

    else:
        raise ValueError(f"Unsupported model: {args.fm}")

    gallery_data = datasets[args.d][0]
    masked_query_data = datasets[args.d][1]

    print("Gallery:", gallery_data)
    print("Masked query:", masked_query_data)

    _, data_loaders = get_face_dataloader(
        16,
        data_dir=data_dir,
        fm=args.fm,
        num_workers=16,
    )

    model = build_model(args.fm)

    print("Extract gallery features...")
    feature_gallery, feature_patch_gallery, labels_gallery, mask_gallery, names_gallery = extract_embedding(
        data_loaders,
        gallery_data,
        model,
        fm=args.fm,
        level=args.l,
    )

    print("Extract masked query features...")
    feature_query, feature_patch_query, labels_query, mask_query, names_query = extract_embedding(
        data_loaders,
        masked_query_data,
        model,
        fm=args.fm,
        level=args.l,
    )

    os.makedirs(args.vis_dir, exist_ok=True)

    end_idx = min(len(names_query), args.start_idx + args.num_vis)
    saved = 0

    for idx in trange(args.start_idx, end_idx):
        query_label = tensor_to_int(labels_query[idx])
        mask_pred = tensor_to_int(mask_query[idx])

        # =========================================================
        # Stage 1: global cosine similarity only
        # =========================================================
        anchor = feature_query[idx]
        sim = feature_gallery @ anchor

        same_mask = torch.tensor(
            [g_name == names_query[idx] for g_name in names_gallery],
            device=feature_gallery.device,
            dtype=torch.bool,
        )

        sim = sim.clone()
        sim[same_mask] = -100.0

        approx_tops = torch.argsort(sim, descending=True)
        stage1_top5 = approx_tops[:5].detach().cpu().tolist()

        # =========================================================
        # MGFace routing:
        # If mask classifier predicts masked -> Stage 2
        # If mask classifier predicts unmasked -> keep Stage 1
        # =========================================================
        if mask_pred == 1:
            topk_eff = min(args.topk, approx_tops.numel())
            top_inds = approx_tops[:topk_eff]

            anchor_patch = feature_patch_query[idx]
            gallery_patch_topk = feature_patch_gallery[top_inds]

            sim_patch = patch_similarity(anchor_patch, gallery_patch_topk)

            top_same = same_mask[top_inds]
            sim_patch = sim_patch.clone()
            sim_patch[top_same] = -100.0

            rank_in_topk = torch.argsort(sim_patch, descending=True)
            reranked_topk = top_inds[rank_in_topk]

            used = torch.zeros(
                feature_gallery.size(0),
                device=feature_gallery.device,
                dtype=torch.bool,
            )
            used[reranked_topk] = True

            rest_tops = approx_tops[~used[approx_tops]]
            mgface_tops = torch.cat([reranked_topk, rest_tops], dim=0)

        else:
            # Classifier predicts unmasked, so MGFace does not use Stage 2.
            mgface_tops = approx_tops

        mgface_top5 = mgface_tops[:5].detach().cpu().tolist()

        # =========================================================
        # Build image paths and labels
        # =========================================================
        query_path = build_image_path(
            names_query[idx],
            data_dir,
            args.vis_masked_folder,
        )

        stage1_paths = [
            build_image_path(names_gallery[i], data_dir, args.vis_orig_folder)
            for i in stage1_top5
        ]

        mgface_paths = [
            build_image_path(names_gallery[i], data_dir, args.vis_orig_folder)
            for i in mgface_top5
        ]

        stage1_labels = [
            tensor_to_int(labels_gallery[i])
            for i in stage1_top5
        ]

        mgface_labels = [
            tensor_to_int(labels_gallery[i])
            for i in mgface_top5
        ]

        save_path = os.path.join(
            args.vis_dir,
            f"masked_query_{idx:05d}_predmask_{mask_pred}_stage1_vs_mgface.jpg",
        )


        save_top5_compare(
            query_path=query_path,
            stage1_paths=stage1_paths,
            stage1_labels=stage1_labels,
            mgface_paths=mgface_paths,
            mgface_labels=mgface_labels,
            query_label=query_label,
            mask_pred=mask_pred,
            save_path=save_path,
            image_size=(args.vis_image_size, args.vis_image_size),
        )
        saved += 1


    print(f"Done. Saved {saved} visualizations to: {args.vis_dir}")


if __name__ == "__main__":
    main()