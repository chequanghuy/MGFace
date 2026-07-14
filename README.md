# MGFace: Mask-Gated Face Matching via Conditional Similarity Routing

## Setup
```bash
git clone https://github.com/chequanghuy/MGFace.git
pip install -r requirements.txt
```
## Download

Please download the required files before running the code:

- **LFW dataset**: [Download here](https://drive.google.com/file/d/1RBbKSafWiNltzA9tCc3lV00rufXWcji9/view?usp=drive_link)
- **Pretrained face recognition model with mask classification head**: [Download here](https://drive.google.com/file/d/18dUoafN1RSD49Wq66zdIQryh5Hv-oLZs/view?usp=sharing)
- (Optional) **Pretrained face recognition w/o mask classification head**: [Download here](https://drive.google.com/file/d/1fmuNQZUhlOS-1fS4F4WGyRRdzupKfzpp/view?usp=sharing)

```text
MGFace/
└── data
└── checkpoints
└── pretrained (optional for EMD evaluation)
└── ....
```
## Evaluation

```bash
python mgface.py -fm arcface -mask
python mgface.py -fm facenet -mask
```
## Run examples

```bash
python example.py -fm 'facenet/arcface'
```


## Citation

```
@INPROCEEDINGS{9924897,
  author={Huy Che and Hoang-Minh Trinh and Dinh-Duy Phan and Duc-Lung Vu},
  booktitle={2026 International Conference on Multimedia Analysis and Pattern Recognition (MAPR)}, 
  title={MGFace: Mask-Gated Face Matching via Conditional Similarity Routing}, 
  year={2026}
 }
```
