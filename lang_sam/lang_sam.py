import os

import groundingdino.datasets.transforms as T
import numpy as np
import torch
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.inference import predict
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from huggingface_hub import hf_hub_download
from segment_anything import sam_model_registry
from segment_anything import SamPredictor

SAM_MODELS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
}

CACHE_PATH = os.environ.get("TORCH_HOME", os.path.expanduser("~/.cache/torch/hub/checkpoints"))


def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file)
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print(f"Model loaded from {cache_file} \n => {log}")
    model.eval()
    return model


def transform_image(image) -> torch.Tensor:
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    image_transformed, _ = transform(image, None)
    return image_transformed


class LangSAM():

    def __init__(self, sam_type="vit_h", ckpt_path=None, return_prompts: bool = False):
        self.sam_type = sam_type
        self.return_prompts = return_prompts
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.build_groundingdino()
        self.build_owlv2()
        self.build_sam(ckpt_path)
        

    def build_sam(self, ckpt_path):
        if self.sam_type is None or ckpt_path is None:
            if self.sam_type is None:
                print("No sam type indicated. Using vit_h by default.")
                self.sam_type = "vit_h"
            checkpoint_url = SAM_MODELS[self.sam_type]
            try:
                sam = sam_model_registry[self.sam_type]()
                state_dict = torch.hub.load_state_dict_from_url(checkpoint_url)
                sam.load_state_dict(state_dict, strict=True)
            except:
                raise ValueError(f"Problem loading SAM please make sure you have the right model type: {self.sam_type} \
                    and a working checkpoint: {checkpoint_url}. Recommend deleting the checkpoint and \
                    re-downloading it.")
            sam.to(device=self.device)
            self.sam = SamPredictor(sam)
        else:
            try:
                sam = sam_model_registry[self.sam_type](ckpt_path)
            except:
                raise ValueError(f"Problem loading SAM. Your model type: {self.sam_type} \
                should match your checkpoint path: {ckpt_path}. Recommend calling LangSAM \
                using matching model type AND checkpoint path")
            sam.to(device=self.device)
            self.sam = SamPredictor(sam)

    def build_groundingdino(self):
        ckpt_repo_id = "ShilongLiu/GroundingDINO"
        ckpt_filename = "groundingdino_swinb_cogcoor.pth"
        ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
        self.groundingdino = load_model_hf(ckpt_repo_id, ckpt_filename, ckpt_config_filename)

    def build_owlv2(self):
        ckpt_repo_id = "google/owlv2-large-patch14-ensemble"  # OWL-ViT의 Hugging Face repo ID
        # ckpt_filename = "pytorch_model.bin"  # OWL-ViT 모델 가중치 파일 이름
        # ckpt_config_filename = "config.json"  # OWL-ViT 모델 설정 파일 이름

        # OWL-ViT 모델과 프로세서 로드
        self.owlv2_processor = Owlv2Processor.from_pretrained(ckpt_repo_id)
        self.owlv2_model = Owlv2ForObjectDetection.from_pretrained(ckpt_repo_id)
        self.owlv2_model.to(self.device)
        print(f"Model loaded done.")

    def predict_dino(self, image_pil, text_prompt, box_threshold, text_threshold):
        image_trans = transform_image(image_pil)
        boxes, logits, phrases = predict(model=self.groundingdino,
                                         image=image_trans,
                                         caption=text_prompt,
                                         box_threshold=box_threshold,
                                         text_threshold=text_threshold,
                                         remove_combined=self.return_prompts,
                                         device=self.device)
        W, H = image_pil.size
        boxes = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

        return boxes, logits, phrases

    def predict_owlv2(self, image_pil, image_path, text_prompt, box_threshold=0.3, text_threshold=0.25):
          imgs = ImageProcessor(image_path, False).img_split()
          all_boxes = []
          all_scores = []
          all_labels = []

          for img in imgs:
              # 이미지와 텍스트를 OWL-ViT v2 모델에 입력으로 줍니다.
              inputs = self.owlv2_processor(text=text_prompt, images=img, return_tensors="pt").to(self.device)
              inputs = {k: v.to(self.device) for k, v in inputs.items()}
              # OWL-ViT v2로 추론 수행
              with torch.no_grad():
                  outputs = self.owlv2_model(**inputs)

              # 결과에서 상자(boxes) 및 점수(scores) 추출
              target_sizes = torch.Tensor([img.size[::-1]]).to(self.device)
              results = self.owlv2_processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=box_threshold)

              # 추출된 박스, 점수, 레이블을 각각 리스트에 추가
              boxes = results[0]["boxes"]
              scores = results[0]["scores"]
              labels = results[0]["labels"]

              all_boxes.append(boxes)
              all_scores.append(scores)
              all_labels.append(labels)

          # 이미지와 예측 결과를 병합합니다.
          boxes, scores, labels = ImageProcessor(image_path).img_concat(imgs, all_boxes, all_scores, all_labels)

          return boxes, scores, labels

    def predict_sam(self, image_pil, boxes):
        image_array = np.asarray(image_pil)
        self.sam.set_image(image_array)
        transformed_boxes = self.sam.transform.apply_boxes_torch(boxes, image_array.shape[:2])
        masks, _, _ = self.sam.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(self.sam.device),
            multimask_output=False,
        )
        return masks.cpu()

    def predict(self, image_pil, text_prompt, box_threshold=0.3, text_threshold=0.25):
        boxes, logits, phrases = self.predict_dino(image_pil, text_prompt, box_threshold, text_threshold)  ## can modify to other model
        masks = torch.tensor([])
        if len(boxes) > 0:
            masks = self.predict_sam(image_pil, boxes)
            masks = masks.squeeze(1)
        return masks, boxes, phrases, logits
