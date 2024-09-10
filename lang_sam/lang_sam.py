import os
import groundingdino.datasets.transforms as T
import numpy as np
import torch
import torchvision.transforms as transforms
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.inference import predict
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from huggingface_hub import hf_hub_download
from segment_anything import sam_model_registry
from segment_anything import SamPredictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from split_n_concat import ImageProcessor
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from .gdino import GroundingDINOAPIWrapper, visualize

SAM_MODELS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
}

SAM2_MODELS = {
    "hiera_tiny" : "facebook/sam2-hiera-tiny",
    "hiera_small" : "facebook/sam2-hiera-small",
    "hiera_base_plus" : "facebook/sam2-hiera-base-plus", 
    "hiera_large" : "facebook/sam2-hiera-large"
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

    def __init__(self, sam_type="vit_h", sam2_type="hiera_large", ckpt_path=None, return_prompts: bool = False, split=True):
        self.sam_type = sam_type
        self.sam2_type = sam2_type
        self.return_prompts = return_prompts
        self.split = split
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.build_groundingdino()
        self.build_sam(ckpt_path)
        self.build_owlv2() 
        self.gdino = GroundingDINOAPIWrapper('key')
        self.build_sam2()

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
    
    def build_sam2(self):
        self.sam2 = SAM2ImagePredictor.from_pretrained(SAM2_MODELS[self.sam2_type])
        print("SAM2 loaded done.")

    def build_groundingdino(self):
        ckpt_repo_id = "ShilongLiu/GroundingDINO"
        ckpt_filename = "groundingdino_swinb_cogcoor.pth"
        ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
        self.groundingdino = load_model_hf(ckpt_repo_id, ckpt_filename, ckpt_config_filename)
        print("Grounding DINO loaded done.")

    def build_owlv2(self):
        ckpt_repo_id = "google/owlv2-large-patch14-ensemble"  # OWLv2 Hugging Face repo ID

        # OWLv2 load
        self.owlv2_processor = Owlv2Processor.from_pretrained(ckpt_repo_id)
        self.owlv2_model = Owlv2ForObjectDetection.from_pretrained(ckpt_repo_id)
        self.owlv2_model.to(self.device)
        print("Owlv2 Model loaded done.")


    def predict_dino(self, image_pil, image_path, text_prompt, box_threshold, text_threshold):
        if self.split:
            imgs = ImageProcessor(image_path, True).img_split()
            all_boxes = []
            all_logits = []
            all_phrases = []
            
            for img in imgs:
                # image preprocessing -> split 8 pieces
                image_trans = transform_image(img)
                boxes, logits, phrases = predict(model=self.groundingdino,
                                                 image=image_trans,
                                                 caption=text_prompt,
                                                 box_threshold=box_threshold,
                                                 text_threshold=text_threshold,
                                                 remove_combined=self.return_prompts,
                                                 device=self.device)
    
                # save predict results
                all_boxes.append(boxes)
                all_logits.append(logits)
                all_phrases.append(phrases)
    
            # concat image
            boxes, logits, phrases = ImageProcessor(image_path).img_concat(imgs, all_boxes, all_logits, all_phrases)

        else:
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
        print("DINO Work done.")
        return boxes, logits, phrases

    def predict_dino15(self, image_pil, image_path, text_prompt, box_threshold, text_threshold, return_mask=False):
        if self.split:
            imgs = ImageProcessor(image_path).img_split()
          
            all_boxes = []
            all_logits = []
            all_phrases = []
    
            for img in imgs:
                # PIL 이미지를 Numpy 배열로 변환
                img_array = np.array(img)
    
                # API 호출에 사용할 프롬프트 설정
                prompts = dict(image=img_array, prompt=text_prompt)  # 변환된 numpy 배열 사용
    
                # Grounding DINO API 호출
                results = self.gdino.inference(prompts, return_mask=return_mask)
    
                # 결과에서 필요한 데이터를 추출
                boxes = torch.tensor(results['boxes'])
                logits = torch.tensor(results['scores'])  # logits 대신 scores 사용
                phrases = results.get('labels', [])  # labels이 있는 경우에만 가져오기
    
                # 예측 결과 저장
                all_boxes.append(boxes)
                all_logits.append(logits)
                all_phrases.append(phrases)
    
            # 이미지 프로세서의 img_concat 메서드를 사용하여 결과 결합
            boxes, logits, phrases = ImageProcessor(image_path).img_concat(imgs, all_boxes, all_logits, all_phrases)
        else:
            # API 호출에 사용할 프롬프트 설정
            prompts = dict(image=image_path, prompt=text_prompt)
              
            # Grounding DINO API 호출
            results = self.gdino.inference(prompts, return_mask=return_mask)
    
            # 결과에서 필요한 데이터를 추출
            boxes = torch.tensor(results['boxes'])
            logits = torch.tensor(results['scores'])  # logits 대신 scores 사용
            phrases = results.get('labels', [])  # labels이 있는 경우에만 가져오기
        print("DINO 1.5 Work done.")
        return boxes, logits, phrases


    def predict_owlv2(self, image_pil, image_path, text_prompt, box_threshold=0.3, text_threshold=0.25):
        inputs = None
        try:
          if self.split:  
              imgs = ImageProcessor(image_path, False).img_split()
              all_boxes = []
              all_scores = []
              all_labels = []
        
              for img in imgs:
                  # input image & text to OWLv2
                  inputs = self.owlv2_processor(text=text_prompt, images=img, return_tensors="pt").to(self.device)
                  print(f"inputs before device: {inputs}")  # 로그 추가
                  inputs = {k: v.to(self.device) for k, v in inputs.items()}
                  print(f"inputs after device: {inputs}")  # 로그 추가
                  # OWLv2 inference
                  with torch.no_grad():
                      outputs = self.owlv2_model(**inputs)
        
                  # extract boxes & scores
                  target_sizes = torch.Tensor([img.size[::-1]]).to(self.device)
                  results = self.owlv2_processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=box_threshold)
                      
                  boxes = results[0]["boxes"]
                  scores = results[0]["scores"]
                  labels = results[0]["labels"]
        
                  all_boxes.append(boxes)
                  all_scores.append(scores)
                  all_labels.append(labels)
        
              # concat image
              boxes, scores, labels = ImageProcessor(image_path).img_concat(imgs, all_boxes, all_scores, all_labels)
          else:
              # input image & text to OWLv2
              inputs = self.owlv2_processor(text=text_prompt, images=image_pil, return_tensors="pt")
              # print(f"inputs before device: {inputs}")  # 로그 추가
              inputs = {k: v.to(self.device) for k, v in inputs.items()}
              # print(f"inputs after device: {inputs}")  # 로그 추가
                
              # OWLv2 inference
              with torch.no_grad():
                  outputs = self.owlv2_model(**inputs)
        
              # extract boxes & scores
              target_sizes = torch.Tensor([image_pil.size[::-1]]).to(self.device)
              results = self.owlv2_processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=box_threshold)
                      
              boxes = results[0]["boxes"]
              scores = results[0]["scores"]
              labels = results[0]["labels"]
          print("Owlv2 Work done.")
          return boxes, scores, labels
        except Exception as e:
          print(f"Error occurred: {str(e)}")

    def predict_sam(self, image_pil, boxes):
        image_array = np.asarray(image_pil)
        self.sam.set_image(image_array)
        # boxes = boxes.cpu().numpy()
        boxes = boxes.to(self.device)
        transformed_boxes = self.sam.transform.apply_boxes_torch(boxes, image_array.shape[:2])
        masks, _, _ = self.sam.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(self.sam.device),
            multimask_output=False,
        )
        print("SAM Work done.")
        return masks.cpu()
      
    def predict_sam2(self, image_pil, boxes):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
          self.sam2.set_image(image_pil)  
          boxes = boxes.to(self.device)
          masks, _, _ = self.sam2.predict(box=boxes)
          print("SAM2 Work done.")
          return masks

    def predict(self, image_pil, image_pil2, image_path, text_prompt, box_threshold=0.3, text_threshold=0.25):
        boxes, logits, phrases = self.predict_owlv2(image_pil, image_path, text_prompt, box_threshold, text_threshold)  # can change other models
        print("Object detection is done.")
        masks = torch.tensor([])
        if len(boxes) > 0:
            masks = self.predict_sam2(image_pil2, boxes)  ## if you want to use sam2, you should change predict_sam2()
            masks = masks.squeeze(1)
        print("Segmentation is done.")
        return masks, boxes, phrases, logits
