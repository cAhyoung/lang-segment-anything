from PIL import Image
import os
import torch
from groundingdino.util import box_ops

class ImageProcessor:
    def __init__(self, image_path, convert_cxcywh_to_xyxy=False):
        """
        image_path: image file path
        convert_cxcywh_to_xyxy: status to change bbox cxcywh to xyxy or not
        """
        self.image_path = image_path
        self.convert_cxcywh_to_xyxy = convert_cxcywh_to_xyxy

    def img_split(self):
        """
        function to split image 4*2
        """
        split_image = []
        image = Image.open(self.image_path)
        width, height = image.size

        # calculate image size splited 8 pieces
        split_width = width // 4
        split_height = height // 2

        for i in range(2):
            for j in range(4):
                # calculate size of splited image
                left = j * split_width
                top = i * split_height
                right = (j + 1) * split_width
                bottom = (i + 1) * split_height
              
                split_image.append(image.crop((left, top, right, bottom)))

        return split_image

    def img_concat(self, split_image, all_boxes, all_logits, all_phrases):
        """
        function that concat image and predict results(like bbox)
        """
        adjusted_boxes = []
        images = split_image

        # check image size
        image_width, image_height = images[0].size

        new_boxes = []

        for idx, boxes in enumerate(all_boxes):
            W, H = split_image[0].size

            # if it has to convert cxcywh to xyxy
            if self.convert_cxcywh_to_xyxy:
                boxes = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

            # calculate image's row, col size
            row = idx // 4
            col = idx % 4

            # calculate x, y offset
            x_offset = col * image_width
            y_offset = row * image_height

            for box in boxes:
                # reset coordinate
                adjusted_box = [
                    box[0] + x_offset,  # x1
                    box[1] + y_offset,  # y1
                    box[2] + x_offset,  # x2
                    box[3] + y_offset   # y2
                ]

                new_boxes.append(adjusted_box)

        new_boxes = torch.tensor(new_boxes)

        # concat logits and phrases
        result_logits = torch.cat(all_logits, dim=0)
        result_phrases = []
        for phrases in all_phrases:
            result_phrases.extend(phrases)

        return new_boxes, result_logits, result_phrases
