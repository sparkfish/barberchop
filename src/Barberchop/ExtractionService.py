import cv2
import torch
from PIL import Image
from yolov5 import YOLOv5

class Barcode:
  def __init__(self, type_id, type, confidence, image, x1, y1, x2, y2):
    self.type_id = type_id
    self.type = type
    self.confidence = confidence
    self.image = image
    self.x1 = x1
    self.x2 = x2
    self.y1 = y1
    self.y2 = y2

class BarberchopYoloService():
  def __init__(self, weights, resize_shape=(1000,1000), image_sizes=[640]):
    self.resize_shape = resize_shape
    self.image_sizes = image_sizes
    self.detector = YOLOv5(weights, 'cpu')

  def extract(self, document_image):
    barcodes = list()

    width, height = document_image.shape[0], document_image.shape[1]

    for image_size in self.image_sizes:
      if (self.resize_shape is not None):
        resized_image = cv2.resize(document_image, self.resize_shape)
      else:
        resized_image = document_image

      pred = self.detector.predict([resized_image], size=image_size)
      results = pred.xywhn[0].numpy()
      for barcode in results:
        center_x = width * barcode[0]
        center_y = height * barcode[1]
        barcode_width = width * (barcode[2] + .02)
        barcode_height = height * (barcode[3] + .02)

        top_left_x = int(center_x - (barcode_width / 2))
        top_left_y = int(center_y - (barcode_height / 2))

        bottom_right_x = int(center_x + (barcode_width / 2))
        bottom_right_y = int(center_y + (barcode_height / 2))

        barcode_class = pred.names[int(barcode[5])]
        barcodes.append(Barcode(barcode[5], barcode_class, barcode[4], None, top_left_x, top_left_y, bottom_right_x, bottom_right_y))

      barcodes = self.consolidate_overlapping_barcodes(barcodes)
      barcodes = self.crop_barcodes(document_image, barcodes)

    return barcodes

  def crop_barcodes(self, document_image, barcodes):
    for barcode in barcodes:
      barcode.image = document_image[barcode.y1:barcode.y2, barcode.x1:barcode.x2]

    return barcodes

  def consolidate_overlapping_barcodes(self, barcodes):
    overlapping_barcodes = list()
    merged_overlapping_barcodes = list()

    for barcode_a in barcodes:
      for barcode_b in barcodes:

        # Checking against self, skip
        if (barcode_a == barcode_b):
          continue

        # Both barcode bounding boxes are already found to be overlapping
        if (barcode_a in overlapping_barcodes and barcode_b in overlapping_barcodes):
          continue
        
        XA1 = barcode_a.x1
        YA1 = barcode_a.y1
        XA2 = barcode_a.x2
        YA2 = barcode_a.y2

        XB1 = barcode_b.x1
        YB1 = barcode_b.y1
        XB2 = barcode_b.x2
        YB2 = barcode_b.y2

        SA = abs(XA1 - XA2) * abs(YA1-YA2)
        SB = abs(XB1 - XB2) * abs(YB1-YB2)
        SI = max(0, min(XA2, XB2) - max(XA1, XB1)) * max(0, min(YA2, YB2) - max(YA1, YB1))    
        SU = SA + SB - SI
        percent_overlap = SI / SU

        if (percent_overlap > 0.33):
          if (barcode_a not in overlapping_barcodes):
            overlapping_barcodes.append(barcode_a)
          if (barcode_b not in overlapping_barcodes):
            overlapping_barcodes.append(barcode_b)

          new_x1 = min(XA1, XA2, XB1, XB2)
          new_y1 = min(YA1, YA2, YB1, YB2)
          new_x2 = max(XA1, XA2, XB1, XB2)
          new_y2 = max(YA1, YA2, YB1, YB2)

          if (barcode_a.type_id == 0):
            type_id, type, confidence = barcode_b.type_id, barcode_b.type, barcode_b.confidence
          elif (barcode_b.type_id == 0):
            type_id, type, confidence = barcode_a.type_id, barcode_a.type, barcode_a.confidence
          elif (barcode_a.confidence > barcode_b.confidence):
            type_id, type, confidence = barcode_a.type_id, barcode_a.type, barcode_a.confidence
          else:
            type_id, type, confidence = barcode_b.type_id, barcode_b.type, barcode_b.confidence

          merged_overlapping_barcodes.append(Barcode(type_id, type, confidence, None, new_x1, new_y1, new_x2, new_y2))

    for barcode in overlapping_barcodes:
      barcodes.remove(barcode)

    for barcode in merged_overlapping_barcodes:
      barcodes.append(barcode)

    return barcodes