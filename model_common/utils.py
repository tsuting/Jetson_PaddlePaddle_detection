import cv2
from typing import List, Union, Tuple
import numpy as np


def fourPointsTransform(
    frame: np.ndarray, vertices: np.ndarray
) -> np.ndarray:
    """
    get cropped image based on the bounding box
    (vertices of rotated rectangle)

    :param frame: input image
    :type frame: np.ndarray
    :param vertices: bbox vertices
    :type vertices: np.ndarray
    :return: cropped image
    :rtype: np.ndarray
    """
    img = frame.copy()
    points = vertices.astype("float32")
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2])))
    pts_std = np.array([
        [0, 0], 
        [img_crop_width, 0],
        [img_crop_width, img_crop_height],
        [0, img_crop_height]
        ], dtype=np.float32)
    rotation_matrix = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
            img,
            rotation_matrix, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC)
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img

def viz_pddle_distill_db_results(
    input_img: np.ndarray, dt_boxes: List
) -> np.ndarray:
    """
    plot detected boxes on the image
    for paddle distill db output

    :param input_img: input_image
    :type input_img: np.ndarray
    :param dt_boxes: List of detected boxes
    :type dt_boxes: List
    :return: image with box annotations
    :rtype: np.ndarray
    """
    viz_img = input_img.copy()
    for box in dt_boxes:
        box = np.array(box).astype(np.int32).reshape(-1, 2)
        cv2.polylines(viz_img, [box], True, color=(255, 255, 0), thickness=2)
    return viz_img

def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right

    :param dt_boxes: detected text boxes with shape [4, 2]
    :type dt_boxes: np.ndarray
    :return: sorted boxes(array) with shape [4, 2]
    :rtype: np.ndarray

    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
                (_boxes[i + 1][0][0] < _boxes[i][0][0]):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    return _boxes

def get_image_width_by_height(image_size: Tuple[int, int], resized_h: int, multiples: Union[None, int]) -> int:
    """Get the image width by the height
    If the multiples are a number, 
        then the width will be the multiples of the number(multiples) in input parameters
    If the multiples is None,
        then just get width by the ratio of height

    :param image_size: the shape of an image, (height, width)
    :type image_size: Tuple[int, int]
    :param resized_h: the height user what to resize
    :type resized_h: int
    :param multiples: the multiples
    :type multiples: Union[None, int]
    :return: the resized width
    :rtype: int
    """

    h, w = image_size
    
    scale_h = resized_h / h
    tar_w = w * scale_h
    if multiples:
        tar_w = tar_w - tar_w % multiples
        tar_w = max(multiples, tar_w)
    
    return int(tar_w)