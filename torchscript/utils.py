import torch, torchvision
from typing import Tuple
import numpy as np
from torch.nn import functional as F

def ResizeShortestEdge(image: torch.Tensor, short_edge_length: int, max_size: int):
    current_h, current_w = image.shape[1], image.shape[2]
    
    size = short_edge_length * 1.0
    scale = size / min(current_h, current_w)
    if current_h < current_w:
        new_h, new_w = size, scale * current_w
    else:
        new_h, new_w = scale * current_h, size
    if max(new_h, new_w) > max_size:
        scale = max_size * 1.0 / max(new_h, new_w)
        new_h = new_h * scale
        new_w = new_w * scale
    new_w = int(new_w + 0.5)
    new_h = int(new_h + 0.5)
    # since torch vanilla Resize also have max_size to restrict longer edge
    # but the result have slightly different from detectron2
    # different in infer
    ### torchvision only takes PIL.image or Tensor
    newimage = torchvision.transforms.Resize(size=(new_h, new_w))(image)
    return newimage, current_h / new_h, current_w / new_w

def postprocess(result, scale_h, scale_v, og_h, og_w, threshold=0.5):
    ### inplace modified
    '''
    scale_h: horizontal scale factor
    scale_v: vertical scale factor
    og_w: original width size
    og_h: original height size
    threshold: those instance confindence higher than thres. will be preserved
    '''
    # for idx, score in enumerate(result["scores"]):
    #     if score < threshold:
    #         break
    
    # result["scores"] = result["scores"][:idx]
    # result["pred_boxes"] = result["pred_boxes"][:idx]
    # result["pred_classes"] = result["pred_classes"][:idx]
    # result["pred_masks"] = result["pred_masks"][:idx]

    ### scaling (since we have resize image to feed in nn)
    result["pred_boxes"][:, 0::2] *= scale_h
    result["pred_boxes"][:, 1::2] *= scale_v

    ### clipping (after scaling to og size, it may exceed)
    # h, w = box_size
    x1 = result["pred_boxes"][:, 0].clamp(min=0, max=og_w)
    y1 = result["pred_boxes"][:, 1].clamp(min=0, max=og_h)
    x2 = result["pred_boxes"][:, 2].clamp(min=0, max=og_w)
    y2 = result["pred_boxes"][:, 3].clamp(min=0, max=og_h)
    result["pred_boxes"] = torch.stack((x1, y1, x2, y2), dim=-1)

    ### postprocess mask
    ### (N, 1, M, M) -> (N, M, M)
    result["pred_masks"] = result["pred_masks"][:, 0, :, :]
    ### or result["pred_masks"] = torch.squeeze(result["pred_masks"])
    result["pred_masks"] = paste_masks_in_image(result["pred_masks"], 
                                                result["pred_boxes"],
                                                (og_h, og_w))
    
@torch.jit.script_if_tracing
def paste_masks_in_image(
    masks: torch.Tensor, boxes: torch.Tensor, image_shape: Tuple[int, int], threshold: float = 0.5
):
    ### from: detectron2/layers/mask_ops.py
    """
    Paste a set of masks that are of a fixed resolution (e.g., 28 x 28) into an image.
    The location, height, and width for pasting each mask is determined by their
    corresponding bounding boxes in boxes.

    Note:
        This is a complicated but more accurate implementation. In actual deployment, it is
        often enough to use a faster but less accurate implementation.
        See :func:`paste_mask_in_image_old` in this file for an alternative implementation.

    Args:
        masks (tensor): Tensor of shape (Bimg, Hmask, Wmask), where Bimg is the number of
            detected object instances in the image and Hmask, Wmask are the mask width and mask
            height of the predicted mask (e.g., Hmask = Wmask = 28). Values are in [0, 1].
        boxes (Boxes or Tensor): A Boxes of length Bimg or Tensor of shape (Bimg, 4).
            boxes[i] and masks[i] correspond to the same object instance.
        image_shape (tuple): height, width
        threshold (float): A threshold in [0, 1] for converting the (soft) masks to
            binary masks.

    Returns:
        img_masks (Tensor): A tensor of shape (Bimg, Himage, Wimage), where Bimg is the
        number of detected object instances and Himage, Wimage are the image width
        and height. img_masks[i] is a binary mask for object instance i.
    """
    BYTES_PER_FLOAT = 4
    # TODO: This memory limit may be too much or too little. It would be better to
    # determine it based on available resources.
    GPU_MEM_LIMIT = 1024**3  # 1 GB memory limit

    assert masks.shape[-1] == masks.shape[-2], "Only square mask predictions are supported"
    N = len(masks)
    if N == 0:
        return masks.new_empty((0,) + image_shape, dtype=torch.uint8)
    if not isinstance(boxes, torch.Tensor):
        boxes = boxes.tensor
    device = boxes.device
    assert len(boxes) == N, boxes.shape

    img_h, img_w = image_shape
    # The actual implementation split the input into chunks,
    # and paste them chunk by chunk.
    if device.type == "cpu" or torch.jit.is_scripting():
        # CPU is most efficient when they are pasted one by one with skip_empty=True
        # so that it performs minimal number of operations.
        num_chunks = N
    else:
        # GPU benefits from parallelism for larger chunks, but may have memory issue
        # int(img_h) because shape may be tensors in tracing
        num_chunks = int(np.ceil(N * int(img_h) * int(img_w) * BYTES_PER_FLOAT / GPU_MEM_LIMIT))
        assert (
            num_chunks <= N
        ), "Default GPU_MEM_LIMIT in mask_ops.py is too small; try increasing it"
    chunks = torch.chunk(torch.arange(N, device=device), num_chunks)

    img_masks = torch.zeros(
        N, img_h, img_w, device=device, dtype=torch.bool if threshold >= 0 else torch.uint8
    )
    for inds in chunks:
        masks_chunk, spatial_inds = _do_paste_mask(
            masks[inds, None, :, :], boxes[inds], img_h, img_w, skip_empty=device.type == "cpu"
        )

        if threshold >= 0:
            masks_chunk = (masks_chunk >= threshold).to(dtype=torch.bool)
        else:
            # for visualization and debugging
            masks_chunk = (masks_chunk * 255).to(dtype=torch.uint8)

        if torch.jit.is_scripting():  # Scripting does not use the optimized codepath
            img_masks[inds] = masks_chunk
        else:
            img_masks[(inds,) + spatial_inds] = masks_chunk
    return img_masks

def _do_paste_mask(masks, boxes, img_h: int, img_w: int, skip_empty: bool = True):
    ### from: detectron2/layers/mask_ops.py
    """
    Args:
        masks: N, 1, H, W
        boxes: N, 4
        img_h, img_w (int):
        skip_empty (bool): only paste masks within the region that
            tightly bound all boxes, and returns the results this region only.
            An important optimization for CPU.

    Returns:
        if skip_empty == False, a mask of shape (N, img_h, img_w)
        if skip_empty == True, a mask of shape (N, h', w'), and the slice
            object for the corresponding region.
    """
    # On GPU, paste all masks together (up to chunk size)
    # by using the entire image to sample the masks
    # Compared to pasting them one by one,
    # this has more operations but is faster on COCO-scale dataset.
    device = masks.device

    if skip_empty and not torch.jit.is_scripting():
        x0_int, y0_int = torch.clamp(boxes.min(dim=0).values.floor()[:2] - 1, min=0).to(
            dtype=torch.int32
        )
        x1_int = torch.clamp(boxes[:, 2].max().ceil() + 1, max=img_w).to(dtype=torch.int32)
        y1_int = torch.clamp(boxes[:, 3].max().ceil() + 1, max=img_h).to(dtype=torch.int32)
    else:
        x0_int, y0_int = 0, 0
        x1_int, y1_int = img_w, img_h
    x0, y0, x1, y1 = torch.split(boxes, 1, dim=1)  # each is Nx1

    N = masks.shape[0]

    img_y = torch.arange(y0_int, y1_int, device=device, dtype=torch.float32) + 0.5
    img_x = torch.arange(x0_int, x1_int, device=device, dtype=torch.float32) + 0.5
    img_y = (img_y - y0) / (y1 - y0) * 2 - 1
    img_x = (img_x - x0) / (x1 - x0) * 2 - 1
    # img_x, img_y have shapes (N, w), (N, h)

    gx = img_x[:, None, :].expand(N, img_y.size(1), img_x.size(1))
    gy = img_y[:, :, None].expand(N, img_y.size(1), img_x.size(1))
    grid = torch.stack([gx, gy], dim=3)

    if not torch.jit.is_scripting():
        if not masks.dtype.is_floating_point:
            masks = masks.float()
    img_masks = F.grid_sample(masks, grid.to(masks.dtype), align_corners=False)

    if skip_empty and not torch.jit.is_scripting():
        return img_masks[:, 0], (slice(y0_int, y1_int), slice(x0_int, x1_int))
    else:
        return img_masks[:, 0], ()
