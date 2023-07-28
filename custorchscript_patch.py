# Copyright (c) Facebook, Inc. and its affiliates.

import os
import sys
import tempfile
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from unittest import mock
import torch
from torch import nn, device
from typing import Any, Dict, List, Tuple, Union
import itertools
import importlib

# need some explicit imports due to https://github.com/pytorch/pytorch/issues/38964
# import detectron2  # noqa F401
# from detectron2.utils.env import _import_file

_counter = 0

def _import_file(module_name, file_path, make_importable=False):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if make_importable:
        sys.modules[module_name] = module
    return module

class Instances:
    """
    This class represents a list of instances in an image.
    It stores the attributes of instances (e.g., boxes, masks, labels, scores) as "fields".
    All fields must have the same ``__len__`` which is the number of instances.

    All other (non-field) attributes of this class are considered private:
    they must start with '_' and are not modifiable by a user.

    Some basic usage:

    1. Set/get/check a field:

       .. code-block:: python

          instances.gt_boxes = Boxes(...)
          print(instances.pred_masks)  # a tensor of shape (N, H, W)
          print('gt_masks' in instances)

    2. ``len(instances)`` returns the number of instances
    3. Indexing: ``instances[indices]`` will apply the indexing on all the fields
       and returns a new :class:`Instances`.
       Typically, ``indices`` is a integer vector of indices,
       or a binary mask of length ``num_instances``

       .. code-block:: python

          category_3_detections = instances[instances.pred_classes == 3]
          confident_detections = instances[instances.scores > 0.9]
    """

    def __init__(self, image_size: Tuple[int, int], **kwargs: Any):
        """
        Args:
            image_size (height, width): the spatial size of the image.
            kwargs: fields to add to this `Instances`.
        """
        self._image_size = image_size
        self._fields: Dict[str, Any] = {}
        for k, v in kwargs.items():
            self.set(k, v)

    @property
    def image_size(self) -> Tuple[int, int]:
        """
        Returns:
            tuple: height, width
        """
        return self._image_size

    def __setattr__(self, name: str, val: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, val)
        else:
            self.set(name, val)

    def __getattr__(self, name: str) -> Any:
        if name == "_fields" or name not in self._fields:
            raise AttributeError("Cannot find field '{}' in the given Instances!".format(name))
        return self._fields[name]

    def set(self, name: str, value: Any) -> None:
        """
        Set the field named `name` to `value`.
        The length of `value` must be the number of instances,
        and must agree with other existing fields in this object.
        """
        data_len = len(value)
        if len(self._fields):
            assert (
                len(self) == data_len
            ), "Adding a field of length {} to a Instances of length {}".format(data_len, len(self))
        self._fields[name] = value

    def has(self, name: str) -> bool:
        """
        Returns:
            bool: whether the field called `name` exists.
        """
        return name in self._fields

    def remove(self, name: str) -> None:
        """
        Remove the field called `name`.
        """
        del self._fields[name]

    def get(self, name: str) -> Any:
        """
        Returns the field called `name`.
        """
        return self._fields[name]

    def get_fields(self) -> Dict[str, Any]:
        """
        Returns:
            dict: a dict which maps names (str) to data of the fields

        Modifying the returned dict will modify this instance.
        """
        return self._fields

    # Tensor-like methods
    def to(self, *args: Any, **kwargs: Any) -> "Instances":
        """
        Returns:
            Instances: all fields are called with a `to(device)`, if the field has this method.
        """
        ret = Instances(self._image_size)
        for k, v in self._fields.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
            ret.set(k, v)
        return ret

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "Instances":
        """
        Args:
            item: an index-like object and will be used to index all the fields.

        Returns:
            If `item` is a string, return the data in the corresponding field.
            Otherwise, returns an `Instances` where all fields are indexed by `item`.
        """
        if type(item) == int:
            if item >= len(self) or item < -len(self):
                raise IndexError("Instances index out of range!")
            else:
                item = slice(item, None, len(self))

        ret = Instances(self._image_size)
        for k, v in self._fields.items():
            ret.set(k, v[item])
        return ret

    def __len__(self) -> int:
        for v in self._fields.values():
            # use __len__ because len() has to be int and is not friendly to tracing
            return v.__len__()
        raise NotImplementedError("Empty Instances does not support __len__!")

    def __iter__(self):
        raise NotImplementedError("`Instances` object is not iterable!")

    @staticmethod
    def cat(instance_lists: List["Instances"]) -> "Instances":
        """
        Args:
            instance_lists (list[Instances])

        Returns:
            Instances
        """
        assert all(isinstance(i, Instances) for i in instance_lists)
        assert len(instance_lists) > 0
        if len(instance_lists) == 1:
            return instance_lists[0]

        image_size = instance_lists[0].image_size
        if not isinstance(image_size, torch.Tensor):  # could be a tensor in tracing
            for i in instance_lists[1:]:
                assert i.image_size == image_size
        ret = Instances(image_size)
        for k in instance_lists[0]._fields.keys():
            values = [i.get(k) for i in instance_lists]
            v0 = values[0]
            if isinstance(v0, torch.Tensor):
                values = torch.cat(values, dim=0)
            elif isinstance(v0, list):
                values = list(itertools.chain(*values))
            elif hasattr(type(v0), "cat"):
                values = type(v0).cat(values)
            else:
                raise ValueError("Unsupported type {} for concatenation".format(type(v0)))
            ret.set(k, values)
        return ret

    def __str__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_instances={}, ".format(len(self))
        s += "image_height={}, ".format(self._image_size[0])
        s += "image_width={}, ".format(self._image_size[1])
        s += "fields=[{}])".format(", ".join((f"{k}: {v}" for k, v in self._fields.items())))
        return s

    __repr__ = __str__


class Boxes:
    """
    This structure stores a list of boxes as a Nx4 torch.Tensor.
    It supports some common methods about boxes
    (`area`, `clip`, `nonempty`, etc),
    and also behaves like a Tensor
    (support indexing, `to(device)`, `.device`, and iteration over all boxes)

    Attributes:
        tensor (torch.Tensor): float matrix of Nx4. Each row is (x1, y1, x2, y2).
    """

    def __init__(self, tensor: torch.Tensor):
        """
        Args:
            tensor (Tensor[float]): a Nx4 matrix.  Each row is (x1, y1, x2, y2).
        """
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor, dtype=torch.float32, device=torch.device("cpu"))
        else:
            tensor = tensor.to(torch.float32)
        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that does not depend on
            # the inputs (and consequently confuses jit)
            tensor = tensor.reshape((-1, 4)).to(dtype=torch.float32)
        assert tensor.dim() == 2 and tensor.size(-1) == 4, tensor.size()

        self.tensor = tensor

    def clone(self) -> "Boxes":
        """
        Clone the Boxes.

        Returns:
            Boxes
        """
        return Boxes(self.tensor.clone())

    def to(self, device: torch.device):
        # Boxes are assumed float32 and does not support to(dtype)
        return Boxes(self.tensor.to(device=device))

    def area(self) -> torch.Tensor:
        """
        Computes the area of all the boxes.

        Returns:
            torch.Tensor: a vector with areas of each box.
        """
        box = self.tensor
        area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
        return area

    def clip(self, box_size: Tuple[int, int]) -> None:
        """
        Clip (in place) the boxes by limiting x coordinates to the range [0, width]
        and y coordinates to the range [0, height].

        Args:
            box_size (height, width): The clipping box's size.
        """
        assert torch.isfinite(self.tensor).all(), "Box tensor contains infinite or NaN!"
        h, w = box_size
        x1 = self.tensor[:, 0].clamp(min=0, max=w)
        y1 = self.tensor[:, 1].clamp(min=0, max=h)
        x2 = self.tensor[:, 2].clamp(min=0, max=w)
        y2 = self.tensor[:, 3].clamp(min=0, max=h)
        self.tensor = torch.stack((x1, y1, x2, y2), dim=-1)

    def nonempty(self, threshold: float = 0.0) -> torch.Tensor:
        """
        Find boxes that are non-empty.
        A box is considered empty, if either of its side is no larger than threshold.

        Returns:
            Tensor:
                a binary vector which represents whether each box is empty
                (False) or non-empty (True).
        """
        box = self.tensor
        widths = box[:, 2] - box[:, 0]
        heights = box[:, 3] - box[:, 1]
        keep = (widths > threshold) & (heights > threshold)
        return keep

    def __getitem__(self, item) -> "Boxes":
        """
        Args:
            item: int, slice, or a BoolTensor

        Returns:
            Boxes: Create a new :class:`Boxes` by indexing.

        The following usage are allowed:

        1. `new_boxes = boxes[3]`: return a `Boxes` which contains only one box.
        2. `new_boxes = boxes[2:10]`: return a slice of boxes.
        3. `new_boxes = boxes[vector]`, where vector is a torch.BoolTensor
           with `length = len(boxes)`. Nonzero elements in the vector will be selected.

        Note that the returned Boxes might share storage with this Boxes,
        subject to Pytorch's indexing semantics.
        """
        if isinstance(item, int):
            return Boxes(self.tensor[item].view(1, -1))
        b = self.tensor[item]
        assert b.dim() == 2, "Indexing on Boxes with {} failed to return a matrix!".format(item)
        return Boxes(b)

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def __repr__(self) -> str:
        return "Boxes(" + str(self.tensor) + ")"

    def inside_box(self, box_size: Tuple[int, int], boundary_threshold: int = 0) -> torch.Tensor:
        """
        Args:
            box_size (height, width): Size of the reference box.
            boundary_threshold (int): Boxes that extend beyond the reference box
                boundary by more than boundary_threshold are considered "outside".

        Returns:
            a binary vector, indicating whether each box is inside the reference box.
        """
        height, width = box_size
        inds_inside = (
            (self.tensor[..., 0] >= -boundary_threshold)
            & (self.tensor[..., 1] >= -boundary_threshold)
            & (self.tensor[..., 2] < width + boundary_threshold)
            & (self.tensor[..., 3] < height + boundary_threshold)
        )
        return inds_inside

    def get_centers(self) -> torch.Tensor:
        """
        Returns:
            The box centers in a Nx2 array of (x, y).
        """
        return (self.tensor[:, :2] + self.tensor[:, 2:]) / 2

    def scale(self, scale_x: float, scale_y: float) -> None:
        """
        Scale the box with horizontal and vertical scaling factors
        """
        self.tensor[:, 0::2] *= scale_x
        self.tensor[:, 1::2] *= scale_y

    @classmethod
    def cat(cls, boxes_list: List["Boxes"]) -> "Boxes":
        """
        Concatenates a list of Boxes into a single Boxes

        Arguments:
            boxes_list (list[Boxes])

        Returns:
            Boxes: the concatenated Boxes
        """
        assert isinstance(boxes_list, (list, tuple))
        if len(boxes_list) == 0:
            return cls(torch.empty(0))
        assert all([isinstance(box, Boxes) for box in boxes_list])

        # use torch.cat (v.s. layers.cat) so the returned boxes never share storage with input
        cat_boxes = cls(torch.cat([b.tensor for b in boxes_list], dim=0))
        return cat_boxes

    @property
    def device(self) -> device:
        return self.tensor.device

    # type "Iterator[torch.Tensor]", yield, and iter() not supported by torchscript
    # https://github.com/pytorch/pytorch/issues/18627
    @torch.jit.unused
    def __iter__(self):
        """
        Yield a box as a Tensor of shape (4,) at a time.
        """
        yield from self.tensor

def _clear_jit_cache():
    from torch.jit._recursive import concrete_type_store
    from torch.jit._state import _jit_caching_layer

    concrete_type_store.type_store.clear()  # for modules
    _jit_caching_layer.clear()  # for free functions


def _add_instances_conversion_methods(newInstances):
    """
    Add from_instances methods to the scripted Instances class.
    """
    cls_name = newInstances.__name__

    @torch.jit.unused
    def from_instances(instances: Instances):
        """
        Create scripted Instances from original Instances
        """
        fields = instances.get_fields()
        image_size = instances.image_size
        ret = newInstances(image_size)
        for name, val in fields.items():
            assert hasattr(ret, f"_{name}"), f"No attribute named {name} in {cls_name}"
            setattr(ret, name, deepcopy(val))
        return ret

    newInstances.from_instances = from_instances


@contextmanager
def patch_instances(fields):
    """
    A contextmanager, under which the Instances class in detectron2 is replaced
    by a statically-typed scriptable class, defined by `fields`.
    See more in `scripting_with_instances`.
    """

    with tempfile.TemporaryDirectory(prefix="detectron2") as dir, tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".py", dir=dir, delete=False
    ) as f:
        try:
            # Objects that use Instances should not reuse previously-compiled
            # results in cache, because `Instances` could be a new class each time.
            _clear_jit_cache()

            cls_name, s = _gen_instance_module(fields)
            f.write(s)
            f.flush()
            f.close()

            module = _import(f.name)
            new_instances = getattr(module, cls_name)
            _ = torch.jit.script(new_instances)
            # let torchscript think Instances was scripted already
            Instances.__torch_script_class__ = True
            # let torchscript find new_instances when looking for the jit type of Instances
            Instances._jit_override_qualname = torch._jit_internal._qualified_name(new_instances)

            _add_instances_conversion_methods(new_instances)
            yield new_instances
        finally:
            try:
                del Instances.__torch_script_class__
                del Instances._jit_override_qualname
            except AttributeError:
                pass
            sys.modules.pop(module.__name__)


def _gen_instance_class(fields):
    """
    Args:
        fields (dict[name: type])
    """

    class _FieldType:
        def __init__(self, name, type_):
            assert isinstance(name, str), f"Field name must be str, got {name}"
            self.name = name
            self.type_ = type_
            self.annotation = f"{type_.__module__}.{type_.__name__}"

    fields = [_FieldType(k, v) for k, v in fields.items()]

    def indent(level, s):
        return " " * 4 * level + s

    lines = []

    global _counter
    _counter += 1

    cls_name = "ScriptedInstances{}".format(_counter)

    field_names = tuple(x.name for x in fields)
    extra_args = ", ".join([f"{f.name}: Optional[{f.annotation}] = None" for f in fields])
    lines.append(
        f"""
class {cls_name}:
    def __init__(self, image_size: Tuple[int, int], {extra_args}):
        self.image_size = image_size
        self._field_names = {field_names}
"""
    )

    for f in fields:
        lines.append(
            indent(2, f"self._{f.name} = torch.jit.annotate(Optional[{f.annotation}], {f.name})")
        )

    for f in fields:
        lines.append(
            f"""
    @property
    def {f.name}(self) -> {f.annotation}:
        # has to use a local for type refinement
        # https://pytorch.org/docs/stable/jit_language_reference.html#optional-type-refinement
        t = self._{f.name}
        assert t is not None, "{f.name} is None and cannot be accessed!"
        return t

    @{f.name}.setter
    def {f.name}(self, value: {f.annotation}) -> None:
        self._{f.name} = value
"""
        )

    # support method `__len__`
    lines.append(
        """
    def __len__(self) -> int:
"""
    )
    for f in fields:
        lines.append(
            f"""
        t = self._{f.name}
        if t is not None:
            return len(t)
"""
        )
    lines.append(
        """
        raise NotImplementedError("Empty Instances does not support __len__!")
"""
    )

    # support method `has`
    lines.append(
        """
    def has(self, name: str) -> bool:
"""
    )
    for f in fields:
        lines.append(
            f"""
        if name == "{f.name}":
            return self._{f.name} is not None
"""
        )
    lines.append(
        """
        return False
"""
    )

    # support method `to`
    none_args = ", None" * len(fields)
    lines.append(
        f"""
    def to(self, device: torch.device) -> "{cls_name}":
        ret = {cls_name}(self.image_size{none_args})
"""
    )
    for f in fields:
        if hasattr(f.type_, "to"):
            lines.append(
                f"""
        t = self._{f.name}
        if t is not None:
            ret._{f.name} = t.to(device)
"""
            )
        else:
            # For now, ignore fields that cannot be moved to devices.
            # Maybe can support other tensor-like classes (e.g. __torch_function__)
            pass
    lines.append(
        """
        return ret
"""
    )

    # support method `getitem`
    none_args = ", None" * len(fields)
    lines.append(
        f"""
    def __getitem__(self, item) -> "{cls_name}":
        ret = {cls_name}(self.image_size{none_args})
"""
    )
    for f in fields:
        lines.append(
            f"""
        t = self._{f.name}
        if t is not None:
            ret._{f.name} = t[item]
"""
        )
    lines.append(
        """
        return ret
"""
    )

    # support method `cat`
    # this version does not contain checks that all instances have same size and fields
    none_args = ", None" * len(fields)
    lines.append(
        f"""
    def cat(self, instances: List["{cls_name}"]) -> "{cls_name}":
        ret = {cls_name}(self.image_size{none_args})
"""
    )
    for f in fields:
        lines.append(
            f"""
        t = self._{f.name}
        if t is not None:
            values: List[{f.annotation}] = [x.{f.name} for x in instances]
            if torch.jit.isinstance(t, torch.Tensor):
                ret._{f.name} = torch.cat(values, dim=0)
            else:
                ret._{f.name} = t.cat(values)
"""
        )
    lines.append(
        """
        return ret"""
    )

    # support method `get_fields()`
    lines.append(
        """
    def get_fields(self) -> Dict[str, Tensor]:
        ret = {}
    """
    )
    for f in fields:
        if f.type_ == Boxes:
            stmt = "t.tensor"
        elif f.type_ == torch.Tensor:
            stmt = "t"
        else:
            stmt = f'assert False, "unsupported type {str(f.type_)}"'
        lines.append(
            f"""
        t = self._{f.name}
        if t is not None:
            ret["{f.name}"] = {stmt}
        """
        )
    lines.append(
        """
        return ret"""
    )
    return cls_name, os.linesep.join(lines)


def _gen_instance_module(fields):
    # TODO: find a more automatic way to enable import of other classes
    s = """
from copy import deepcopy
import torch
from torch import Tensor
import typing
from typing import *

import detectron2
from detectron2.structures import Boxes, Instances

"""

    cls_name, cls_def = _gen_instance_class(fields)
    s += cls_def
    return cls_name, s


def _import(path):
    return _import_file(
        "{}{}".format(sys.modules[__name__].__name__, _counter), path, make_importable=True
    )


# @contextmanager
# def patch_builtin_len(modules=()):
#     """
#     Patch the builtin len() function of a few detectron2 modules
#     to use __len__ instead, because __len__ does not convert values to
#     integers and therefore is friendly to tracing.

#     Args:
#         modules (list[stsr]): names of extra modules to patch len(), in
#             addition to those in detectron2.
#     """

#     def _new_len(obj):
#         return obj.__len__()

#     with ExitStack() as stack:
#         MODULES = [
#             "detectron2.modeling.roi_heads.fast_rcnn",
#             "detectron2.modeling.roi_heads.mask_head",
#             "detectron2.modeling.roi_heads.keypoint_head",
#         ] + list(modules)
#         ctxs = [stack.enter_context(mock.patch(mod + ".len")) for mod in MODULES]
#         for m in ctxs:
#             m.side_effect = _new_len
#         yield


# def patch_nonscriptable_classes():
#     """
#     Apply patches on a few nonscriptable detectron2 classes.
#     Should not have side-effects on eager usage.
#     """
#     # __prepare_scriptable__ can also be added to models for easier maintenance.
#     # But it complicates the clean model code.

#     from detectron2.modeling.backbone import ResNet, FPN

#     # Due to https://github.com/pytorch/pytorch/issues/36061,
#     # we change backbone to use ModuleList for scripting.
#     # (note: this changes param names in state_dict)

#     def prepare_resnet(self):
#         ret = deepcopy(self)
#         ret.stages = nn.ModuleList(ret.stages)
#         for k in self.stage_names:
#             delattr(ret, k)
#         return ret

#     ResNet.__prepare_scriptable__ = prepare_resnet

#     def prepare_fpn(self):
#         ret = deepcopy(self)
#         ret.lateral_convs = nn.ModuleList(ret.lateral_convs)
#         ret.output_convs = nn.ModuleList(ret.output_convs)
#         for name, _ in self.named_children():
#             if name.startswith("fpn_"):
#                 delattr(ret, name)
#         return ret

#     FPN.__prepare_scriptable__ = prepare_fpn

#     # Annotate some attributes to be constants for the purpose of scripting,
#     # even though they are not constants in eager mode.
#     from detectron2.modeling.roi_heads import StandardROIHeads

#     if hasattr(StandardROIHeads, "__annotations__"):
#         # copy first to avoid editing annotations of base class
#         StandardROIHeads.__annotations__ = deepcopy(StandardROIHeads.__annotations__)
#         StandardROIHeads.__annotations__["mask_on"] = torch.jit.Final[bool]
#         StandardROIHeads.__annotations__["keypoint_on"] = torch.jit.Final[bool]


# # These patches are not supposed to have side-effects.
# patch_nonscriptable_classes()


# @contextmanager
# def freeze_training_mode(model):
#     """
#     A context manager that annotates the "training" attribute of every submodule
#     to constant, so that the training codepath in these modules can be
#     meta-compiled away. Upon exiting, the annotations are reverted.
#     """
#     classes = {type(x) for x in model.modules()}
#     # __constants__ is the old way to annotate constants and not compatible
#     # with __annotations__ .
#     classes = {x for x in classes if not hasattr(x, "__constants__")}
#     for cls in classes:
#         cls.__annotations__["training"] = torch.jit.Final[bool]
#     yield
#     for cls in classes:
#         cls.__annotations__["training"] = bool
