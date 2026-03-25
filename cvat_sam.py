import os
import cv2
import numpy as np
import torch
from PIL import Image
from cvat_sdk import models
from cvat_sdk.core.client import Client, Config
from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor
from accelerate import Accelerator
from collections import defaultdict

# CVAT & Environment Configurations
CVAT_URL = "http://10.4.4.24:8080"
USERNAME = "zhixinma"
PASSWORD = "dv2W4VTDYxFrsLr"
HF_MODEL  = "facebook/sam3"
HOME_PATH = os.path.expanduser("~")
VIDEO_DIR = os.path.join(HOME_PATH, "data/SCVOS/dataset/videos_fps6/videos")


def calculate_iou(poly1, poly2, width, height):
    """
    计算两个 Polygon 之间的 Intersection over Union (IoU)
    poly1, poly2: flat lists [x1, y1, x2, y2, ...]
    """
    mask1 = np.zeros((height, width), dtype=np.uint8)
    mask2 = np.zeros((height, width), dtype=np.uint8)

    pts1 = np.array(poly1, dtype=np.int32).reshape(-1, 2)
    pts2 = np.array(poly2, dtype=np.int32).reshape(-1, 2)

    cv2.fillPoly(mask1, [pts1], 1)
    cv2.fillPoly(mask2, [pts2], 1)

    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 0.0
    return intersection / union

def get_polygon_from_cvat_shape(shape):
    """
    将 CVAT 的不同 Shape 类型统一转换为 flat polygon points
    """
    stype = str(shape.type)
    if stype == "polygon":
        return shape.points
    elif stype == "rectangle":
        x1, y1, x2, y2 = shape.points
        return [x1, y1, x2, y1, x2, y2, x1, y2]
    return None


def filter_polygons_by_iou(new_shapes, existing_shapes, width, height, iou_threshold=0.5):
    """
    考虑每个 frame 上所有的 polygons 的 overlap 情况。优先保留原有的 annotation。
    针对同一帧的新生成 polygon：如果 polygon A 的 95% 以上被 polygon B 覆盖，则删掉 polygon A。
    """
    existing_masks = {}
    if existing_shapes:
        for shape in existing_shapes:
            key = (shape.frame, shape.label_id)
            if key not in existing_masks:
                existing_masks[key] = np.zeros((height, width), dtype=np.uint8)

            poly_points = get_polygon_from_cvat_shape(shape)
            if poly_points:
                pts = np.array(poly_points, dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(existing_masks[key], [pts], 1)

    filtered_shapes = []

    new_frames_map = defaultdict(list)
    for shape in new_shapes:
        new_frames_map[shape.frame].append(shape)

    for frame_idx, frame_new_shapes in new_frames_map.items():
        kept_in_frame = []

        for new_shape in frame_new_shapes:
            label_id = new_shape.label_id
            discard = False

            # 栅格化新生成的 Polygon
            new_mask = np.zeros((height, width), dtype=np.uint8)
            pts = np.array(new_shape.points, dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(new_mask, [pts], 1)
            new_area = new_mask.sum()

            if new_area == 0:
                continue

            key = (frame_idx, label_id)
            if key in existing_masks:
                ex_mask = existing_masks[key]
                intersection = np.logical_and(new_mask, ex_mask).sum()
                union = new_area + ex_mask.sum() - intersection

                iou = intersection / union if union > 0 else 0
                io_new = intersection / new_area if new_area > 0 else 0

                # 判定规则：
                # 1. 整体 IoU 大于阈值 (e.g., > 0.5)
                # 2. 或者新生成的 Polygon 有 70% 以上的面积和原有 Mask 重合
                if iou > iou_threshold or io_new > 0.7:
                    discard = True

            if discard:
                continue

            shapes_to_remove = []
            for kept_shape in kept_in_frame:
                if kept_shape.label_id != label_id:
                    continue

                kept_mask = np.zeros((height, width), dtype=np.uint8)
                k_pts = np.array(kept_shape.points, dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(kept_mask, [k_pts], 1)
                kept_area = kept_mask.sum()

                # 计算交集和覆盖率
                intersection = np.logical_and(new_mask, kept_mask).sum()
                union = new_area + kept_area - intersection
                iou = intersection / union if union > 0 else 0

                io_new = intersection / new_area if new_area > 0 else 0
                io_kept = intersection / kept_area if kept_area > 0 else 0

                # Rule 1：如果两个 polygon 高度重合，或者【新的 Polygon 95% 以上被已保留的覆盖】，则丢弃新的
                if iou > 0.8 or io_new >= 0.80:
                    discard = True
                    break

                # Rule 2：如果【已保留的 Polygon 95% 以上被新的 Polygon 覆盖】，则准备删掉旧的
                if io_kept >= 0.80:
                    shapes_to_remove.append(kept_shape)

            # 如果新的 Polygon 没有被丢弃，则将其加入保留列表，并剔除被它覆盖掉的旧 Polygon
            if not discard:
                for s_remove in shapes_to_remove:
                    kept_in_frame.remove(s_remove)
                kept_in_frame.append(new_shape)

        filtered_shapes.extend(kept_in_frame)

    original_len = len(new_shapes)
    filtered_len = len(filtered_shapes)
    if original_len != filtered_len:
        print(f"[INFO] Mask Filter removed {original_len - filtered_len} overlapping/duplicate polygon(s).")

    return filtered_shapes


def mask_to_polygons(mask: np.ndarray, min_points=3):
    mask_u8 = np.asarray(mask, dtype=np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, epsilon=1.5, closed=True)
        if approx.shape[0] >= min_points:
            polygons.append(approx.flatten().tolist())
    return polygons


class SAM3CVATPipeline:
    def __init__(self):
        print("[INFO] Connecting to CVAT server...")
        self.client = Client(url=CVAT_URL, config=Config(verify_ssl=False))
        self.client.login((USERNAME, PASSWORD))

        print("[INFO] Loading Sam3TrackerVideoModel from HuggingFace...")
        self.device = Accelerator().device
        self.model = Sam3TrackerVideoModel.from_pretrained(
            HF_MODEL, torch_dtype=torch.bfloat16
        ).to(self.device)
        self.processor = Sam3TrackerVideoProcessor.from_pretrained(HF_MODEL)

    def _get_video_path(self, job):
        task = self.client.tasks.retrieve(job.task_id)
        meta = task.get_meta()
        if not meta.frames:
            raise RuntimeError("No frame metadata found.")
        return os.path.join(VIDEO_DIR, meta.frames[0].name)

    def _load_video_frames(self, video_fp, start_frame, stop_frame):
        video_frames = []
        cap = cv2.VideoCapture(video_fp)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {video_fp}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for frame_id in range(start_frame, stop_frame + 1):
            ret, bgr = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            video_frames.append(Image.fromarray(rgb))
        cap.release()

        width, height = video_frames[0].size
        return video_frames, width, height

    def _add_prompts_from_annotations(self, inference_session, annotations, target_obj_id, start_abs_frame, stop_abs_frame):
        num_shape_added = 0
        for shape in annotations.shapes:
            if shape.label_id != target_obj_id:
                continue
            if shape.frame < start_abs_frame or shape.frame > stop_abs_frame:
                continue

            frame_idx_relative = shape.frame - start_abs_frame
            stype = str(shape.type)

            if stype == "rectangle":
                self.processor.add_inputs_to_inference_session(
                    inference_session=inference_session, frame_idx=frame_idx_relative,
                    obj_ids=[target_obj_id], input_boxes=[[shape.points]]
                )
            elif stype == "points":
                pts = np.array(shape.points, dtype=np.float32).reshape(-1, 2).tolist()
                labels = [1] * len(pts)
                self.processor.add_inputs_to_inference_session(
                    inference_session=inference_session, frame_idx=frame_idx_relative,
                    obj_ids=[target_obj_id], input_points=[[pts]], input_labels=[[labels]]
                )
            elif stype == "polygon":
                pts = np.array(shape.points, dtype=np.float32).reshape(-1, 2)
                x_min, y_min = pts.min(axis=0)
                x_max, y_max = pts.max(axis=0)
                box = [float(x_min), float(y_min), float(x_max), float(y_max)]
                self.processor.add_inputs_to_inference_session(
                    inference_session=inference_session, frame_idx=frame_idx_relative,
                    obj_ids=[target_obj_id], input_boxes=[[box]]
                )
            else:
                raise ValueError(f"Unsupported shape type: {stype}")
            num_shape_added += 1
        return num_shape_added

    def _run_propagation(self, inference_session, start_abs_frame, total_frames):
        new_shapes = []
        for out in self.model.propagate_in_video_iterator(
            inference_session=inference_session,
            start_frame_idx=0,
            max_frame_num_to_track=total_frames
        ):
            video_res_masks = self.processor.post_process_masks(
                masks=[out.pred_masks],
                original_sizes=[[inference_session.video_height, inference_session.video_width]],
                binarize=True
            )[0]

            abs_frame_id = start_abs_frame + out.frame_idx

            for i, obj_id in enumerate(inference_session.obj_ids):
                mask_tensor = video_res_masks[i]
                mask = mask_tensor.cpu().numpy().squeeze()
                if mask.ndim > 2:
                    mask = mask[0]

                mask_bin = (mask > 0.0).astype(np.uint8)
                polygons = mask_to_polygons(mask_bin)

                for poly_pts in polygons:
                    new_shapes.append(
                        models.LabeledShapeRequest(
                            type="polygon",
                            frame=abs_frame_id,
                            label_id=obj_id,
                            points=poly_pts
                        )
                    )
        return new_shapes

    def display_labels(self, job_id):
        job = self.client.jobs.retrieve(job_id)
        task = self.client.tasks.retrieve(job.task_id)
        lable_id_to_name = {}
        for label in task.get_labels():
            lable_id_to_name[label.id] = label.name
        print(f"[INFO] Label ID to name mapping:")
        for label_id, label_name in lable_id_to_name.items():
            print(f"  {label_id}: {label_name}")
        print()

    def propagate_from_frame(self, job_id, mask_frame_index, num_frame_propagate, object_id):
        print(f"\n[MODE 1] Propagating object {object_id} from frame {mask_frame_index} for {num_frame_propagate} frames.")
        job = self.client.jobs.retrieve(job_id)
        video_fp = self._get_video_path(job)

        stop_frame_index = mask_frame_index + num_frame_propagate
        frames, width, height = self._load_video_frames(video_fp, mask_frame_index, stop_frame_index)

        inference_session = self.processor.init_video_session(
            video=frames, inference_device=self.device, dtype=torch.bfloat16
        )

        annotations = job.get_annotations()
        num_shape_added = self._add_prompts_from_annotations(
            inference_session, annotations, object_id,
            mask_frame_index, mask_frame_index
        )
        if num_shape_added == 0:
            print("[WARNING] No valid shapes found in this job range. Exiting tracking.")
            return []
        new_shapes = self._run_propagation(inference_session, mask_frame_index, len(frames))
        return filter_polygons_by_iou(new_shapes, annotations.shapes, width, height, iou_threshold=0.9)

    def segment_by_text(self, job_id, mask_frame_index, text_prompt, object_id):
        print(f"\n[MODE 2] Text prompting '{text_prompt}' on frame {mask_frame_index} for object {object_id}.")
        job = self.client.jobs.retrieve(job_id)
        video_fp = self._get_video_path(job)

        frames, width, height = self._load_video_frames(video_fp, mask_frame_index, mask_frame_index)
        frame_pil = frames[0]

        if not hasattr(self, "text_model"):
            print("[INFO] Loading Sam3Model (Image/Text) from HuggingFace...")
            from transformers import Sam3Model, Sam3Processor
            self.text_model = Sam3Model.from_pretrained(
                HF_MODEL, torch_dtype=torch.bfloat16
            ).to(self.device)
            self.text_processor = Sam3Processor.from_pretrained(HF_MODEL)

        print(f"[INFO] Running text segmentation for concept: '{text_prompt}'")

        inputs = self.text_processor(
            images=frame_pil, text=text_prompt, return_tensors="pt"
        )
        inputs = {
            k: v.to(self.device, dtype=torch.bfloat16) if torch.is_floating_point(v) else v.to(self.device)
            for k, v in inputs.items()
        }

        with torch.no_grad():
            outputs = self.text_model(**inputs)

        results = self.text_processor.post_process_instance_segmentation(
            outputs,
            threshold=0.3, # 置信度阈值，如果你觉得模型找得不够准可以调高这个值
            mask_threshold=0.5,
            target_sizes=[(height, width)]
        )[0]

        masks = results["masks"].cpu().numpy()

        new_shapes = []
        for mask in masks:
            mask_bin = (mask > 0).astype(np.uint8)
            polygons = mask_to_polygons(mask_bin)

            for poly_pts in polygons:
                new_shapes.append(
                    models.LabeledShapeRequest(
                        type="polygon",
                        frame=mask_frame_index,
                        label_id=object_id,
                        points=poly_pts
                    )
                )

        annotations = job.get_annotations()
        return filter_polygons_by_iou(new_shapes, annotations.shapes, width, height, iou_threshold=0.9)

    def propagate_range(self, job_id, start_frame, end_frame, object_id):
        print(f"\n[MODE 4] Propagating object {object_id} from frame {start_frame} to {end_frame}.")
        job = self.client.jobs.retrieve(job_id)
        video_fp = self._get_video_path(job)

        frames, width, height = self._load_video_frames(video_fp, start_frame, end_frame)

        annotations = job.get_annotations()

        # 提取范围内所有的 frames，包含多个实例或分离的多边形
        target_shapes = [s for s in annotations.shapes if s.label_id == object_id and start_frame <= s.frame <= end_frame]

        if not target_shapes:
            print(f"[WARNING] No valid shapes found for object {object_id} between frame {start_frame} and {end_frame}.")
            return []

        print(f"[INFO] Found {len(target_shapes)} distinct shapes. Tracking each independently to prevent memory crashes...")

        new_shapes = []

        for shape in target_shapes:
            shape_start_idx = shape.frame - start_frame

            # 为当前 Shape 独立初始化一个 Session，彻底隔离状态，防止内存污染和崩溃
            inference_session = self.processor.init_video_session(
                video=frames, inference_device=self.device, dtype=torch.bfloat16
            )

            stype = str(shape.type)
            internal_id = 1 # 因为每次都是独立的全新 session，所以 ID 可以设为 1

            # 注入 Prompt
            if stype == "rectangle":
                self.processor.add_inputs_to_inference_session(
                    inference_session=inference_session, frame_idx=shape_start_idx,
                    obj_ids=[internal_id], input_boxes=[[shape.points]]
                )
            elif stype == "points":
                pts = np.array(shape.points, dtype=np.float32).reshape(-1, 2).tolist()
                labels = [1] * len(pts)
                self.processor.add_inputs_to_inference_session(
                    inference_session=inference_session, frame_idx=shape_start_idx,
                    obj_ids=[internal_id], input_points=[[pts]], input_labels=[[labels]]
                )
            elif stype == "polygon":
                pts = np.array(shape.points, dtype=np.float32).reshape(-1, 2)
                x_min, y_min = pts.min(axis=0)
                x_max, y_max = pts.max(axis=0)
                box = [float(x_min), float(y_min), float(x_max), float(y_max)]
                self.processor.add_inputs_to_inference_session(
                    inference_session=inference_session, frame_idx=shape_start_idx,
                    obj_ids=[internal_id], input_boxes=[[box]]
                )

            for out in self.model.propagate_in_video_iterator(
                inference_session=inference_session,
                start_frame_idx=shape_start_idx,
                max_frame_num_to_track=len(frames) - shape_start_idx
            ):
                abs_frame_id = start_frame + out.frame_idx

                video_res_masks = self.processor.post_process_masks(
                    masks=[out.pred_masks],
                    original_sizes=[[inference_session.video_height, inference_session.video_width]],
                    binarize=True
                )[0]

                mask_tensor = video_res_masks[0] # 因为只有一个对象，直接取索引 0
                mask = mask_tensor.cpu().numpy().squeeze()
                if mask.ndim > 2:
                    mask = mask[0]

                mask_bin = (mask > 0.0).astype(np.uint8)
                polygons = mask_to_polygons(mask_bin)

                for poly_pts in polygons:
                    new_shapes.append(
                        models.LabeledShapeRequest(
                            type="polygon",
                            frame=abs_frame_id,
                            label_id=object_id, # 恢复为真实的 CVAT object_id
                            points=poly_pts
                        )
                    )

        return filter_polygons_by_iou(new_shapes, annotations.shapes, width, height, iou_threshold=0.5)


    def delete_range(self, job_id, start_frame, end_frame, object_id):
        print(f"\n[MODE 5] Deleting object {object_id} from frame {start_frame} to {end_frame}.")
        job = self.client.jobs.retrieve(job_id)

        # 抓取目前所有的 annotations
        annotations = job.get_annotations()

        # 过滤并转换：找出符合条件的形状，并将其转换为 CVAT 强制要求的 Request 类型
        shapes_to_delete = []
        for s in annotations.shapes:
            if s.label_id == object_id and start_frame <= s.frame <= end_frame:
                shapes_to_delete.append(
                    models.LabeledShapeRequest(
                        id=s.id,
                        frame=s.frame,
                        label_id=s.label_id,
                        type=s.type,
                        points=s.points
                    )
                )

        if not shapes_to_delete:
            print(f"[INFO] No shapes found to delete for object {object_id} in range {start_frame}-{end_frame}.")
            return 0

        delete_request = models.PatchedLabeledDataRequest(
            version=annotations.version,
            tags=[],
            shapes=shapes_to_delete,
            tracks=[]
        )

        class CVATDeleteAction:
            value = "delete"

        job.update_annotations(delete_request, action=CVATDeleteAction)
        print(f"[INFO] Successfully deleted {len(shapes_to_delete)} shapes.")

        return len(shapes_to_delete)

    def change_label_range(self, job_id, start_frame, end_frame, object_src, object_tgt):
        print(f"\n[MODE 6] Changing label from {object_src} to {object_tgt} between frames {start_frame}-{end_frame}.")
        job = self.client.jobs.retrieve(job_id)

        # 获取当前的 annotations
        annotations = job.get_annotations()

        # 找出范围内属于 object_src 的 shapes，并将其 label_id 替换为 object_tgt
        shapes_to_update = []
        for s in annotations.shapes:
            if s.label_id == object_src and start_frame <= s.frame <= end_frame:
                shapes_to_update.append(
                    models.LabeledShapeRequest(
                        id=s.id,              # 保留原有的 ID
                        frame=s.frame,
                        label_id=object_tgt,  # 替换为新的 target label
                        type=s.type,
                        points=s.points
                    )
                )

        if not shapes_to_update:
            print(f"[INFO] No shapes found to update for object {object_src} in range {start_frame}-{end_frame}.")
            return 0

        update_request = models.PatchedLabeledDataRequest(
            version=annotations.version,
            tags=[],
            shapes=shapes_to_update,
            tracks=[]
        )

        class CVATUpdateAction:
            value = "update"

        job.update_annotations(update_request, action=CVATUpdateAction)
        print(f"[INFO] Successfully updated label for {len(shapes_to_update)} shapes.")

        return len(shapes_to_update)

    def track_full_video(self, job_id, object_id):
        print(f"\n[MODE 3] Full video track for object {object_id} in job {job_id}.")
        job = self.client.jobs.retrieve(job_id)
        video_fp = self._get_video_path(job)

        start_frame = job.start_frame
        stop_frame = job.stop_frame

        frames, width, height = self._load_video_frames(video_fp, start_frame, stop_frame)

        inference_session = self.processor.init_video_session(
            video=frames, inference_device=self.device, dtype=torch.bfloat16
        )

        annotations = job.get_annotations()
        self._add_prompts_from_annotations(
            inference_session, annotations, object_id,
            start_frame, stop_frame
        )

        new_shapes = self._run_propagation(inference_session, start_frame, len(frames))
        return filter_polygons_by_iou(new_shapes, annotations.shapes, width, height, iou_threshold=0.9)

    def upload_to_cvat(self, job_id, new_shapes):
        if not new_shapes:
            print("[WARNING] No shapes generated to upload.")
            return

        print(f"[INFO] Uploading {len(new_shapes)} tracking results back to CVAT job {job_id}...")
        job = self.client.jobs.retrieve(job_id)

        update_request = models.PatchedLabeledDataRequest(
            version=0,
            tags=[],
            shapes=new_shapes,
            tracks=[]
        )

        job.update_annotations(update_request)
        print("[INFO] All done! Automatic tracking synchronised to CVAT.")

