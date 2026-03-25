from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from cvat_sam import SAM3CVATPipeline

HOST = "0.0.0.0"
PORT = 8081

app = FastAPI(title="SAM3 CVAT Tracker API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("[INFO] Initializing SAM3 Pipeline... This might take a moment.")
pipeline = SAM3CVATPipeline()

class PropagateNextRequest(BaseModel):
    job_id: int
    mask_frame_index: int
    object_id: int

class TextPromptRequest(BaseModel):
    job_id: int
    mask_frame_index: int
    text_prompt: str
    object_id: int

class PropagateRangeRequest(BaseModel):
    job_id: int
    start_frame: int
    end_frame: int
    object_id: int

class DeleteRangeRequest(BaseModel):
    job_id: int
    start_frame: int
    end_frame: int
    object_id: int

class ChangeLabelRangeRequest(BaseModel):
    job_id: int
    start_frame: int
    end_frame: int
    object_src: int
    object_tgt: int

class GetLabelsRequest(BaseModel):
    job_id: int

@app.post("/api/propagate_next")
def api_propagate_next(req: PropagateNextRequest):
    shapes = pipeline.propagate_from_frame(
        job_id=req.job_id,
        mask_frame_index=req.mask_frame_index,
        num_frame_propagate=1,
        object_id=req.object_id
    )
    pipeline.upload_to_cvat(req.job_id, shapes)
    return {"status": "success", "generated": len(shapes)}

@app.post("/api/segment_text")
def api_segment_text(req: TextPromptRequest):
    shapes = pipeline.segment_by_text(
        job_id=req.job_id,
        mask_frame_index=req.mask_frame_index,
        text_prompt=req.text_prompt,
        object_id=req.object_id
    )
    pipeline.upload_to_cvat(req.job_id, shapes)
    return {"status": "success", "generated": len(shapes)}

@app.post("/api/propagate_range")
def api_propagate_range(req: PropagateRangeRequest):
    shapes = pipeline.propagate_range(
        job_id=req.job_id,
        start_frame=req.start_frame,
        end_frame=req.end_frame,
        object_id=req.object_id
    )
    pipeline.upload_to_cvat(req.job_id, shapes)
    return {"status": "success", "generated": len(shapes)}

@app.post("/api/delete_range")
def api_delete_range(req: DeleteRangeRequest):
    deleted_count = pipeline.delete_range(
        job_id=req.job_id,
        start_frame=req.start_frame,
        end_frame=req.end_frame,
        object_id=req.object_id
    )
    return {"status": "success", "deleted": deleted_count}

@app.post("/api/change_label_range")
def api_change_label_range(req: ChangeLabelRangeRequest):
    print(f"[API] Change Label Range: Job {req.job_id}, Frames {req.start_frame}-{req.end_frame}, {req.object_src} -> {req.object_tgt}")
    updated_count = pipeline.change_label_range(
        job_id=req.job_id,
        start_frame=req.start_frame,
        end_frame=req.end_frame,
        object_src=req.object_src,
        object_tgt=req.object_tgt
    )
    return {"status": "success", "updated": updated_count}

@app.post("/api/get_labels")
def api_get_labels(req: GetLabelsRequest):
    try:
        job = pipeline.client.jobs.retrieve(req.job_id)
        task = pipeline.client.tasks.retrieve(job.task_id)
        labels = [{"id": label.id, "name": label.name} for label in task.get_labels()]
        return {"status": "success", "labels": labels}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)