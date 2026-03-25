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

# 【新增】请求结构
class DeleteRangeRequest(BaseModel):
    job_id: int
    start_frame: int
    end_frame: int
    object_id: int

class GetLabelsRequest(BaseModel):
    job_id: int

@app.post("/api/propagate_next")
def api_propagate_next(req: PropagateNextRequest):
    print(f"[API] Propagate Next: Job {req.job_id}, Frame {req.mask_frame_index}, Obj {req.object_id}")
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
    print(f"[API] Text Prompt: '{req.text_prompt}' on Job {req.job_id}, Frame {req.mask_frame_index}")
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
    print(f"[API] Range Propagate: Job {req.job_id}, Obj {req.object_id}, Frames {req.start_frame}-{req.end_frame}")
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
    print(f"[API] Delete Range: Job {req.job_id}, Obj {req.object_id}, Frames {req.start_frame}-{req.end_frame}")
    deleted_count = pipeline.delete_range(
        job_id=req.job_id,
        start_frame=req.start_frame,
        end_frame=req.end_frame,
        object_id=req.object_id
    )
    return {"status": "success", "deleted": deleted_count}


@app.post("/api/get_labels")
def api_get_labels(req: GetLabelsRequest):
    print(f"[API] Fetching labels for Job {req.job_id}")
    try:
        job = pipeline.client.jobs.retrieve(req.job_id)
        task = pipeline.client.tasks.retrieve(job.task_id)

        labels = []
        for label in task.get_labels():
            labels.append({"id": label.id, "name": label.name})

        return {"status": "success", "labels": labels}
    except Exception as e:
        print(f"[ERROR] Failed to fetch labels: {e}")
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)