from enum import Enum
from pydantic import BaseModel, ConfigDict, Field
from datetime import datetime
from typing import List, Optional, Union, Literal


class TaskStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class BaseModelWithDatetime(BaseModel):
    """Base model with proper datetime serialization"""
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda dt: dt.isoformat() if dt else None
        },
        use_enum_values=True
    )


class ResponseStatus(BaseModelWithDatetime):
    """Response status schema"""
    status: TaskStatus = Field(..., description="The status of the response")
    progress: float = Field(..., ge=0.0, le=100.0, description="The progress of the response as a percentage (0-100)")
    queue_position: Optional[int] = Field(None, description="Position in queue (only for queued status) - use with total_queue_size for format like '2/5'")
    total_queue_size: Optional[int] = Field(None, description="Total number of tasks in queue (only for queued status) - use with queue_position for format like '2/5'")
    started_at: Optional[datetime] = Field(None, description="When processing started (only for processing status)")
    estimated_wait_time: Optional[float] = Field(None, description="Estimated wait time in seconds (null if not in queue, queue_position * average_processing_time if in queue)")



class AudioChunk(BaseModelWithDatetime):
    """Individual image data in the response"""
    id: str = Field(..., description="Request identifier - same for all chunks in a request")
    content: Optional[str] = Field(None, description="The content of the chunk")
    audio: List[int] = Field([], description="The audio data")
    status: ResponseStatus = Field(..., description="The status of the response")
    sampling_rate: int = Field(0, description="The sampling rate of the audio")
    finish_reason: Union[Literal["stop", "error"], None] = Field(None, description="The finish reason of the chunk")
