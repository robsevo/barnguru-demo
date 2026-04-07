from typing import TypedDict

def gretzky_version() -> str: ...

class TrackDict(TypedDict):
    id: int
    cx: float
    cy: float
    w: float
    h: float
    vx: float
    vy: float
    state: str        # "confirmed"
    class_id: int

class Tracker:
    """
    Multi-object Kalman tracker (Feature 16.6).

    State vector : [cx, cy, vx, vy, w, h]
    Observations : [cx, cy, w, h]

    Track lifecycle:
        Tentative → Confirmed (after n_init hits)
        Confirmed → Lost      (first missed frame)
        Lost      → Deleted   (after max_lost more misses)

    Re-ID: a Lost track is re-linked when a new detection reappears
    within fps*2 frames with IoU ≥ reid_iou.
    """

    def __init__(
        self,
        fps: float = 30.0,
        n_init: int = 3,
        max_lost: int = 30,
        iou_threshold: float = 0.3,
        reid_iou: float = 0.3,
    ) -> None: ...

    def update(self, detections: list[list[float]]) -> list[TrackDict]:
        """
        Process one frame of detections.

        Args:
            detections: list of [x, y, w, h, confidence, class_id]
                        where (x, y) is the top-left corner in pixels.

        Returns:
            List of dicts for every confirmed track.
        """
        ...

    def reset(self) -> None:
        """Clear all tracks and reset the frame counter."""
        ...

    def confirmed_count(self) -> int:
        """Number of currently confirmed tracks."""
        ...

    def active_count(self) -> int:
        """Number of active tracks (confirmed + tentative)."""
        ...

    def frame_count(self) -> int:
        """Frames processed since last reset."""
        ...
