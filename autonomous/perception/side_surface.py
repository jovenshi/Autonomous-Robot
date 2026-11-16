"""Side camera: the road fraction across SIDE_BOX (docs/quiz3_plan.md, stage 8).

The side camera looks out to the right of the robot. SIDE_BOX is a band
across the middle of its (wide) view. Coming into the junction the band's
road fraction ramps up as the corner's grass slides out of view, and back
down past the cross road as the far side's grass slides in;
estimation/location.py turns where it crosses SIDE_EDGE_FRAC into position
fixes (SIDE_EDGES_M). The webcam's colours are washed out (grass barely
passes for green), so the band is judged by the road mask alone. For the
display and the log it is also classed: ROAD at SIDE_ROAD_MIN_FRAC or more
road, GRASS at SIDE_GRASS_MAX_FRAC or less, mixed in between.

process_side_frame is the side camera thread's entry point.
"""
import cv2
import numpy as np

from perception.color import make_side_masks, roi_brightness
from robot_types import Annotation, AnnotationKind, SideResult, Surface


def classify_surface(road_mask, cfg):
    # (Surface or None, road fraction) inside SIDE_BOX.
    x, y, w, h = cfg.SIDE_BOX
    frac = np.count_nonzero(road_mask[y:y + h, x:x + w]) / float(w * h)
    if frac >= cfg.SIDE_ROAD_MIN_FRAC:
        return Surface.ROAD, frac
    if frac <= cfg.SIDE_GRASS_MAX_FRAC:
        return Surface.GRASS, frac
    return None, frac


def process_side_frame(frame, cfg):
    # One side frame -> SideResult with the band's road fraction and class
    # and, for the display, the band and its label.
    road, _ = make_side_masks(frame, cfg)
    surface, frac = classify_surface(road, cfg)
    return SideResult(view=frame.image, side=None,
                      mean_v=_mean_v(frame.image), t=frame.t, seq=frame.seq,
                      surface=surface, annotations=_box_notes(surface, frac, cfg),
                      road_frac=frac)


def _mean_v(image):
    # Mean brightness (V) of the whole frame, as for the front camera.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return roi_brightness(hsv, (0, 0, image.shape[1], image.shape[0]))


def _box_notes(surface, frac, cfg):
    # SIDE_BOX in the surface's colour, and "road 0.85" under it.
    x, y, w, h = cfg.SIDE_BOX
    colour = {Surface.ROAD: cfg.SIDE_ROAD_COLOUR, Surface.GRASS: cfg.SIDE_GRASS_COLOUR}.get(
        surface, cfg.SIDE_MIXED_COLOUR)
    label = f"{surface.name.lower() if surface else 'mixed'} {frac:.2f}"
    return (Annotation(AnnotationKind.RECT, ((x, y), (x + w, y + h)), "", colour, "side"),
            Annotation(AnnotationKind.TEXT, ((x, y + h + 16),), label, colour, "side"))
