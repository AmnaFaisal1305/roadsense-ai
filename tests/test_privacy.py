import numpy as np

from src.privacy import blur_boxes


def test_blur_changes_only_the_box():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
    out = blur_boxes(frame, [[10, 10, 50, 50]])
    assert not np.array_equal(out[10:50, 10:50], frame[10:50, 10:50])  # box is blurred
    out[10:50, 10:50] = frame[10:50, 10:50]
    assert np.array_equal(out, frame)  # nothing outside the box changed


def test_out_of_frame_and_empty_boxes_are_safe():
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    out = blur_boxes(frame, [[-5, -5, 30, 30], [5, 5, 5, 5]])
    assert out.shape == frame.shape
