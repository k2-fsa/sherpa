from typing import List


def convert_timestamp(
    frames: List[int],
    subsampling_factor: int,
    frame_shift_ms: float = 10,
) -> List[float]:
    """Convert frame numbers to time (in seconds) given subsampling factor
    and frame shift (in milliseconds).

    Args:
      frames:
        A list of frame numbers after subsampling.
      subsampling_factor:
        The subsampling factor of the model.
      frame_shift_ms:
        Frame shift in milliseconds between two contiguous frames.
    Return:
      Return the time in seconds corresponding to each given frame.
    """
    frame_shift = frame_shift_ms / 1000.0
    ans = []
    for f in frames:
        ans.append(f * subsampling_factor * frame_shift)

    return ans
