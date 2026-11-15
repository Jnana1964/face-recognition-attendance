"""Head-turn liveness check (anti-spoofing).

Idea: estimate head yaw from YuNet's 5 landmarks as the horizontal offset of the nose tip
from the midpoint of the eyes, divided by the distance between the eyes.

    yaw = (nose_x - eyes_mid_x) / eye_distance

A real head turning left/right moves the nose clearly to one side (|yaw| ~ 0.3-0.6).
A flat printed or on-screen photo cannot do this: tilting it only compresses the whole
image, so the nose stays at the same relative position (yaw stays near its start value).

A person passes when, within CHALLENGE_SECONDS, their yaw goes beyond +YAW_THRESHOLD
on one side AND below -YAW_THRESHOLD on the other.
"""
import time

YAW_THRESHOLD = 0.22
CHALLENGE_SECONDS = 10.0


def head_yaw(face):
    """face = YuNet row: x, y, w, h, re_x, re_y, le_x, le_y, nose_x, nose_y, ... score."""
    re_x, le_x, nose_x = face[4], face[6], face[8]
    eye_dist = abs(le_x - re_x) + 1e-6
    return float((nose_x - (re_x + le_x) / 2) / eye_dist)


class LivenessChallenge:
    """Tracks one person's head-turn challenge."""

    def __init__(self, now=None):
        self.reset(now)

    def reset(self, now=None):
        self.start = time.time() if now is None else now
        self.min_yaw = 0.0
        self.max_yaw = 0.0

    def update(self, yaw, now=None):
        now = time.time() if now is None else now
        if now - self.start > CHALLENGE_SECONDS and not self.passed:
            self.reset(now)  # took too long - start over
        self.min_yaw = min(self.min_yaw, yaw)
        self.max_yaw = max(self.max_yaw, yaw)

    @property
    def turned_one_side(self):
        return self.max_yaw > YAW_THRESHOLD

    @property
    def turned_other_side(self):
        return self.min_yaw < -YAW_THRESHOLD

    @property
    def passed(self):
        return self.turned_one_side and self.turned_other_side

    def hint(self):
        if self.passed:
            return "live"
        if not (self.turned_one_side or self.turned_other_side):
            return "turn head left & right"
        return "now turn the other way"
