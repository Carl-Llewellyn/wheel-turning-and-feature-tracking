#If anyone actually reads this. This mostly works and could be made to fully work.
#Although it detects ~-360 to +360 deg for my car. It should be a lot more than this haha.
#It will detect the wheel angle quite well. However, the only way I could get this to consistently work
#was by just resetting the feature detection, etc. every 5 seconds. It may work better for you.
#I also made this in the summer and used it in late Autumn. This could be a source of a lot of the issues as
#the lighting was so different.
#I'm also not a python person. I did not enjoy making this. I tried this in GoCV and that was a mess.

from __future__ import annotations

import argparse
import math

import sys
import os
import mmap
import struct
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class TrackerConfig:
    #These were made when I first started this project. Many of them may no longer be relavent.
    maxCorners: int = 450
    qualityLevel: float = 0.01
    minDistance: int = 7
    windowSize: Tuple[int, int] = (21, 21)
    lkMaxLevel: int = 3
    LkCriteria: Tuple[int, int, float] = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        30,
        0.01,
    )
    redetectEveryNFrames: int = 35
    minTrackedPoints: int = 25
    minAngleSamples: int = 12
    SmoothAlpha: float = 0.15
    nearZeroThresholdDeg: float = 0.4
    maxAngleStdDeg: float = 6.0
    GradientJumpThreshold: float = 12.0
    motionAngleThresholdDeg: float = 9.0  #T_θ in paper
    MotionDerivativeThresholdDeg: float = 3.0  #T_Δθ
    derivativeWindow: int = 5
    alignBackToZeroDeg: float = 0.8
    allowManualClick: bool = True
    debugMode: bool = False


class SteeringLKTracker:
    def __init__(self, config: TrackerConfig):
        self.cfg = config
        self.center: Optional[np.ndarray] = None
        self.radius: Optional[float] = None
        self.mask: Optional[np.ndarray] = None

        #state that evolves per frame
        self.prevGray: Optional[np.ndarray] = None
        self.prevPts: np.ndarray = np.empty((0, 2), dtype=np.float32)
        self.angleDeg: float = 0.0
        self.smoothedAngle: float = 0.0
        self.motionType: str = "rect"  # <rect> or <curve>
        self.lastRelativeAngle: float = 0.0
        self.angleHistory: List[float] = [0.0]
        self.frameIdx: int = 0
        self.lastReseedFrame: int = 0
        self.periodicResetIntervalFrames: int = 0
        self.lastPeriodicResetFrame: int = 0
        self.shmFd: Optional[int] = None
        self.shmMap: Optional[mmap.mmap] = None
        self.shmPath = "/dev/shm/steering_angle"
        self.shmSize = 8  #double buffer only

    def resetDetection(
        self,
        resetAngle: bool,
        wheelDetectionReset: bool,
        currentFrame: Optional[np.ndarray] = None,
    ) -> None:
        """Reset feature tracking and optionally the wheel detection and accumulated steering angle."""
        if resetAngle:
            self.angleDeg = 0.0
            self.smoothedAngle = 0.0
            self.angleHistory = [0.0]
        else:
            self.angleHistory = [self.angleDeg]
            self.smoothedAngle = self.angleDeg

        self.lastRelativeAngle = 0.0
        self.motionType = "rect"
        self.prevPts = np.empty((0, 2), dtype=np.float32)
        self.prevGray = None
        self.frameIdx = 0
        self.lastReseedFrame = 0
        if wheelDetectionReset:
            self.center = None
            self.radius = None
            self.mask = None
        self.lastPeriodicResetFrame = self.frameIdx

        if currentFrame is None:
            return

        frameInput = currentFrame.copy()
        if frameInput.ndim == 2 or (frameInput.ndim == 3 and frameInput.shape[2] == 1):
            frameBgr = cv2.cvtColor(frameInput, cv2.COLOR_GRAY2BGR)
        else:
            frameBgr = frameInput

        if wheelDetectionReset or self.center is None or self.mask is None:
            #quick hack: reseed mask because otherwise it forgets spoke pattern.
            self._initialise_from_first_frame(frameBgr)

        gray = cv2.cvtColor(frameBgr, cv2.COLOR_BGR2GRAY)
        self.prevGray = gray

        p0 = self._detect_features(gray, useMask=self.mask is not None)
        if p0 is None or len(p0) == 0:
            p0 = self._detect_features(gray, useMask=False)
        if p0 is not None and len(p0) > 0:
            self.prevPts = p0.reshape(-1, 2)
            self.lastReseedFrame = self.frameIdx
        else:
            self.prevPts = np.empty((0, 2), dtype=np.float32)

        if self.cfg.debugMode:
            print(
                f"[reset] detection reset (resetAngle={resetAngle}, wheelDetectionReset={wheelDetectionReset})"
            )

    def _initSharedMem(self) -> None:
        if self.shmMap is not None:
            return
        try:
            fd = os.open(self.shmPath, os.O_CREAT | os.O_RDWR, 0o666)
            os.ftruncate(fd, self.shmSize)
            mm = mmap.mmap(fd, self.shmSize, mmap.MAP_SHARED, mmap.PROT_WRITE)
        except OSError as shmErr:
            if "fd" in locals():
                os.close(fd)
            if self.cfg.debugMode:
                print("shm err", shmErr)
            self.shmFd = None
            self.shmMap = None
            return
        self.shmFd = fd
        self.shmMap = mm
        self._writeSharedAngle(float("nan"))  #start as NaN so reader knows it's not ready

    def _writeSharedAngle(self, angleVal: float) -> None:
        if self.shmMap is None:
            return
        try:
            self.shmMap.seek(0)
            self.shmMap.write(struct.pack("<d", angleVal))
            self.shmMap.flush()
        except (OSError, ValueError) as writeErr:
            if self.cfg.debugMode:
                print("shm write??", writeErr)

    def _closeSharedMem(self) -> None:
        if self.shmMap is not None:
            try:
                self.shmMap.close()
            except BufferError:
                if self.cfg.debugMode:
                    print("shm close buffer busy?")
            self.shmMap = None
        if self.shmFd is not None:
            os.close(self.shmFd)
            self.shmFd = None

    def run(self, videoSource: str) -> None:
        cap = cv2.VideoCapture(videoSource)
        if not cap.isOpened():
            print(f"Cannot open video: {videoSource}", file=sys.stderr)
            sys.exit(1)

        if isinstance(videoSource, int):
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  #This is needed because otherwise it lags like shit at 1080p. For my camera anyway (some 1080p logitech).
            dbgWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            dbgHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            dbgFps = cap.get(cv2.CAP_PROP_FPS)
            if self.cfg.debugMode:
                print("cam-set?", dbgWidth, dbgHeight, dbgFps)

        self._initSharedMem()
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        #TODO(fixlater) usb camera lies about fps sometimes but i eyeball it.

        ok, frame = cap.read()
        if not ok:
            print("Vid empty.", file=sys.stderr)
            sys.exit(1)

        self._initialise_from_first_frame(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.prevGray = gray
        self.prevPts = self._detect_features(gray, useMask=True)
        self.prevPts = self.prevPts.reshape(-1, 2) if self.prevPts is not None else np.empty((0, 2))
        self.frameIdx = 0
        self.angleDeg = 0.0
        self.smoothedAngle = 0.0
        self.motionType = "rect"
        self.lastRelativeAngle = 0.0
        self.angleHistory = [0.0]
        self.lastReseedFrame = 0
        self.periodicResetIntervalFrames = max(1, int(round(fps * 5.0)))

        #Can't get this to work for long periods of time. Resetting every 5 secs makes it work well enough.
        quickResetTick = self.periodicResetIntervalFrames  
        if self.cfg.debugMode and quickResetTick < 240:
            print("resetTick?", quickResetTick)
        self.lastPeriodicResetFrame = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            self.frameIdx += 1
            timeSec = self.frameIdx / fps
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            measurement = self._process_frame(gray, frame=frame)
            self._apply_measurement(measurement)
            self._maybe_reseed(gray, force=measurement.forceReseed)

            vis = self._draw_overlay(frame.copy())
            cv2.imshow("steering-lk", vis)
            if measurement.valid:
                self._writeSharedAngle(float(self.smoothedAngle))
            else:
                self._writeSharedAngle(float("nan"))

            self.prevGray = gray
            self.prevPts = measurement.nextPoints.reshape(-1, 2) if measurement.nextPoints is not None else np.empty((0, 2))

            if (
                self.periodicResetIntervalFrames > 0
                and self.frameIdx - self.lastPeriodicResetFrame >= self.periodicResetIntervalFrames
            ):
                self.resetDetection(
                    resetAngle=False,
                    wheelDetectionReset=False,
                    currentFrame=frame,
                )

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            if key == ord("r"):
                self.resetDetection(resetAngle=True, wheelDetectionReset=True, currentFrame=frame)

        cap.release()
        cv2.destroyAllWindows()
        self._writeSharedAngle(float("nan"))  #tell downstream we're idle now.
        self._closeSharedMem()

    def _initialise_from_first_frame(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        print("Hough circle start...") 
        blurred = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=120,
            param1=100,
            param2=35,
            minRadius=int(min(h, w) * 0.08),
            maxRadius=int(min(h, w) * 0.55),
        )
        if circles is not None and len(circles) > 0:
            c = circles[0, 0]
            self.center = np.array([c[0], c[1]], dtype=np.float32)
            self.radius = float(c[2])
            print(f"Wheel centre at {self.center}, radius {self.radius:.1f}")
        elif self.cfg.allowManualClick:
            print("Can't find wheel. Click center then rim.")
            self.center, self.radius = self._manual_select_center(frame)
        else:
            self.center = np.array([w / 2.0, h / 2.0], dtype=np.float32)
            self.radius = min(h, w) * 0.25
            print("Automatic detection failed; using frame centre heuristic.")

        if self.center is None or self.radius is None:
            self.center = np.array([w / 2.0, h / 2.0], dtype=np.float32)
            self.radius = min(h, w) * 0.25
            print("Falling back to frame centre due to missing clicks.")

        self.mask = np.zeros_like(gray)
        outerR = int(self.radius * 0.97)
        innerR = int(self.radius * 0.52)
        cv2.circle(self.mask, (int(self.center[0]), int(self.center[1])), outerR, 255, -1)
        cv2.circle(self.mask, (int(self.center[0]), int(self.center[1])), innerR, 0, -1)

    def _manual_select_center(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        coords: List[Tuple[int, int]] = []

        def onclick(event, x, y, *_):
            if event == cv2.EVENT_LBUTTONDOWN:
                coords.append((x, y))
                print(f"Clicked: ({x}, {y})")

        cv2.namedWindow("select-wheel")
        cv2.setMouseCallback("select-wheel", onclick)
        while len(coords) < 2:
            disp = frame.copy()
            for c in coords:
                cv2.circle(disp, c, 5, (0, 255, 0), -1)
            cv2.imshow("select-wheel", disp)
            if cv2.waitKey(30) & 0xFF == 27:
                break
        cv2.destroyWindow("select-wheel")

        if len(coords) >= 2:
            centre = np.array(coords[0], dtype=np.float32)
            rim = np.array(coords[1], dtype=np.float32)
            radius = float(np.linalg.norm(rim - centre))
            print(f"Manual centre {centre}, radius {radius:.1f}")
            return centre, radius

        h, w = frame.shape[:2]
        print("No clicks registered; defaulting to frame centre heuristic.")
        return np.array([w / 2.0, h / 2.0], dtype=np.float32), float(min(h, w) * 0.25)

    def _detect_features(self, gray: np.ndarray, useMask: bool) -> Optional[np.ndarray]:
        mask = self.mask if useMask and self.mask is not None else None
        p0 = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.cfg.maxCorners, 
            qualityLevel=self.cfg.qualityLevel,
            minDistance=self.cfg.minDistance,
            mask=mask,
        )
        return p0

    @dataclass
    class Measurement:
        valid: bool
        relativeAngle: float
        nextPoints: np.ndarray
        trackedPoints: int
        forceReseed: bool
        motionType: str
        RawCandidates: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.float32))

    def _process_frame(self, gray: np.ndarray, frame: Optional[np.ndarray] = None) -> "SteeringLKTracker.Measurement":
        if self.prevGray is None:
            return self.Measurement(
                valid=False,
                relativeAngle=0.0,
                nextPoints=self.prevPts,
                trackedPoints=len(self.prevPts),
                forceReseed=True,
                motionType=self.motionType,
            )

        nextPts, prevPtsValid = self._track_points(self.prevGray, gray, self.prevPts)
        trackedPoints = len(nextPts)
        ptsOk = trackedPoints  #v2 of angle calc uses this count differently.
        if self.cfg.debugMode and self.frameIdx % 7 == 0:
            print(ptsOk) 

        if trackedPoints < self.cfg.minTrackedPoints:
            return self.Measurement(
                valid=False,
                relativeAngle=0.0,
                nextPoints=nextPts,
                trackedPoints=trackedPoints,
                forceReseed=True,
                motionType=self.motionType,
            )

        angleDeltas = self._compute_angleDeltas(prevPtsValid, nextPts)
        stats = self._angle_statistics(angleDeltas)
        noiseFlag = self._detect_horizontal_noise(self.prevGray, gray)
        motionType = self._classify_motion(stats, predictedAngle=self.angleDeg + stats.median)
        relativeAngle, measureValid = self._decide_relativeAngle(stats, motionType, noiseFlag)

        if not measureValid and self.cfg.debugMode:
            print(
                f"[frame {self.frameIdx}] invalid measurement: "
                f"pts={trackedPoints}, "
                f"std={stats.std:.2f}, "
                f"noise={noiseFlag}"
            )
            

        return self.Measurement(
            valid=measureValid,
            relativeAngle=relativeAngle,
            nextPoints=nextPts,
            trackedPoints=trackedPoints,
            forceReseed=trackedPoints < self.cfg.minTrackedPoints * 1.5,
            motionType=motionType,
            RawCandidates=angleDeltas,
        )

    def _apply_measurement(self, measurement: "SteeringLKTracker.Measurement") -> None:
        previousMotion = self.motionType
        if measurement.valid:
            self.angleDeg += measurement.relativeAngle
            self.lastRelativeAngle = measurement.relativeAngle
        else:
            #invalid condition handling according to k1/k2 branches
            if previousMotion == measurement.motionType:
                #hold angle
                pass
            else:
                #continue with previous relative angle for short gaps
                self.angleDeg += self.lastRelativeAngle

        self.angleDeg = self._maybe_align_zero(self.angleDeg)
        self.angleHistory.append(self.angleDeg)
        self.motionType = measurement.motionType
        #TODO revisit smoothing once I trust calibration again.
        alpha = 0.18
        # alpha = self.cfg.SmoothAlpha
        self.smoothedAngle = alpha * self.angleDeg + (1.0 - alpha) * self.smoothedAngle

    def _maybe_reseed(self, gray: np.ndarray, force: bool) -> None:
        needReseed = force or (self.frameIdx - self.lastReseedFrame > self.cfg.redetectEveryNFrames)
        if not needReseed:
            return

        p0 = self._detect_features(gray, useMask=True)
        if p0 is None or len(p0) == 0:
            p0 = self._detect_features(gray, useMask=False)
        if p0 is not None and len(p0) > 0:
            self.prevPts = p0.reshape(-1, 2)
            self.lastReseedFrame = self.frameIdx

    def _track_points(
        self, prevGray: np.ndarray, gray: np.ndarray, prevPts: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if len(prevPts) == 0:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
        p1, st, _err = cv2.calcOpticalFlowPyrLK(
            prevGray,
            gray,
            prevPts.astype(np.float32),
            None,
            winSize=self.cfg.windowSize,
            maxLevel=self.cfg.lkMaxLevel,
            criteria=self.cfg.LkCriteria,
        )
        if p1 is None or st is None:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
        goodNew = p1[st.reshape(-1) == 1]
        goodOld = prevPts[st.reshape(-1) == 1]
        return goodNew, goodOld

    @dataclass
    class AngleStats:
        median: float
        std: float
        positiveMean: Optional[float]
        negativeMean: Optional[float]
        positiveCount: int
        negativeCount: int
        sampleCount: int

    def _compute_angleDeltas(self, prevPts: np.ndarray, nextPts: np.ndarray) -> np.ndarray:
        center = self.center if self.center is not None else np.array([0.0, 0.0], dtype=np.float32)
        prevVectors = prevPts - center
        nextVectors = nextPts - center
        prevAngles = np.rad2deg(np.arctan2(prevVectors[:, 1], prevVectors[:, 0]))
        nextAngles = np.rad2deg(np.arctan2(nextVectors[:, 1], nextVectors[:, 0]))
        rawDelta = nextAngles - prevAngles
        angleDelta = rawDelta 
        #map to [-180, 180]
        rawDelta = (angleDelta + 180.0) % 360.0 - 180.0
        return rawDelta.astype(np.float32)

    def _angle_statistics(self, deltaAngles: np.ndarray) -> AngleStats:
        if deltaAngles.size == 0:
            return self.AngleStats(0.0, float("inf"), None, None, 0, 0, 0)

        mask = np.abs(deltaAngles) > self.cfg.nearZeroThresholdDeg
        filtered = deltaAngles[mask]
        if filtered.size == 0:
            filtered = np.array([0.0], dtype=np.float32)

        median = float(np.median(filtered))
        std = float(np.std(filtered))
        positive = filtered[filtered > 0]
        negative = filtered[filtered < 0]

        positiveMean = float(np.mean(positive)) if positive.size else None
        negativeMean = float(np.mean(negative)) if negative.size else None

        return self.AngleStats(
            median=median,
            std=std,
            positiveMean=positiveMean,
            negativeMean=negativeMean,
            positiveCount=int(positive.size),
            negativeCount=int(negative.size),
            sampleCount=int(filtered.size),
        )

    def _classify_motion(self, stats: AngleStats, predictedAngle: float) -> str:
        prevMotion = self.motionType
        if len(self.angleHistory) <= self.cfg.derivativeWindow:
            derivative = 0.0
        else:
            derivative = abs(predictedAngle - self.angleHistory[-self.cfg.derivativeWindow])

        if (
            prevMotion == "rect"
            and abs(predictedAngle) < 9.0 
        ) or (
            prevMotion == "curve"
            and abs(predictedAngle) < 9.0
            and derivative < 3.0
        ):
            return "rect"
        return "curve"

    def _decide_relativeAngle(
        self, stats: AngleStats, motionType: str, noiseFlag: bool
    ) -> Tuple[float, bool]:
        if stats.sampleCount < self.cfg.minAngleSamples:
            return 0.0, False
        if stats.std > self.cfg.maxAngleStdDeg:
            return 0.0, False
        if noiseFlag:
            return 0.0, False

        if motionType == "rect":
            relAngle = stats.median
        else:
            if stats.positiveCount >= stats.negativeCount and stats.positiveMean is not None:
                relAngle = stats.positiveMean
            elif stats.negativeMean is not None:
                relAngle = stats.negativeMean
            else:
                relAngle = stats.median

        return relAngle, True

    def _detect_horizontal_noise(self, prevGray: np.ndarray, gray: np.ndarray) -> bool:
        sobelPrev = cv2.Sobel(prevGray, cv2.CV_32F, 0, 1, ksize=3)
        sobelCur = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        avgPrev = float(np.mean(np.abs(sobelPrev)))
        avgCur = float(np.mean(np.abs(sobelCur)))
        return abs(avgCur - avgPrev) > 12.0  

    def _maybe_align_zero(self, angle: float) -> float:
        if abs(angle) < 0.8:  
            return 0.0
        return angle

    def _draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        if self.center is not None:
            cv2.circle(frame, (int(self.center[0]), int(self.center[1])), 4, (0, 255, 0), -1)
        if self.radius is not None and self.center is not None:
            cv2.circle(frame, (int(self.center[0]), int(self.center[1])), int(self.radius), (0, 128, 255), 2)
        for p in self.prevPts:
            cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0, 200, 100), -1)
        tmp = len(self.prevPts)  
        cv2.putText(
            frame,
            f"angle: {self.smoothedAngle:.2f} deg ({self.motionType})",
            (12, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"pts: {tmp}",
            (12, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Steering wheel angle")
    parser.add_argument("--video", dest="video", required=True, help="Path to video (or 0 for camera)")
    parser.add_argument("--debug", dest="debug", action="store_true", help="Debug")
    parser.add_argument("--no-manual", dest="noManual", action="store_true", help="Disable manual click fallback")
    args = parser.parse_args()

    videoSource = args.video

    if videoSource.isdigit():
        videoSource = int(videoSource)

    config = TrackerConfig(
        allowManualClick=not args.noManual,
        debugMode=args.debug,
    )
    tracker = SteeringLKTracker(config)
    tracker.run(videoSource)


if __name__ == "__main__":
    main()
