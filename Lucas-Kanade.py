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
import json
import time
from datetime import datetime
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

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
    minAngleSamples: int = 8
    SmoothAlpha: float = 0.15
    nearZeroThresholdDeg: float = 0.05
    maxAngleStdDeg: float = 6.0
    GradientJumpThreshold: float = 12.0
    motionAngleThresholdDeg: float = 9.0  #T_θ in paper
    MotionDerivativeThresholdDeg: float = 3.0  #T_Δθ
    derivativeWindow: int = 5
    alignBackToZeroDeg: float = 0.8
    allowManualClick: bool = True
    debugMode: bool = False
    maskOuterFraction: float = 0.97
    maskInnerFraction: float = 0.52
    debugPrintEvery: int = 30
    debugHistorySize: int = 180
    debugHistogramRangeDeg: Tuple[float, float] = (-30.0, 30.0)
    debugDumpDir: Optional[str] = "debug_dump"
    debugDumpEvery: int = 5
    wheelRefreshIntervalFrames: int = 360
    wheelMaxCenterShiftPx: float = 28.0
    wheelMaxRadiusScaleChange: float = 0.12
    rejectSkinFeatures: bool = True
    skinHueRangeLow: Tuple[int, int] = (0, 25)  #inclusive
    skinHueRangeHigh: Tuple[int, int] = (160, 180)
    minSkinSaturation: int = 40
    maxSkinValue: int = 255
    showWindows: bool = True
    alignZeroFrameThreshold: int = 45
    alignZeroMaxAbsAngle: float = 120.0


class SteeringLKTracker:
    def __init__(self, config: TrackerConfig):
        self.cfg = config
        self.center: Optional[np.ndarray] = None
        self.radius: Optional[float] = None
        self.mask: Optional[np.ndarray] = None
        self.innerRadius: Optional[float] = None
        self.outerRadius: Optional[float] = None

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
        historySize = max(1, self.cfg.debugHistorySize)
        self.debugHistory: Deque["SteeringLKTracker.DebugFrameStats"] = deque(maxlen=historySize)
        self.latestDebugStats: Optional["SteeringLKTracker.DebugFrameStats"] = None
        if self.cfg.debugMode and self.cfg.debugDumpDir:
            dumpPath = os.path.abspath(self.cfg.debugDumpDir)
            os.makedirs(dumpPath, exist_ok=True)
            self._debugDumpDir: Optional[str] = dumpPath
        else:
            self._debugDumpDir = None
        self.maxFrames: Optional[int] = None
        self.lastWheelRefreshFrame: int = 0
        self.rectZeroStreak: int = 0

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
            self.innerRadius = None
            self.outerRadius = None
        self.lastPeriodicResetFrame = self.frameIdx
        self.lastWheelRefreshFrame = self.frameIdx
        self.rectZeroStreak = 0
        self.latestDebugStats = None
        if self.cfg.debugMode:
            self.debugHistory.clear()

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

    def run(
        self,
        videoSource: str | int,
        maxFrames: Optional[int] = None,
        outputPath: Optional[str] = None,
    ) -> None:
        cap: cv2.VideoCapture
        if isinstance(videoSource, int):
            cap = cv2.VideoCapture(videoSource, cv2.CAP_V4L2)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(videoSource)
        else:
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
        requestedFps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        writer: Optional[cv2.VideoWriter] = None
        writerPath: Optional[str] = None
        writerBuffer: List[np.ndarray] = []
        writerStartTime: Optional[float] = None
        writerLastTime: Optional[float] = None
        writerFrameCount: int = 0
        measuredFps: Optional[float] = None
        #TODO(fixlater) usb camera lies about fps sometimes but i eyeball it.

        ok, frame = cap.read()
        if not ok:
            retries = 3 if isinstance(videoSource, int) else 0
            for attempt in range(retries):
                time.sleep(0.05)
                ok, frame = cap.read()
                if ok:
                    break
            if not ok:
                print("Vid empty.", file=sys.stderr)
                cap.release()
                sys.exit(1)

        if outputPath is not None and isinstance(videoSource, int):
            try:
                writerPath = os.path.abspath(outputPath)
                outputDir = os.path.dirname(writerPath)
                os.makedirs(outputDir, exist_ok=True)
            except OSError as writerErr:
                print(f"Cannot prepare video writer directory: {writerErr}", file=sys.stderr)
                writerPath = None

        def _record_frame(frame_to_write: np.ndarray) -> None:
            nonlocal writer, writerBuffer, writerStartTime, writerFrameCount, measuredFps, writerPath, writerLastTime
            if writerPath is None:
                return
            now = time.perf_counter()
            if writerStartTime is None:
                writerStartTime = now
            writerLastTime = now
            writerFrameCount += 1

            if writer is None:
                writerBuffer.append(frame_to_write)
                elapsed = max(now - writerStartTime, 0.0)
                enoughFrames = writerFrameCount >= 30 and elapsed >= 1.0
                tooLong = elapsed >= 3.0
                if not (enoughFrames or tooLong):
                    return
                measuredFps = max(writerFrameCount / max(elapsed, 1e-6), 1.0)
                frameHeight, frameWidth = frame_to_write.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(
                    writerPath,
                    fourcc,
                    float(measuredFps),
                    (frameWidth, frameHeight),
                )
                if not writer.isOpened():
                    print(f"Cannot open video writer: {writerPath}", file=sys.stderr)
                    writer.release()
                    writer = None
                    writerPath = None
                    writerBuffer.clear()
                    return
                for buffered in writerBuffer:
                    writer.write(buffered)
                writerBuffer.clear()
                if self.cfg.debugMode:
                    print(
                        f"[dbg] video writer initialised at {measuredFps:.3f} fps "
                        f"({writerFrameCount} frames buffered)"
                    )
            else:
                writer.write(frame_to_write)

        self._initialise_from_first_frame(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.prevGray = gray
        self.prevPts = self._detect_features(gray, useMask=True)
        self.prevPts = self.prevPts.reshape(-1, 2) if self.prevPts is not None else np.empty((0, 2))
        self.frameIdx = 0
        self.maxFrames = maxFrames
        self.angleDeg = 0.0
        self.smoothedAngle = 0.0
        self.motionType = "rect"
        self.lastRelativeAngle = 0.0
        self.angleHistory = [0.0]
        self.lastReseedFrame = 0
        self.periodicResetIntervalFrames = max(1, int(round(requestedFps * 5.0)))

        #Can't get this to work for long periods of time. Resetting every 5 secs makes it work well enough.
        quickResetTick = self.periodicResetIntervalFrames  
        if self.cfg.debugMode and quickResetTick < 240:
            print("resetTick?", quickResetTick)
        self.lastPeriodicResetFrame = 0

        initialRender = self._draw_overlay(frame.copy(), measurement=None)
        _record_frame(initialRender)

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            self.frameIdx += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            measurement = self._process_frame(gray, frame=frame)
            self._apply_measurement(measurement)
            self._maybe_reseed(gray, force=measurement.forceReseed)
            self._maybe_refresh_wheel_geometry(frame)

            needsRender = (writerPath is not None and (writer is not None or writerBuffer)) or self.cfg.showWindows
            renderFrame: Optional[np.ndarray] = None
            if needsRender:
                renderFrame = self._draw_overlay(frame.copy(), measurement)

            if renderFrame is not None:
                _record_frame(renderFrame)

            if self.cfg.showWindows and renderFrame is not None:
                cv2.imshow("steering-lk", renderFrame)
            elif self.cfg.showWindows:
                cv2.imshow("steering-lk", frame)

            if measurement.valid:
                self._writeSharedAngle(float(self.smoothedAngle))
            else:
                self._writeSharedAngle(float("nan"))

            self.prevGray = gray
            if (
                measurement.nextPoints is not None
                and len(measurement.nextPoints) >= self.cfg.minTrackedPoints
            ):
                self.prevPts = measurement.nextPoints.reshape(-1, 2)

            if self.maxFrames is not None and self.frameIdx >= self.maxFrames:
                if self.cfg.debugMode:
                    print(f"[dbg] reached max frame limit {self.maxFrames}")
                break

            if self.cfg.showWindows:
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                if key == ord("r"):
                    self.resetDetection(resetAngle=True, wheelDetectionReset=True, currentFrame=frame)

        cap.release()
        if writerPath is not None and writer is None and writerBuffer:
            finalNow = time.perf_counter()
            if writerStartTime is None:
                writerStartTime = finalNow
            if writerLastTime is None:
                writerLastTime = finalNow
            totalElapsed = max(writerLastTime - writerStartTime, 1e-6)
            measuredFps = max(writerFrameCount / totalElapsed, 1.0)
            frameHeight, frameWidth = writerBuffer[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(
                writerPath,
                fourcc,
                float(measuredFps),
                (frameWidth, frameHeight),
            )
            if writer.isOpened():
                for buffered in writerBuffer:
                    writer.write(buffered)
                writerBuffer.clear()
            else:
                print(f"Cannot open video writer: {writerPath}", file=sys.stderr)
                writer.release()
                writer = None
        if writer is not None:
            writer.release()
            if writerStartTime is not None and writerLastTime is not None and writerFrameCount > 0:
                totalElapsed = max(writerLastTime - writerStartTime, 1e-6)
                finalFps = writerFrameCount / totalElapsed
                print(
                    f"[record] saved {writerFrameCount} frames over {totalElapsed:.2f}s "
                    f"(~{finalFps:.3f} fps metadata)"
                )
        if self.cfg.showWindows:
            cv2.destroyAllWindows()
        self._writeSharedAngle(float("nan"))  #tell downstream we're idle now.
        self._closeSharedMem()

    def _initialise_from_first_frame(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        print("Hough circle start...") 
        blurred = cv2.medianBlur(gray, 5)
        detectionSource = "hough"
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
            detectionSource = "manual"
        else:
            self.center = np.array([w / 2.0, h / 2.0], dtype=np.float32)
            self.radius = min(h, w) * 0.25
            print("Automatic detection failed; using frame centre heuristic.")
            detectionSource = "fallback"

        if self.center is None or self.radius is None:
            self.center = np.array([w / 2.0, h / 2.0], dtype=np.float32)
            self.radius = min(h, w) * 0.25
            print("Falling back to frame centre due to missing clicks.")
            detectionSource = "fallback"

        self._rebuild_mask(frameShape=(h, w), stage="initial", detectionSource=detectionSource)

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

    def _rebuild_mask(self, frameShape: Tuple[int, int], stage: str, detectionSource: str) -> None:
        if self.center is None or self.radius is None:
            return
        self.outerRadius = float(self.radius * self.cfg.maskOuterFraction)
        self.innerRadius = float(self.radius * self.cfg.maskInnerFraction)
        if self.innerRadius >= self.outerRadius:
            self.innerRadius = max(self.outerRadius * 0.5, 1.0)

        h, w = frameShape
        blank = np.zeros((h, w), dtype=np.uint8)
        self.mask = blank
        outerR = max(1, int(round(self.outerRadius)))
        innerR = max(1, int(round(self.innerRadius)))
        innerR = min(innerR, max(outerR - 2, 1))
        cv2.circle(self.mask, (int(self.center[0]), int(self.center[1])), outerR, 255, -1)
        cv2.circle(self.mask, (int(self.center[0]), int(self.center[1])), innerR, 0, -1)
        self._dump_wheel_geometry(stage=stage, detectionSource=detectionSource, frameShape=frameShape)

    def _dump_wheel_geometry(self, stage: str, detectionSource: str, frameShape: Tuple[int, int]) -> None:
        if not self.cfg.debugMode or self._debugDumpDir is None:
            return
        data = {
            "frameIdx": int(self.frameIdx),
            "stage": stage,
            "detectionSource": detectionSource,
            "center": self.center.tolist() if self.center is not None else None,
            "radius": float(self.radius) if self.radius is not None else None,
            "innerRadius": float(self.innerRadius) if self.innerRadius is not None else None,
            "outerRadius": float(self.outerRadius) if self.outerRadius is not None else None,
            "maskOuterFraction": float(self.cfg.maskOuterFraction),
            "maskInnerFraction": float(self.cfg.maskInnerFraction),
            "frameShape": [int(frameShape[0]), int(frameShape[1])],
        }
        path = os.path.join(
            self._debugDumpDir,
            f"wheel_geometry_{stage}_{self.frameIdx:06d}.json",
        )
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, separators=(",", ":"))
        except OSError as dumpErr:
            if self.cfg.debugMode:
                print(f"[dbg] failed to dump wheel geometry: {dumpErr}")

    @dataclass
    class Measurement:
        valid: bool
        relativeAngle: float
        nextPoints: np.ndarray
        trackedPoints: int
        rawTrackedPoints: int
        forceReseed: bool
        motionType: str
        RawCandidates: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.float32))
        invalidReason: Optional[str] = None
        pointInsideMask: Optional[np.ndarray] = None
        pointRadii: Optional[np.ndarray] = None
        appearanceAccepted: Optional[np.ndarray] = None
        maskRejectedCount: int = 0
        appearanceRejectedCount: int = 0
        medianAngle: float = 0.0

    @dataclass
    class DebugFrameStats:
        frameIdx: int
        trackedPoints: int
        rawTrackedPoints: int
        inMask: int
        outOfMask: int
        maskRejected: int
        appearanceRejected: int
        relativeAngle: float
        smoothedAngle: float
        motionType: str
        measurementValid: bool
        invalidReason: Optional[str]
        angleMedian: Optional[float]
        angleStd: Optional[float]
        sampleCount: int
        positiveCount: int
        negativeCount: int
        noiseFlag: bool

    def _process_frame(self, gray: np.ndarray, frame: Optional[np.ndarray] = None) -> "SteeringLKTracker.Measurement":
        if self.prevGray is None:
            measurement = self.Measurement(
                valid=False,
                relativeAngle=0.0,
                nextPoints=self.prevPts,
                trackedPoints=len(self.prevPts),
                rawTrackedPoints=len(self.prevPts),
                forceReseed=True,
                motionType=self.motionType,
                invalidReason="bootstrap",
                medianAngle=0.0,
            )
            self._record_frame_debug(
                measurement=measurement,
                stats=None,
                angleDeltas=None,
                inMask=None,
                noiseFlag=False,
            )
            return measurement

        nextPts, prevPtsValid = self._track_points(self.prevGray, gray, self.prevPts)
        rawTrackedPoints = len(nextPts)
        inMaskFlags, radii = self._split_points_by_annulus(nextPts)

        maskRejectedCount = int((~inMaskFlags).sum()) if inMaskFlags is not None and len(inMaskFlags) else 0
        if inMaskFlags is not None and len(inMaskFlags):
            nextPts = nextPts[inMaskFlags]
            prevPtsValid = prevPtsValid[inMaskFlags]
            radii = radii[inMaskFlags]
            inMaskFlags = np.ones(len(nextPts), dtype=bool)

        if frame is not None and len(nextPts) > 0:
            appearanceMask = self._filter_points_by_appearance(frame, nextPts)
            appearanceRejectedCount = int((~appearanceMask).sum())
            if appearanceRejectedCount:
                nextPts = nextPts[appearanceMask]
                prevPtsValid = prevPtsValid[appearanceMask]
                radii = radii[appearanceMask]
            appearanceAccepted = np.ones(len(nextPts), dtype=bool)
        else:
            appearanceMask = None
            appearanceRejectedCount = 0
            appearanceAccepted = np.ones(len(nextPts), dtype=bool)

        trackedPoints = len(nextPts)

        if trackedPoints < self.cfg.minTrackedPoints:
            measurement = self.Measurement(
                valid=False,
                relativeAngle=0.0,
                nextPoints=nextPts,
                trackedPoints=trackedPoints,
                rawTrackedPoints=rawTrackedPoints,
                forceReseed=True,
                motionType=self.motionType,
                invalidReason="few-tracks",
                pointInsideMask=inMaskFlags,
                pointRadii=radii,
                appearanceAccepted=appearanceAccepted,
                maskRejectedCount=maskRejectedCount,
                appearanceRejectedCount=appearanceRejectedCount,
                medianAngle=0.0,
            )
            self._record_frame_debug(
                measurement=measurement,
                stats=None,
                angleDeltas=None,
                inMask=inMaskFlags,
                noiseFlag=False,
            )
            return measurement

        angleDeltas = self._compute_angleDeltas(prevPtsValid, nextPts)
        stats = self._angle_statistics(angleDeltas)
        noiseFlag = self._detect_horizontal_noise(self.prevGray, gray)
        motionType = self._classify_motion(stats, predictedAngle=self.angleDeg + stats.median)
        relativeAngle, measureValid = self._decide_relativeAngle(stats, motionType, noiseFlag)

        invalidReason = None
        if not measureValid:
            if stats.sampleCount < self.cfg.minAngleSamples:
                invalidReason = "few-angle-samples"
            elif stats.std > self.cfg.maxAngleStdDeg:
                invalidReason = "angle-std"
            elif noiseFlag:
                invalidReason = "noise-spike"
            else:
                invalidReason = "rule-reject"

        measurement = self.Measurement(
            valid=measureValid,
            relativeAngle=relativeAngle,
            nextPoints=nextPts,
            trackedPoints=trackedPoints,
            rawTrackedPoints=rawTrackedPoints,
            forceReseed=trackedPoints < self.cfg.minTrackedPoints * 1.5,
            motionType=motionType,
            RawCandidates=angleDeltas,
            invalidReason=invalidReason,
            pointInsideMask=inMaskFlags,
            pointRadii=radii,
            appearanceAccepted=appearanceAccepted,
            maskRejectedCount=maskRejectedCount,
            appearanceRejectedCount=appearanceRejectedCount,
            medianAngle=stats.median,
        )
        self._record_frame_debug(
            measurement=measurement,
            stats=stats,
            angleDeltas=angleDeltas,
            inMask=inMaskFlags,
            noiseFlag=noiseFlag,
        )
        return measurement

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

        if measurement.trackedPoints >= self.cfg.minAngleSamples:
            if abs(measurement.medianAngle) < self.cfg.alignBackToZeroDeg:
                self.rectZeroStreak += 1
            else:
                self.rectZeroStreak = 0
        else:
            self.rectZeroStreak = 0

        aligned = False
        if (
            self.rectZeroStreak >= self.cfg.alignZeroFrameThreshold
            and abs(self.angleDeg) > self.cfg.alignBackToZeroDeg
            and abs(self.angleDeg) <= self.cfg.alignZeroMaxAbsAngle
        ):
            if self.cfg.debugMode:
                print(
                    f"[dbg] align-to-zero at frame {self.frameIdx}: angle {self.angleDeg:.2f} deg"
                )
            self.angleDeg = 0.0
            self.smoothedAngle = 0.0
            self.lastRelativeAngle = 0.0
            self.rectZeroStreak = 0
            self.angleHistory = [0.0]
            aligned = True

        if not aligned:
            self.angleHistory.append(self.angleDeg)

        self.motionType = measurement.motionType
        #TODO revisit smoothing once I trust calibration again.
        #alpha = 0.18
        alpha = self.cfg.SmoothAlpha
        self.smoothedAngle = alpha * self.angleDeg + (1.0 - alpha) * self.smoothedAngle
        if self.cfg.debugMode and self.latestDebugStats is not None:
            self.latestDebugStats.relativeAngle = float(self.lastRelativeAngle)
            self.latestDebugStats.smoothedAngle = float(self.smoothedAngle)
            self.latestDebugStats.motionType = self.motionType
            self.latestDebugStats.measurementValid = bool(measurement.valid)
            self.latestDebugStats.invalidReason = measurement.invalidReason

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

    def _maybe_refresh_wheel_geometry(self, frame: np.ndarray) -> None:
        if self.cfg.wheelRefreshIntervalFrames <= 0:
            return
        if self.center is None or self.radius is None:
            return
        if self.frameIdx - self.lastWheelRefreshFrame < self.cfg.wheelRefreshIntervalFrames:
            return

        self.lastWheelRefreshFrame = self.frameIdx

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 5)
        h, w = gray.shape[:2]
        cx, cy = self.center
        roiPad = int(self.radius * 1.4)
        x0 = max(int(cx - roiPad), 0)
        y0 = max(int(cy - roiPad), 0)
        x1 = min(int(cx + roiPad), w)
        y1 = min(int(cy + roiPad), h)
        roi = blurred[y0:y1, x0:x1]

        if roi.size == 0:
            return

        minRadius = int(max(20.0, self.radius * (1.0 - self.cfg.wheelMaxRadiusScaleChange)))
        maxRadius = int(min(min(h, w) * 0.6, self.radius * (1.0 + self.cfg.wheelMaxRadiusScaleChange)))
        if minRadius >= maxRadius:
            maxRadius = minRadius + 5

        circles = cv2.HoughCircles(
            roi,
            cv2.HOUGH_GRADIENT,
            dp=1.1,
            minDist=80,
            param1=100,
            param2=30,
            minRadius=minRadius,
            maxRadius=maxRadius,
        )
        if circles is None or len(circles) == 0:
            if self.cfg.debugMode:
                print(f"[dbg] wheel refresh failed: no circles at frame {self.frameIdx}")
            return

        cand = circles[0, 0]
        newCenter = np.array([cand[0] + x0, cand[1] + y0], dtype=np.float32)
        newRadius = float(cand[2])

        shift = float(np.linalg.norm(newCenter - self.center))
        radiusDiff = abs(newRadius - self.radius) / max(self.radius, 1e-3)

        if shift > self.cfg.wheelMaxCenterShiftPx or radiusDiff > self.cfg.wheelMaxRadiusScaleChange * 1.5:
            if self.cfg.debugMode:
                print(
                    f"[dbg] wheel refresh rejected: shift {shift:.1f}px radiusDelta {radiusDiff:.3f} at frame {self.frameIdx}"
                )
            return

        self.center = newCenter
        self.radius = newRadius
        self._rebuild_mask(frameShape=(h, w), stage="refresh", detectionSource="hough-refresh")
        if self.cfg.debugMode:
            print(
                f"[dbg] wheel refresh accepted at frame {self.frameIdx}: center shift {shift:.1f}px radius {newRadius:.1f}"
            )

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

    def _split_points_by_annulus(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if points is None or len(points) == 0:
            return np.zeros((0,), dtype=bool), np.zeros((0,), dtype=np.float32)
        if (
            self.center is None
            or self.innerRadius is None
            or self.outerRadius is None
        ):
            return np.ones(len(points), dtype=bool), np.linalg.norm(points, axis=1).astype(np.float32)
        radii = np.linalg.norm(points - self.center, axis=1)
        inMask = (radii >= self.innerRadius) & (radii <= self.outerRadius)
        return inMask.astype(bool), radii.astype(np.float32)

    def _filter_points_by_appearance(self, frame: np.ndarray, points: np.ndarray) -> np.ndarray:
        if points is None or len(points) == 0:
            return np.ones((0,), dtype=bool)
        if not self.cfg.rejectSkinFeatures:
            return np.ones(len(points), dtype=bool)

        hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hLow0, hHigh0 = self.cfg.skinHueRangeLow
        hLow1, hHigh1 = self.cfg.skinHueRangeHigh
        satMin = self.cfg.minSkinSaturation
        valMax = self.cfg.maxSkinValue

        mask = np.ones(len(points), dtype=bool)
        height, width = hsvFrame.shape[:2]

        for idx, pt in enumerate(points):
            x = int(round(pt[0]))
            y = int(round(pt[1]))
            x0 = max(x - 2, 0)
            y0 = max(y - 2, 0)
            x1 = min(x + 3, width)
            y1 = min(y + 3, height)
            if x0 >= x1 or y0 >= y1:
                continue

            patch = hsvFrame[y0:y1, x0:x1]
            if patch.size == 0:
                continue

            h = patch[:, :, 0]
            s = patch[:, :, 1]
            v = patch[:, :, 2]
            cond0 = (h >= hLow0) & (h <= hHigh0)
            cond1 = (h >= hLow1) & (h <= hHigh1)
            hueMask = cond0 | cond1
            satMask = s >= satMin
            valMask = v <= valMax
            skinPixels = hueMask & satMask & valMask

            if np.mean(skinPixels) > 0.35:
                mask[idx] = False

        return mask

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
            sampleCount = int(deltaAngles.size)
            return self.AngleStats(
                median=0.0,
                std=0.0,
                positiveMean=None,
                negativeMean=None,
                positiveCount=0,
                negativeCount=0,
                sampleCount=sampleCount,
            )

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

    def _record_frame_debug(
        self,
        measurement: "SteeringLKTracker.Measurement",
        stats: Optional["SteeringLKTracker.AngleStats"],
        angleDeltas: Optional[np.ndarray],
        inMask: Optional[np.ndarray],
        noiseFlag: bool,
    ) -> None:
        if not self.cfg.debugMode:
            return

        if inMask is None and measurement.nextPoints is not None:
            inMask, _ = self._split_points_by_annulus(measurement.nextPoints)

        tracked = int(measurement.trackedPoints)
        rawTracked = int(getattr(measurement, "rawTrackedPoints", tracked))
        inCount = int(inMask.sum()) if inMask is not None else tracked
        outCount = int(getattr(measurement, "maskRejectedCount", 0))
        appearanceRejected = int(getattr(measurement, "appearanceRejectedCount", 0))

        angleMedian = stats.median if stats is not None else None
        angleStd = stats.std if stats is not None else None
        positiveCount = stats.positiveCount if stats is not None else 0
        negativeCount = stats.negativeCount if stats is not None else 0
        sampleCount = stats.sampleCount if stats is not None else 0

        debugEntry = self.DebugFrameStats(
            frameIdx=self.frameIdx,
            trackedPoints=tracked,
            rawTrackedPoints=rawTracked,
            inMask=inCount,
            outOfMask=outCount,
            maskRejected=outCount,
            appearanceRejected=appearanceRejected,
            relativeAngle=float(measurement.relativeAngle),
            smoothedAngle=float(self.smoothedAngle),
            motionType=measurement.motionType,
            measurementValid=measurement.valid,
            invalidReason=measurement.invalidReason,
            angleMedian=angleMedian,
            angleStd=angleStd,
            sampleCount=sampleCount,
            positiveCount=positiveCount,
            negativeCount=negativeCount,
            noiseFlag=noiseFlag,
        )
        self.latestDebugStats = debugEntry
        self.debugHistory.append(debugEntry)

        reason = measurement.invalidReason or "ok"
        if (
            self.cfg.debugPrintEvery > 0
            and self.frameIdx % self.cfg.debugPrintEvery == 0
        ):
            print(
                f"[dbg] frame {debugEntry.frameIdx:05d} "
                f"tracked={tracked} raw={rawTracked} in={inCount} maskRej={outCount} skinRej={appearanceRejected} "
                f"valid={measurement.valid} reason={reason} "
                f"angle={measurement.relativeAngle:.2f} std={angleStd if angleStd is not None else float('nan'):.2f} "
                f"samples={sampleCount} noise={noiseFlag}"
            )

        self._show_angle_histogram(angleDeltas)
        self._dump_debug_frame(
            measurement=measurement,
            stats=stats,
            angleDeltas=angleDeltas,
        )

    def _show_angle_histogram(self, angleDeltas: Optional[np.ndarray]) -> None:
        if not self.cfg.debugMode or not self.cfg.showWindows:
            return

        histHeight = 120
        histWidth = 240
        canvas = np.zeros((histHeight, histWidth, 3), dtype=np.uint8)

        if angleDeltas is None or len(angleDeltas) == 0:
            cv2.putText(
                canvas,
                "no angle data",
                (12, histHeight // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (160, 160, 160),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow("debug-angle-histogram", canvas)
            return

        rangeMin, rangeMax = self.cfg.debugHistogramRangeDeg
        bins = histWidth // 4
        clipped = np.clip(angleDeltas, rangeMin, rangeMax)
        hist, _ = np.histogram(clipped, bins=bins, range=(rangeMin, rangeMax))
        hist = hist.astype(np.float32)
        maxCount = float(hist.max()) if hist.size else 1.0
        if maxCount <= 0.0:
            maxCount = 1.0

        for idx, count in enumerate(hist):
            barHeight = int((count / maxCount) * (histHeight - 20))
            x0 = idx * 4
            cv2.rectangle(
                canvas,
                (x0, histHeight - 10),
                (x0 + 3, histHeight - 10 - barHeight),
                (0, 200, 255),
                -1,
            )

        cv2.line(canvas, (0, histHeight - 10), (histWidth - 1, histHeight - 10), (90, 90, 90), 1)
        cv2.putText(
            canvas,
            f"[{rangeMin:.0f},{rangeMax:.0f}] deg",
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow("debug-angle-histogram", canvas)

    def _dump_debug_frame(
        self,
        measurement: "SteeringLKTracker.Measurement",
        stats: Optional["SteeringLKTracker.AngleStats"],
        angleDeltas: Optional[np.ndarray],
    ) -> None:
        if (
            not self.cfg.debugMode
            or self._debugDumpDir is None
            or self.cfg.debugDumpEvery <= 0
            or (self.frameIdx % self.cfg.debugDumpEvery) != 0
        ):
            return

        pointsSource = measurement.nextPoints
        if pointsSource is None or len(pointsSource) == 0:
            pointsSource = self.prevPts
        points = pointsSource.reshape(-1, 2) if pointsSource is not None else np.empty((0, 2), dtype=np.float32)
        insideMask = measurement.pointInsideMask
        radii = measurement.pointRadii

        pointRecords: List[dict] = []
        appearanceAccepted = measurement.appearanceAccepted
        for idx, pt in enumerate(points):
            insideVal: Optional[bool] = None
            radiusVal: Optional[float] = None
            if insideMask is not None and idx < len(insideMask):
                insideVal = bool(insideMask[idx])
            if radii is not None and idx < len(radii):
                radiusVal = float(radii[idx])
            appearanceVal: Optional[bool] = None
            if appearanceAccepted is not None and idx < len(appearanceAccepted):
                appearanceVal = bool(appearanceAccepted[idx])
            pointRecords.append(
                {
                    "x": float(pt[0]),
                    "y": float(pt[1]),
                    "insideMask": insideVal,
                    "radius": radiusVal,
                    "appearanceAccepted": appearanceVal,
                }
            )

        statsData: Optional[dict]
        if stats is not None:
            statsData = {
                "median": float(stats.median),
                "std": float(stats.std),
                "positiveMean": float(stats.positiveMean) if stats.positiveMean is not None else None,
                "negativeMean": float(stats.negativeMean) if stats.negativeMean is not None else None,
                "positiveCount": int(stats.positiveCount),
                "negativeCount": int(stats.negativeCount),
                "sampleCount": int(stats.sampleCount),
            }
        else:
            statsData = None

        measurementData = {
            "valid": bool(measurement.valid),
            "relativeAngle": float(measurement.relativeAngle),
            "trackedPoints": int(measurement.trackedPoints),
            "rawTrackedPoints": int(getattr(measurement, "rawTrackedPoints", measurement.trackedPoints)),
            "maskRejectedCount": int(getattr(measurement, "maskRejectedCount", 0)),
            "appearanceRejectedCount": int(getattr(measurement, "appearanceRejectedCount", 0)),
            "forceReseed": bool(measurement.forceReseed),
            "motionType": measurement.motionType,
            "invalidReason": measurement.invalidReason,
            "medianAngle": float(getattr(measurement, "medianAngle", 0.0)),
        }

        debugData = {
            "frameIdx": int(self.frameIdx),
            "absoluteAngleDeg": float(self.angleDeg),
            "smoothedAngleDeg": float(self.smoothedAngle),
            "lastRelativeAngleDeg": float(self.lastRelativeAngle),
            "center": self.center.tolist() if self.center is not None else None,
            "radius": float(self.radius) if self.radius is not None else None,
            "innerRadius": float(self.innerRadius) if self.innerRadius is not None else None,
            "outerRadius": float(self.outerRadius) if self.outerRadius is not None else None,
            "points": pointRecords,
            "measurement": measurementData,
            "angleStats": statsData,
            "rawAngleCandidatesDeg": angleDeltas.tolist() if angleDeltas is not None else [],
        }

        path = os.path.join(self._debugDumpDir, f"frame_{self.frameIdx:06d}.json")
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(debugData, fh, separators=(",", ":"))
        except OSError as dumpErr:
            if self.cfg.debugMode:
                print(f"[dbg] failed to dump frame data: {dumpErr}")

    def _draw_overlay(
        self,
        frame: np.ndarray,
        measurement: Optional["SteeringLKTracker.Measurement"] = None,
    ) -> np.ndarray:
        if self.center is not None:
            cv2.circle(frame, (int(self.center[0]), int(self.center[1])), 4, (0, 255, 0), -1)
        if self.radius is not None and self.center is not None:
            cv2.circle(frame, (int(self.center[0]), int(self.center[1])), int(self.radius), (0, 128, 255), 2)

        if measurement is not None and measurement.nextPoints is not None:
            points = measurement.nextPoints.reshape(-1, 2)
            insideMask = measurement.pointInsideMask
        else:
            points = self.prevPts
            insideMask = None

        if insideMask is None and points is not None and len(points) > 0:
            insideMask, _ = self._split_points_by_annulus(points)

        if points is not None:
            for idx, p in enumerate(points):
                inside = True
                if insideMask is not None and idx < len(insideMask):
                    inside = bool(insideMask[idx])
                color = (0, 200, 100) if inside else (0, 0, 255)
                cv2.circle(frame, (int(p[0]), int(p[1])), 3, color, -1)

        trackedCount = (
            int(measurement.trackedPoints)
            if measurement is not None
            else len(self.prevPts)
        )
        rawCount = (
            int(getattr(measurement, "rawTrackedPoints", trackedCount))
            if measurement is not None
            else trackedCount
        )
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

        dbg = self.latestDebugStats
        if dbg is not None:
            cv2.putText(
                frame,
                f"pts: {trackedCount}/{rawCount} (mask {dbg.maskRejected}, skin {dbg.appearanceRejected})",
                (12, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            if dbg.angleStd is not None:
                cv2.putText(
                    frame,
                    f"std: {dbg.angleStd:.2f} deg samples: {dbg.sampleCount}",
                    (12, 84),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            if not dbg.measurementValid and dbg.invalidReason:
                cv2.putText(
                    frame,
                    f"invalid: {dbg.invalidReason}",
                    (12, 108),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 180, 255),
                    1,
                    cv2.LINE_AA,
                )
        else:
            cv2.putText(
                frame,
                f"pts: {trackedCount}",
                (12, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        cv2.putText(
            frame,
            f"frame: {self.frameIdx}",
            (12, 132),
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
    parser.add_argument("--headless", dest="headless", action="store_true", help="Disable GUI windows")
    parser.add_argument("--max-frames", dest="max_frames", type=int, default=None, help="Process at most N frames")
    parser.add_argument(
        "--output",
        dest="output_path",
        default=None,
        help="Directory where webcam recordings are saved using timestamped filenames (MJPG codec)",
    )
    parser.add_argument(
        "--align-zero-frames",
        dest="align_zero_frames",
        type=int,
        default=None,
        help="Override frame streak required before zero realignment (use 0 to disable)",
    )
    parser.add_argument(
        "--align-zero-max-angle",
        dest="align_zero_max_angle",
        type=float,
        default=None,
        help="Override max absolute angle (deg) eligible for zero realignment",
    )
    args = parser.parse_args()

    videoArg = args.video
    isCameraSource = videoArg.isdigit()
    videoSource = int(videoArg) if isCameraSource else videoArg
    outputPath: Optional[str] = None
    if args.output_path and not isCameraSource:
        print("--output flag is only supported with webcam sources; ignoring.", file=sys.stderr)
    elif isCameraSource and args.output_path:
        outputBase = os.path.abspath(args.output_path)
        outputDir = outputBase
        if not os.path.isdir(outputBase):
            baseName = os.path.basename(outputBase)
            root, ext = os.path.splitext(baseName)
            if ext:
                outputDir = os.path.dirname(outputBase) or os.getcwd()
            else:
                outputDir = outputBase
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        outputPath = os.path.join(outputDir, f"{timestamp}.avi")
        if args.debug:
            print(f"[dbg] recording to {outputPath}")

    config = TrackerConfig(
        allowManualClick=not args.noManual,
        debugMode=args.debug,
        showWindows=not args.headless,
    )
    if args.align_zero_frames is not None:
        if args.align_zero_frames <= 0:
            config.alignZeroFrameThreshold = 10**9
        else:
            config.alignZeroFrameThreshold = args.align_zero_frames
    if args.align_zero_max_angle is not None:
        config.alignZeroMaxAbsAngle = args.align_zero_max_angle
    tracker = SteeringLKTracker(config)
    tracker.run(videoSource, maxFrames=args.max_frames, outputPath=outputPath)


if __name__ == "__main__":
    main()
