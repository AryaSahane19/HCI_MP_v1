#!/usr/bin/env python3
"""
Optimized Gesture Controller using OpenCV, Mediapipe, and PyAutoGUI.

Features:
- Hand gesture recognition using Mediapipe landmarks.
- Mapped gestures (using an IntEnum) to various mouse and system controls.
- Smoother cursor control implemented with an exponential moving average.
- Detailed comments and error handling for unexpected gesture values.
"""

import cv2
import mediapipe as mp
import pyautogui
import math
from enum import IntEnum
from ctypes import cast, POINTER
import comtypes
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from google.protobuf.json_format import MessageToDict
import screen_brightness_control as sbcontrol
from typing import Optional, Tuple

# Disable PyAutoGUI’s failsafe to allow uninterrupted control
pyautogui.FAILSAFE = False

# Initialize Mediapipe drawing and hand solutions
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# =============================================================================
# GESTURE ENUMERATIONS
# =============================================================================

class Gest(IntEnum):
    FIST = 0
    PINKY = 1
    RING = 2
    MID = 4
    LAST3 = 7
    INDEX = 8
    FIRST2 = 12
    LAST4 = 15
    THUMB = 16    
    PALM = 31

    # Extra gesture mappings
    V_GEST = 33
    TWO_FINGER_CLOSED = 34
    PINCH_MAJOR = 35
    PINCH_MINOR = 36

    # For unknown computed finger states (e.g., 14)
    UNKNOWN = 14

class HLabel(IntEnum):
    MINOR = 0
    MAJOR = 1

# =============================================================================
# HAND GESTURE RECOGNITION CLASS
# =============================================================================

class HandRecog:
    """
    Processes Mediapipe landmarks to recognize hand gestures.
    """
    def __init__(self, hand_label: HLabel):
        self.finger: int = 0
        self.ori_gesture: Gest = Gest.PALM
        self.prev_gesture: Gest = Gest.PALM
        self.frame_count: int = 0
        self.hand_result = None
        self.hand_label: HLabel = hand_label

    def update_hand_result(self, hand_result) -> None:
        self.hand_result = hand_result

    def get_signed_dist(self, point: list) -> float:
        """Compute signed Euclidean distance between two landmarks."""
        sign = 1 if self.hand_result.landmark[point[0]].y < self.hand_result.landmark[point[1]].y else -1
        dx = self.hand_result.landmark[point[0]].x - self.hand_result.landmark[point[1]].x
        dy = self.hand_result.landmark[point[0]].y - self.hand_result.landmark[point[1]].y
        return sign * math.sqrt(dx * dx + dy * dy)

    def get_dist(self, point: list) -> float:
        """Compute Euclidean distance between two landmarks."""
        dx = self.hand_result.landmark[point[0]].x - self.hand_result.landmark[point[1]].x
        dy = self.hand_result.landmark[point[0]].y - self.hand_result.landmark[point[1]].y
        return math.sqrt(dx * dx + dy * dy)

    def get_dz(self, point: list) -> float:
        """Compute absolute difference along the z-axis between two landmarks."""
        return abs(self.hand_result.landmark[point[0]].z - self.hand_result.landmark[point[1]].z)

    def set_finger_state(self) -> None:
        """
        Determine finger state by comparing distances between
        fingertip, middle knuckle, and base.
        """
        if self.hand_result is None:
            return

        points = [[8, 5, 0], [12, 9, 0], [16, 13, 0], [20, 17, 0]]
        self.finger = 0
        for point in points:
            dist = self.get_signed_dist(point[:2])
            dist2 = self.get_signed_dist(point[1:])
            try:
                ratio = round(dist / dist2, 1)
            except ZeroDivisionError:
                ratio = round(dist / 0.01, 1)
            self.finger = self.finger << 1
            if ratio > 0.5:
                self.finger |= 1

    def get_gesture(self) -> Gest:
        """
        Determine the gesture based on the computed finger state.
        Uses a debouncing mechanism to ensure stable recognition.
        """
        if self.hand_result is None:
            return Gest.PALM

        try:
            current_gesture = Gest(self.finger)
        except ValueError:
            current_gesture = Gest.UNKNOWN

        # Custom logic for pinch gestures
        if self.finger in [Gest.LAST3, Gest.LAST4] and self.get_dist([8, 4]) < 0.05:
            current_gesture = Gest.PINCH_MINOR if self.hand_label == HLabel.MINOR else Gest.PINCH_MAJOR
        elif self.finger == Gest.FIRST2:
            dist1 = self.get_dist([8, 12])
            dist2 = self.get_dist([5, 9])
            ratio = dist1 / dist2 if dist2 != 0 else 0
            if ratio > 1.7:
                current_gesture = Gest.V_GEST
            else:
                current_gesture = Gest.TWO_FINGER_CLOSED if self.get_dz([8, 12]) < 0.1 else Gest.MID

        if current_gesture == self.prev_gesture:
            self.frame_count += 1
        else:
            self.frame_count = 0

        self.prev_gesture = current_gesture
        if self.frame_count > 4:
            self.ori_gesture = current_gesture

        return self.ori_gesture

# =============================================================================
# SYSTEM & MOUSE CONTROL CLASS
# =============================================================================

class Controller:
    """
    Maps recognized gestures to mouse and system controls.
    Implements smoother cursor movement using exponential smoothing.
    """
    prev_hand: Optional[Tuple[float, float]] = None
    smoothing_alpha: float = 0.8

    flag: bool = False
    grabflag: bool = False
    pinchmajorflag: bool = False
    pinchminorflag: bool = False

    pinchstartxcoord: Optional[float] = None
    pinchstartycoord: Optional[float] = None
    pinchdirectionflag: Optional[bool] = None
    prevpinchlv: float = 0
    pinchlv: float = 0
    framecount: int = 0
    pinch_threshold: float = 0.3

    @staticmethod
    def getpinchylv(hand_result) -> float:
        return round((Controller.pinchstartycoord - hand_result.landmark[8].y) * 10, 1)

    @staticmethod
    def getpinchxlv(hand_result) -> float:
        return round((hand_result.landmark[8].x - Controller.pinchstartxcoord) * 10, 1)

    @staticmethod
    def changesystembrightness() -> None:
        """Adjust system brightness based on pinch level."""
        brightness = sbcontrol.get_brightness(display=0)
        if isinstance(brightness, list):
            brightness = brightness[0]
        current_brightness = brightness / 100.0
        current_brightness += Controller.pinchlv / 50.0
        current_brightness = max(0.0, min(1.0, current_brightness))
        sbcontrol.fade_brightness(int(100 * current_brightness), start=brightness)

    @staticmethod
    def changesystemvolume() -> None:
        """Adjust system volume based on pinch level."""
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        current_volume = volume.GetMasterVolumeLevelScalar()
        current_volume += Controller.pinchlv / 50.0
        current_volume = max(0.0, min(1.0, current_volume))
        volume.SetMasterVolumeLevelScalar(current_volume, None)

    @staticmethod
    def scrollVertical() -> None:
        pyautogui.scroll(120 if Controller.pinchlv > 0.0 else -120)

    @staticmethod
    def scrollHorizontal() -> None:
        pyautogui.keyDown('shift')
        pyautogui.keyDown('ctrl')
        pyautogui.scroll(-120 if Controller.pinchlv > 0.0 else 120)
        pyautogui.keyUp('ctrl')
        pyautogui.keyUp('shift')

    @classmethod
    def get_position(cls, hand_result) -> Tuple[int, int]:
        """Compute the (smoothed) cursor position from hand landmarks."""
        x_target = hand_result.landmark[9].x * pyautogui.size()[0]
        y_target = hand_result.landmark[9].y * pyautogui.size()[1]
        
        if cls.prev_hand is None:
            cls.prev_hand = (x_target, y_target)
            return int(x_target), int(y_target)
        
        alpha = cls.smoothing_alpha
        x_smoothed = alpha * cls.prev_hand[0] + (1 - alpha) * x_target
        y_smoothed = alpha * cls.prev_hand[1] + (1 - alpha) * y_target

        cls.prev_hand = (x_smoothed, y_smoothed)
        return int(x_smoothed), int(y_smoothed)

    @staticmethod
    def pinch_control_init(hand_result) -> None:
        Controller.pinchstartxcoord = hand_result.landmark[8].x
        Controller.pinchstartycoord = hand_result.landmark[8].y
        Controller.pinchlv = 0
        Controller.prevpinchlv = 0
        Controller.framecount = 0

    @staticmethod
    def pinch_control(hand_result, controlHorizontal, controlVertical) -> None:
        if Controller.framecount == 5:
            Controller.framecount = 0
            Controller.pinchlv = Controller.prevpinchlv
            if Controller.pinchdirectionflag:
                controlHorizontal()
            else:
                controlVertical()
        lvx = Controller.getpinchxlv(hand_result)
        lvy = Controller.getpinchylv(hand_result)
        if abs(lvy) > abs(lvx) and abs(lvy) > Controller.pinch_threshold:
            Controller.pinchdirectionflag = False
            if abs(Controller.prevpinchlv - lvy) < Controller.pinch_threshold:
                Controller.framecount += 1
            else:
                Controller.prevpinchlv = lvy
                Controller.framecount = 0
        elif abs(lvx) > Controller.pinch_threshold:
            Controller.pinchdirectionflag = True
            if abs(Controller.prevpinchlv - lvx) < Controller.pinch_threshold:
                Controller.framecount += 1
            else:
                Controller.prevpinchlv = lvx
                Controller.framecount = 0

    @staticmethod
    def handle_controls(gesture: Gest, hand_result) -> None:
        if gesture != Gest.PALM:
            x, y = Controller.get_position(hand_result)
        else:
            x = y = None

        if gesture != Gest.FIST and Controller.grabflag:
            Controller.grabflag = False
            pyautogui.mouseUp(button="left")
        if gesture != Gest.PINCH_MAJOR and Controller.pinchmajorflag:
            Controller.pinchmajorflag = False
        if gesture != Gest.PINCH_MINOR and Controller.pinchminorflag:
            Controller.pinchminorflag = False

        if gesture == Gest.V_GEST:
            Controller.flag = True
            pyautogui.moveTo(x, y, duration=0.1)
        elif gesture == Gest.FIST:
            if not Controller.grabflag:
                Controller.grabflag = True
                pyautogui.mouseDown(button="left")
            pyautogui.moveTo(x, y, duration=0.1)
        elif gesture == Gest.MID and Controller.flag:
            pyautogui.click()
            Controller.flag = False
        elif gesture == Gest.INDEX and Controller.flag:
            pyautogui.click(button='right')
            Controller.flag = False
        elif gesture == Gest.TWO_FINGER_CLOSED and Controller.flag:
            pyautogui.doubleClick()
            Controller.flag = False
        elif gesture == Gest.PINCH_MINOR:
            if not Controller.pinchminorflag:
                Controller.pinch_control_init(hand_result)
                Controller.pinchminorflag = True
            Controller.pinch_control(hand_result, Controller.scrollHorizontal, Controller.scrollVertical)
        elif gesture == Gest.PINCH_MAJOR:
            if not Controller.pinchmajorflag:
                Controller.pinch_control_init(hand_result)
                Controller.pinchmajorflag = True
            Controller.pinch_control(hand_result, Controller.changesystembrightness, Controller.changesystemvolume)

# =============================================================================
# MAIN GESTURE CONTROLLER CLASS
# =============================================================================

class GestureController:
    """
    Captures video, processes hand landmarks using Mediapipe, and invokes controls.
    """
    gc_mode: int = 0
    cap: Optional[cv2.VideoCapture] = None
    hr_major = None
    hr_minor = None
    dom_hand: bool = True  # True: right hand as dominant; False: left hand as dominant

    def __init__(self):
        GestureController.gc_mode = 1
        self.cap = cv2.VideoCapture(0)

    @staticmethod
    def classify_hands(results) -> None:
        left = right = None
        if results.multi_handedness:
            for idx, hand_handedness in enumerate(results.multi_handedness):
                handedness_dict = MessageToDict(hand_handedness)
                label = handedness_dict.get('classification', [{}])[0].get('label', '')
                if label == 'Right':
                    right = results.multi_hand_landmarks[idx]
                elif label == 'Left':
                    left = results.multi_hand_landmarks[idx]
        if GestureController.dom_hand:
            GestureController.hr_major = right
            GestureController.hr_minor = left
        else:
            GestureController.hr_major = left
            GestureController.hr_minor = right

    def start(self) -> None:
        handmajor = HandRecog(HLabel.MAJOR)
        handminor = HandRecog(HLabel.MINOR)

        with mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as hands:
            while self.cap.isOpened() and GestureController.gc_mode:
                success, image = self.cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue

                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = hands.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.multi_hand_landmarks:
                    GestureController.classify_hands(results)
                    if GestureController.hr_major:
                        handmajor.update_hand_result(GestureController.hr_major)
                        handmajor.set_finger_state()
                    if GestureController.hr_minor:
                        handminor.update_hand_result(GestureController.hr_minor)
                        handminor.set_finger_state()

                    # Prioritize pinch gesture from the minor hand.
                    if GestureController.hr_minor and handminor.get_gesture() == Gest.PINCH_MINOR:
                        Controller.handle_controls(Gest.PINCH_MINOR, handminor.hand_result)
                    elif GestureController.hr_major:
                        gesture = handmajor.get_gesture()
                        Controller.handle_controls(gesture, handmajor.hand_result)

                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                else:
                    Controller.prev_hand = None

                cv2.imshow('Gesture Controller', image)
                if cv2.waitKey(5) & 0xFF == 13:  # Press Enter to exit
                    break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    gc = GestureController()
    gc.start()
