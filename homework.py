import os
import sys
import glob
import tensorflow as tf
import numpy as np
import time
from abc import ABC, abstractmethod

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

SENSOR_TICK = 0.0  # 0 means as fast as possible
SAVE_EVERY_NTH_FRAME = 50
CONTROL_EVERY_NTH_FRAME = 8
USE_LAST_N_FRAMES = 1
# input processing
TRIM_TOP_PX = 32
GRAYSCALE = True
# output processing
STEER_NORM_FACTOR = 0.05


def to_bgra_array(image):
    """Convert a CARLA raw image to a BGRA numpy array."""
    if not isinstance(image, carla.Image):
        raise ValueError("Argument must be a carla.sensor.Image")
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    return array


def to_rgb_array(image):
    """Convert a CARLA raw image to a RGB numpy array."""
    array = to_bgra_array(image)
    # Convert BGRA to RGB.
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    return array


class AiControl(ABC):
    def __init__(self):
        self.vehicle = None
        self.vehicle_control = None
        self.switch_on = False
        self.default_velocity = 17  # m/s

    def register_vehicle_control(self, vehicle, vehicle_control):
        self.vehicle = vehicle
        self.vehicle_control = vehicle_control
        self.switch_on = self.vehicle is not None and self.vehicle_control is not None
        print(f"Vehicle AI control {type(self)} switched {'on for:' if self.switch_on else 'off.'}")
        if self.switch_on:
            print(f"\t vehicle: {self.vehicle} \n\t control: {self.vehicle_control}")
        else:
            return
        self.vehicle_control.gear = 1

    def control(self, image):
        if not self.switch_on:
            return
        self.control_implementation(image)

    @abstractmethod
    def control_implementation(self, image):
        raise NotImplementedError


class ConstantVelocityControl (AiControl):
    def __init__(self, velocity_kph=60):
        super().__init__()
        self.default_velocity = velocity_kph / 3.6

    def register_vehicle_control(self, vehicle, vehicle_control):
        old_vehicle_ref = self.vehicle
        super().register_vehicle_control(vehicle, vehicle_control)
        if self.switch_on:
            self.vehicle.enable_constant_velocity(carla.Vector3D(self.default_velocity, 0, 0))
        elif old_vehicle_ref is not None:
            old_vehicle_ref.disable_constant_velocity()

    @abstractmethod
    def control_implementation(self, image):
        raise NotImplementedError


class DummyControl (AiControl):
    def control_implementation(self, image):
        self.vehicle_control.throttle = 0.77
        self.vehicle_control.steer = 0.05


class DummyCVControl (ConstantVelocityControl
                      ):
    def control_implementation(self, image):
        self.vehicle_control.steer = 0.02
        self.vehicle_control.throttle = 0.0


class SimpleCVControl (ConstantVelocityControl):
    def __init__(self, velocity_kph=10):
        super().__init__(velocity_kph)
        if not tf.config.get_visible_devices("GPU"):
            print("Warning! No GPU found! (maybe install CUDA, cuDNN, set environment variable LD_LIBRARY_PATH)")
        self.tf_model = tf.keras.models.load_model("TRAININGOLD3.h5")
        self.counter = 0
        self.images = []

    def control_implementation(self, image):
        if (self.counter-1) % CONTROL_EVERY_NTH_FRAME >= CONTROL_EVERY_NTH_FRAME - USE_LAST_N_FRAMES:
            image_arr = to_rgb_array(image)[TRIM_TOP_PX:, :, :] / 255.0
            if GRAYSCALE:
                image_arr = np.mean(image_arr, axis=2, keepdims=True)
            self.images.append(image_arr)
        if self.counter % CONTROL_EVERY_NTH_FRAME == 0 and self.images:
            start_time = time.time()
            input_image_arr = np.array(self.images[-USE_LAST_N_FRAMES:])
            output_steer_vector = self.tf_model.predict(input_image_arr)
            output_steer = np.average(output_steer_vector, axis=0)[0]
            self.vehicle_control.steer = output_steer * STEER_NORM_FACTOR
            print(f"Steering {self.vehicle_control.steer:7.3f} in {time.time() - start_time:5.3f} s")
            self.images.clear()
        self.counter += 1
