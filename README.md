# Glov5: Object Detection and Control for Unitree GO2 Robot

This project is designed for detecting and tracking objects, specifically balloons, using a combination of machine vision and robot control. The system utilizes deep learning models to identify objects in real-time and enables the Unitree GO2 robot to track their location and adjust its position accordingly. The project is built using the Unitree Robotics SDK and leverages PyTorch for object detection.

## Features
- **Object Detection**: Uses a pre-trained deep learning model to detect balloons (or other objects) from camera feeds.
- **Real-Time Tracking**: Continuously tracks the detected object, calculating its position relative to a defined center.
- **Robot Control**: Integrates with the Unitree GO2 robot, allowing it to move and adjust its orientation based on the object's position.
- **Laser Control**: Activates a laser when the object reaches a predefined position for further actions (e.g., marking or targeting).

## Installation

### Prerequisites
- Python 3.x
- PyTorch (for deep learning model inference)
- OpenCV (for image processing)
- Unitree SDK (for robot control)
- Serial library (for laser control)

### Install dependencies:
```bash
pip install torch torchvision opencv-python pyserial
```

### Setup Unitree SDK
Ensure that you have the Unitree Robotics SDK installed and set up for Python. Follow the [Unitree SDK documentation](https://github.com/UnitreeRobotics) to complete the setup.

### Clone the repository:
```bash
git clone https://github.com/yourusername/glov5.git
cd glov5
```

## Usage

To run the object detection and tracking system:

1. Initialize the robot and video clients.
2. Load the pre-trained model for object detection (balloon detection).
3. Start the object detection loop and control the robot based on the detection results.

### Start the project:
```bash
python glov5/sport.py
```

This will start the process of detecting and tracking the object. The robot will adjust its movements based on the object's location.

## Code Structure

- `glov5/sport.py`: Main file for controlling the robot and interacting with the laser system. It processes object detection results and adjusts the robot's movements.
- `glov5/detect.py`: File responsible for handling the object detection model and running inference on camera frames.
- `balloon60.pt`: Pre-trained model file for object detection.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
- **Team Members**: I would like to express my gratitude to my team members: **He Shuheng**, **Zhao Junda**, **Yang Tiancheng**, **Chen Chen**, and **Zhang Xinwei** for their hard work and contributions to the project.
- **Special Thanks**: We would like to thank **Teaching Assistant Chen Shiyi**, **Professor Zhang Chun**, and the **School of Integrated Circuit, Tsinghua University** for providing experimental equipment, facilities, and timely assistance throughout the project.