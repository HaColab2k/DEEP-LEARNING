# PID Control Project
Self-Driving Car Engineer Nanodegree Program

---
## PID Control

PID control, which stands for Proportional-Integral-Derivative control, is a widely used feedback control algorithm in various engineering and industrial applications. It is designed to regulate a system's output by adjusting a control variable based on the error between the desired setpoint and the actual process variable.

### Components of PID Control

Proportional (P) Term:
The proportional term produces an output proportional to the current error. It aims to reduce the error by applying a control action that is directly proportional to the error magnitude. A higher proportional gain results in a stronger response but can lead to overshooting and instability.

Integral (I) Term:
The integral term accounts for the accumulated error over time. It helps in eliminating steady-state errors caused by small, persistent errors that the proportional term alone cannot correct. The integral term adds a corrective action that increases with the duration of the error.

Derivative (D) Term:
The derivative term anticipates future error by considering the rate of change of the error. It provides a damping effect to counteract rapid changes in the process variable, reducing overshooting and oscillations. The derivative term can also enhance system stability.

Working of PID Control:
PID control operates by continuously calculating an output control signal based on the proportional, integral, and derivative terms and applying it to the control mechanism. The control signal influences the process variable, which is then measured and compared to the desired setpoint. The resulting error is used to calculate the PID output, aiming to minimize the error and maintain the process variable close to the setpoint.

Tuning PID Controllers:
The effectiveness of a PID controller heavily depends on tuning its three components: proportional, integral, and derivative gains. Proper tuning ensures stable, fast, and accurate control responses. Tuning methods include manual tuning, Ziegler-Nichols method, Cohen-Coon method, and advanced methods that employ optimization techniques.

### Advantages of PID Control

Simple and widely understood algorithm.
Effective for a wide range of control problems.
Suitable for systems with varying dynamics and disturbances.
Can be used in combination with other control strategies.
Limitations of PID Control:

PID control might struggle with complex systems and nonlinearities.
Requires careful tuning to achieve optimal performance.
Not well-suited for processes with large time delays.
Applications:
PID control finds application in various industries, including manufacturing, robotics, automotive control, temperature control, motor control, chemical processes, and more. It is used in scenarios where precise control of processes is necessary to achieve desired outcomes.

## Cross Track Error (CTE)

Cross Track Error (CTE), also known as lateral error or lateral deviation, is a critical concept in navigation, control systems, robotics, and vehicle dynamics. It represents the perpendicular distance between the desired path (reference trajectory) and the actual position of a vehicle or system. CTE is particularly relevant when dealing with systems that follow a predefined trajectory, such as autonomous vehicles, aircraft, ships, and industrial robots.

## Understanding Cross Track Error:
Imagine a vehicle or robot moving along a desired path, such as a road or a track. The ideal path is represented by a reference trajectory. The cross track error is the distance between the vehicle's actual position and the reference trajectory, measured perpendicular to the trajectory.

Mathematically, if ![image](https://github.com/HaColab2k/SDC/assets/127838132/3f6c4952-0095-4a69-9f6e-26ca9ca97e87)
 represents a point on the reference trajectory and ![image](https://github.com/HaColab2k/SDC/assets/127838132/8916a975-f26f-4086-b561-95efbdd6751b)
 represents the actual position of the vehicle, the cross track error CTE is given by:
![image](https://github.com/HaColab2k/SDC/assets/127838132/3b8e2f12-9a8f-4110-b2ec-a1303b2d1987)

## Dependencies

* cmake >= 3.5
 * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1(mac, linux), 3.81(Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools]((https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)
* [uWebSockets](https://github.com/uWebSockets/uWebSockets)
  * Run `./install-ubuntu.sh`.
  * If you install from source, checkout to commit `e94b6e1`, i.e.
    ```
    git clone https://github.com/uWebSockets/uWebSockets 
    cd uWebSockets
    git checkout e94b6e1
    ```
    Some function signatures have changed in v0.14.x. See [this PR](https://github.com/udacity/CarND-MPC-Project/pull/3) for more details.
* Simulator. You can download these from the [project intro page](https://github.com/udacity/self-driving-car-sim/releases) in the classroom.

There's an experimental patch for windows in this [PR](https://github.com/udacity/CarND-PID-Control-Project/pull/3)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./pid`. 

## Simulator
You can download the simualtor [here](https://github.com/udacity/self-driving-car-sim/releases). 
## Video
You can watch the result [here](https://www.youtube.com/watch?v=PdtyBNztSsk).

