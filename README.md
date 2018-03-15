It seems as though TensorFlow's repositories don't support the TX2, so follow these instructions if you're using the TX2: https://github.com/jetsonhacks/installTensorFlowJetsonTX
Install the wheel file for Python 2.7 because that's what ROS supports
Follow this to install TensorFlow Object Detection: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md

But instead put a line in the form of this into your .bashrc file:

export PYTHONPATH=$PYTHONPATH:/home/[USERNAME]/git/models/research/:/home/[USERNAME]/git/models/research/slim/


You want to install the models repo into /home/[USERNAME]/git/



Install this: https://github.com/stevenlovegrove/Pangolin


Must install ROS. Also, I recommend you use "catkin build" rather than "catkin_make": http://catkin-tools.readthedocs.io/en/latest/installing.html



Other prereqs:
sudo apt install ros-kinetic-ros-control ros-kinetic-ros-controllers ros-kinetic-gazebo-ros ros-kinetic-gazebo-ros-control 


Change the CPU architecture at the bottom of car_localisation/orbslam2/CMakeLists.txt, otherwise you'll get "error adding symbols: File in wrong format" while compiling. If you want to add a new architecture, you'll need to download ORB-SLAM2 (https://github.com/raulmur/ORB_SLAM2), compile g2o and DBoW2 in the ThirdParty folder, and copy their .so files across.


Compiling them is as easy as going into their folder, entering "cmake ." then "make". Their .so files are found in their lib folders.
