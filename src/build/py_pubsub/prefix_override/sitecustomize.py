import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/afs/ec.auckland.ac.nz/users/j/z/jzen379/unixhome/ros2_ws/src/install/py_pubsub'
