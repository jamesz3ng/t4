import os
from glob import glob

from setuptools import find_packages, setup

package_name = "turtlebot_sensor"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            os.path.join("share", package_name, "launch"),
            glob(os.path.join("launch", "*launch.[pxy][yma]*")),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="anyone",
    maintainer_email="anyone@todo.todo",
    description="TODO: Package description",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        'console_scripts': [
            'sync_node = turtlebot_sensor.sync_node:main',
            'explorer = turtlebot_sensor.explorer:main',  # Change this line to match your file name
            'plan_node = turtlebot_sensor.plan_node:main',
            'value_test = turtlebot_sensor.hsv_tuner:main'
        ],
    },
)