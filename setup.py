from setuptools import find_packages, setup

package_name = 'module4_perception'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ajay',
    maintainer_email='ajay@drone.dev',
    description='Phase 5 real 3D perception: stereo SGM depth + point cloud + obstacle extraction from the OAK-D Pro W pair.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'stereo_depth_node = module4_perception.stereo_depth_node:main',
            'obstacle_extractor_node = module4_perception.obstacle_extractor_node:main',
            'yolo_detector_node = module4_perception.yolo_detector_node:main',
            'fusion_node = module4_perception.fusion_node:main',
            'dataset_recorder = module4_perception.dataset_recorder:main',
        ],
    },
)
