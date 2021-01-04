from setuptools import setup, find_packages
setup(
    name="tflite_posenet",
    version="0.1",
    packages=['tflite_posenet'],
    install_requires=['tflite_runtime', 'pillow', 'numpy'],

    # metadata to display on PyPI
    author="Ricardo de Azambuja",
    author_email="ricardo.azambuja@gmail.com",
    description="Python library that uses a ready-made TFLite model and spits poses",
    keywords="TensorFlow TFLite Runtime PoseNet",
    url="https://github.com/ricardodeazambuja/PoseNet_TFLite_Runtime",
    classifiers=[
        'Programming Language :: Python :: 3 :: Only' # https://pypi.org/classifiers/
    ]
)

# https://setuptools.readthedocs.io/en/latest/setuptools.html