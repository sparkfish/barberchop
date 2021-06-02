import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Barberchop",
    version="1.0.0",
    author="Sparkfish LLC",
    author_email="packages@sparkfish.com",
    description="Segments, classifies and extracts barcode images from larger images containing one or more barcodes.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sparkfish/barberchop",
    project_urls={
        "Bug Tracker": "https://github.com/sparkfish/barberchop/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7",    
    install_requires=[
        "yolov5 == 4.0.14"
    ],
)