# Color Attenuation Prior Dehazing
An implementation of the single image dehazing algorithm proposed in the paper [A Fast Single Image Haze Removal Algorithm using Color Attenuation Prior](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7128396) by Qingsong Zhu, Jiaming Mai and Ling Shao. 

## Prerequisites
- OpenCV-2.4
- CMake-3.5

## Running
To compile video dehazing
```cmd
g++ video_main.cpp CAP.cpp guidedfilter.cpp -o dehaze_video `pkg-config --cflags --libs opencv4`
```

To run
```cmd
./dehaze_video path/to/video.mp4
```

To compile single image dehazing
```cmd
g++ main.cpp CAP.cpp guidedfilter.cpp -o dehaze `pkg-config --cflags --libs opencv4`
```

To run single image dehazing
```cmd
./dehazing path/to/file.png
```
## References
- [JiamingMai/Color-Attenuation-Prior-Dehazing](https://github.com/JiamingMai/Color-Attenuation-Prior-Dehazing)
- [atilimcetin/guided-filter](https://github.com/atilimcetin/guided-filter)
